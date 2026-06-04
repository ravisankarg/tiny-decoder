import argparse
import csv
import json
import math
import os
import random
from datetime import datetime
from typing import Iterable

import sentencepiece as spm
import torch
from datasets import concatenate_datasets, load_from_disk
from safetensors.torch import save_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import TinyDecoder
from model_config import TinyDecoderConfig
from model_config import estimate_param_count
from prepare_datasets import augment_ocr, encode_pair


DEFAULTS = {
    0: dict(epochs=3, batch_size=8, grad_accum_steps=2, lr=2e-4, warmup_steps=1000, min_lr=1e-5, weight_decay=0.01, label_smoothing=0.0, val_every=500, save_every=2000),
    1: dict(epochs=3, batch_size=8, grad_accum_steps=4, lr=3e-4, warmup_steps=500, min_lr=1e-5, weight_decay=0.1, label_smoothing=0.1, val_every=500, save_every=2000),
    2: dict(epochs=5, batch_size=4, grad_accum_steps=4, lr=1e-4, warmup_steps=200, min_lr=3e-6, weight_decay=0.1, label_smoothing=0.1, val_every=500, save_every=1000),
    3: dict(epochs=10, batch_size=4, grad_accum_steps=4, lr=5e-5, warmup_steps=100, min_lr=1e-6, weight_decay=0.05, label_smoothing=0.05, val_every=300, save_every=1000),
}


class ArrowTorchDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.ds[int(idx)]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
        }


class RuntimeOCRDataset(Dataset):
    def __init__(self, ds, sp: spm.SentencePieceProcessor, max_seq_len: int):
        self.ds = ds
        self.sp = sp
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.ds[int(idx)]
        input_text = row["input_text"]
        task, rest = input_text.split(" ", 1)
        enc = encode_pair(self.sp, task + " " + augment_ocr(rest, 0.3), row["target_text"], self.max_seq_len)
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "labels": torch.tensor(enc["labels"], dtype=torch.long),
        }


class CombinedDataset(Dataset):
    def __init__(self, ocr_sets: list[Dataset], instr_sets: list[Dataset], auto_set: Dataset, length: int):
        groups = [ocr_sets, instr_sets, [auto_set]]
        weights = [2, 1, 2]
        self.groups = [[ds for ds in group if len(ds) > 0] for group in groups]
        self.weights = [w if group else 0 for w, group in zip(weights, self.groups)]
        if sum(self.weights) == 0:
            raise ValueError("CombinedDataset has no non-empty task datasets")
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        del idx
        group = random.choices([0, 1, 2], weights=self.weights, k=1)[0]
        ds = random.choice(self.groups[group])
        return ds[random.randrange(len(ds))]


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(int(x["input_ids"].numel()) for x in batch)
    ids, labels = [], []
    for row in batch:
        row_ids = row["input_ids"]
        row_labels = row["labels"]
        pad = max_len - int(row_ids.numel())
        if pad > 0:
            row_ids = torch.nn.functional.pad(row_ids, (0, pad), value=0)
            row_labels = torch.nn.functional.pad(row_labels, (0, pad), value=-100)
        ids.append(row_ids)
        labels.append(row_labels)
    return {
        "input_ids": torch.stack(ids),
        "labels": torch.stack(labels),
    }


def find_tokenizer(data_dir: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates = [
        os.path.join("tokenizer", "tokenizer.model"),
        os.path.join(os.path.dirname(os.path.abspath(data_dir)), "tokenizer", "tokenizer.model"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Stage 3 runtime OCR augmentation needs tokenizer.model; pass --tokenizer_path")


def load_train_val(stage: int, data_dir: str, tokenizer_path: str | None, max_seq_len: int):
    if stage == 0:
        lm_name = os.environ.get("LM_DATASET_NAME", "lm_prose")
        return ArrowTorchDataset(load_from_disk(os.path.join(data_dir, lm_name, "train"))), [ArrowTorchDataset(load_from_disk(os.path.join(data_dir, lm_name, "val")))]
    if stage == 1:
        return ArrowTorchDataset(load_from_disk(os.path.join(data_dir, "stage1", "train"))), [ArrowTorchDataset(load_from_disk(os.path.join(data_dir, "stage1", "val")))]
    if stage == 2:
        names = ["conll2003", "cord-v2", "sroie", "funsd", "snips", "atis", "dolly_filtered"]
        train = concatenate_datasets([load_from_disk(os.path.join(data_dir, "stage2", n, "train")) for n in names])
        vals = [ArrowTorchDataset(load_from_disk(os.path.join(data_dir, "stage2", n, "test"))) for n in names]
        return ArrowTorchDataset(train), vals
    sp = spm.SentencePieceProcessor(model_file=find_tokenizer(data_dir, tokenizer_path))
    ocr = [RuntimeOCRDataset(load_from_disk(os.path.join(data_dir, "stage3", n, "train")), sp, max_seq_len) for n in ["cord-v2", "sroie", "funsd"]]
    instr = [ArrowTorchDataset(load_from_disk(os.path.join(data_dir, "stage3", n, "train"))) for n in ["snips", "atis", "dolly_filtered"]]
    auto = ArrowTorchDataset(load_from_disk(os.path.join(data_dir, "stage3", "msmarco_auto", "train")))
    length = 5 * max(sum(len(x) for x in ocr), sum(len(x) for x in instr), len(auto))
    train = CombinedDataset(ocr, instr, auto, length)
    vals = [ArrowTorchDataset(load_from_disk(os.path.join(data_dir, "stage3", n, "test"))) for n in ["cord-v2", "sroie", "funsd", "snips", "atis", "dolly_filtered", "msmarco_auto"]]
    return train, vals


def lm_dataset_name() -> str:
    return os.environ.get("LM_DATASET_NAME", "lm_prose")


def dataset_fingerprint(data_dir: str, stage: int) -> dict[str, object]:
    if stage == 0:
        name = lm_dataset_name()
        train_path = os.path.join(data_dir, name, "train")
        val_path = os.path.join(data_dir, name, "val")
        train = load_from_disk(train_path)
        val = load_from_disk(val_path)
        return {
            "dataset_name": name,
            "train_rows": len(train),
            "val_rows": len(val),
            "train_fingerprint": getattr(train, "_fingerprint", ""),
            "val_fingerprint": getattr(val, "_fingerprint", ""),
        }
    return {"dataset_name": f"stage{stage}"}


def lr_at(step: int, total_steps: int, base_lr: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def resolve_precision(args, device: torch.device) -> str:
    precision = args.precision or ("fp16" if args.fp16 else "fp32")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 precision was requested, but torch.cuda.is_bf16_supported() is False.")
    if precision in {"bf16", "fp16"} and device.type != "cuda":
        print(f"WARNING: {precision} requested without CUDA; falling back to fp32")
        return "fp32"
    return precision


def autocast_kwargs(device: torch.device, precision: str) -> dict[str, object]:
    if device.type != "cuda" or precision == "fp32":
        return {"enabled": False}
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return {"enabled": True, "dtype": dtype}


@torch.no_grad()
def validate(model, loaders, device, precision: str, label_smoothing: float) -> float:
    model.eval()
    losses = []
    for loader in loaders:
        local = []
        for batch in loader:
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast(**autocast_kwargs(device, precision)):
                _, loss = model(ids, labels=labels, checkpoint_blocks=False, label_smoothing=label_smoothing)
            local.append(float(loss.item()))
            if len(local) >= 50:
                break
        if local:
            losses.append(sum(local) / len(local))
    model.train()
    return sum(losses) / max(1, len(losses))


def save_checkpoint(path: str, model, optimizer, scaler, step: int, epoch: int, val_loss: float, metadata: dict[str, object] | None = None) -> None:
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "model_state_dict": raw.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "step": step,
            "epoch": epoch,
            "val_loss": val_loss,
            "config": raw.config.__dict__,
            "metadata": metadata or {},
        },
        path,
    )


def save_safetensors(path: str, model) -> None:
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    save_model(raw, path)
    raw.config.save_json(os.path.splitext(path)[0] + "_config.json")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, choices=[0, 1, 2, 3], required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--config_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--tokenizer_path")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--grad_accum_steps", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--warmup_steps", type=int)
    p.add_argument("--max_seq_len", type=int)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], help="Training precision. Prefer bf16 on GPUs that support it.")
    p.add_argument("--fp16", action="store_true", help="Deprecated alias for --precision fp16.")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max_steps", type=int, help="Stop after this many optimizer steps; useful for smoke tests.")
    p.add_argument("--no_compile", action="store_true", help="Disable torch.compile.")
    p.add_argument("--wandb_project", help="Enable W&B logging with this project name.")
    p.add_argument("--require_cuda", action="store_true", help="Exit immediately unless CUDA is available.")
    p.add_argument("--reset_optimizer_state", action="store_true", help="Load checkpoint model weights but start optimizer/step/epoch from scratch.")
    p.add_argument("--experiment_name", help="Stable experiment name for comparing model sizes.")
    p.add_argument("--notes", default="", help="Short free-form note written to metadata.json.")
    args = p.parse_args()

    d = DEFAULTS[args.stage]
    for k in ["epochs", "batch_size", "grad_accum_steps", "lr", "warmup_steps"]:
        if getattr(args, k) is None:
            setattr(args, k, d[k])
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training_log.csv")
    if os.path.exists(log_path) and not args.overwrite and not args.checkpoint:
        print(f"{log_path} exists; pass --overwrite or --checkpoint to continue")
        print("=== DONE: train.py ===")
        return

    config = TinyDecoderConfig.load_json(args.config_path)
    config.max_seq_len = args.max_seq_len or (256 if args.stage in {0, 1} else 512 if args.stage == 2 else 1024)
    param_count = estimate_param_count(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this run, but torch.cuda.is_available() is False.")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
    precision = resolve_precision(args, device)
    print(f"precision={precision}")
    model = TinyDecoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=d["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=precision == "fp16" and device.type == "cuda")
    global_step, start_epoch, best_val = 0, 0, float("inf")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt and not args.reset_optimizer_state:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            global_step = int(ckpt.get("step", 0))
            start_epoch = int(ckpt.get("epoch", 0))
            best_val = float(ckpt.get("val_loss", best_val))

    if not args.no_compile:
        try:
            v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
            if v >= (2, 0):
                model = torch.compile(model)
        except Exception as exc:
            print(f"WARNING: torch.compile unavailable: {exc}")

    train_ds, val_ds = load_train_val(args.stage, args.data_dir, args.tokenizer_path, config.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    val_loaders = [DataLoader(v, batch_size=args.batch_size, shuffle=False, collate_fn=collate) for v in val_ds]
    total_steps = max(1, (len(train_loader) * args.epochs) // args.grad_accum_steps)
    effective_batch_tokens = args.batch_size * args.grad_accum_steps * config.max_seq_len
    metadata = {
        "experiment_name": args.experiment_name or os.path.basename(os.path.abspath(args.output_dir)),
        "notes": args.notes,
        "stage": args.stage,
        "config_path": args.config_path,
        "param_count": param_count,
        "max_seq_len": config.max_seq_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_tokens": effective_batch_tokens,
        "lr": args.lr,
        "min_lr": d["min_lr"],
        "warmup_steps": args.warmup_steps,
        "weight_decay": d["weight_decay"],
        "label_smoothing": d["label_smoothing"],
        "precision": precision,
        "fp16": precision == "fp16",
        "bf16": precision == "bf16",
        "max_steps": args.max_steps,
        "dataset": dataset_fingerprint(args.data_dir, args.stage),
        "created_at": datetime.utcnow().isoformat(),
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2, sort_keys=True)
    print("experiment_metadata=" + json.dumps(metadata, sort_keys=True))

    new_log = not os.path.exists(log_path) or args.overwrite
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_name",
                "param_count",
                "dataset_name",
                "step",
                "epoch",
                "tokens_seen",
                "train_loss",
                "val_loss",
                "lr",
                "timestamp",
            ],
        )
        if new_log:
            writer.writeheader()
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, config=vars(args), reinit=True) if args.wandb_project else None
        except Exception:
            run = None
        model.train()
        stop_training = False
        for epoch in range(start_epoch, args.epochs):
            pbar = tqdm(train_loader, desc=f"stage{args.stage} epoch{epoch + 1}")
            running = 0.0
            running_batches = 0
            optimizer.zero_grad(set_to_none=True)
            for i, batch in enumerate(pbar):
                lr = lr_at(global_step, total_steps, args.lr, d["min_lr"], args.warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                with torch.cuda.amp.autocast(**autocast_kwargs(device, precision)):
                    _, loss = model(ids, labels=labels, checkpoint_blocks=True, label_smoothing=d["label_smoothing"])
                    loss = loss / args.grad_accum_steps
                scaler.scale(loss).backward()
                running += float(loss.item()) * args.grad_accum_steps
                running_batches += 1
                if (i + 1) % args.grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if global_step % d["val_every"] == 0:
                        val = validate(model, val_loaders, device, precision, d["label_smoothing"])
                        row = {
                            "experiment_name": metadata["experiment_name"],
                            "param_count": param_count,
                            "dataset_name": metadata["dataset"]["dataset_name"] if isinstance(metadata["dataset"], dict) else "",
                            "step": global_step,
                            "epoch": epoch + 1,
                            "tokens_seen": global_step * effective_batch_tokens,
                            "train_loss": running / max(1, running_batches),
                            "val_loss": val,
                            "lr": lr,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                        writer.writerow(row)
                        f.flush()
                        if run:
                            run.log(row)
                        running = 0.0
                        running_batches = 0
                        if val < best_val:
                            best_val = val
                            save_checkpoint(os.path.join(args.output_dir, "best.pt"), model, optimizer, scaler, global_step, epoch, best_val, metadata)
                            save_safetensors(os.path.join(args.output_dir, "best.safetensors"), model)
                    if global_step % d["save_every"] == 0:
                        save_checkpoint(os.path.join(args.output_dir, f"step_{global_step}.pt"), model, optimizer, scaler, global_step, epoch, best_val, metadata)
                    if args.max_steps and global_step >= args.max_steps:
                        stop_training = True
                        break
                pbar.set_postfix(step=global_step, loss=float(loss.item()) * args.grad_accum_steps)
            save_checkpoint(os.path.join(args.output_dir, f"epoch_{epoch + 1}.pt"), model, optimizer, scaler, global_step, epoch + 1, best_val, metadata)
            save_checkpoint(os.path.join(args.output_dir, "latest.pt"), model, optimizer, scaler, global_step, epoch + 1, best_val, metadata)
            if stop_training:
                break
        if run:
            run.finish()
    print("=== DONE: train.py ===")


if __name__ == "__main__":
    main()
