import argparse
import json
import os
import re
from collections import Counter

import sentencepiece as spm
import torch
from datasets import concatenate_datasets, load_from_disk
from safetensors.torch import load_file, load_model as load_safetensors_model
from tqdm import tqdm

from model import TinyDecoder
from model_config import TinyDecoderConfig


def norm(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).lower().strip())


def load_model(path: str, device: torch.device) -> TinyDecoder:
    cfg_path = os.path.splitext(path)[0] + "_config.json"
    cfg = TinyDecoderConfig.load_json(cfg_path) if os.path.exists(cfg_path) else TinyDecoderConfig()
    model = TinyDecoder(cfg).to(device)
    if path.endswith(".safetensors"):
        raw = load_file(path, device="cpu")
        if any(k.endswith("__scale") for k in raw):
            state = {}
            for key, tensor in raw.items():
                if key.endswith("__scale"):
                    continue
                scale = raw.get(key + "__scale")
                state[key] = tensor.float() * float(scale.item()) if scale is not None and tensor.dtype == torch.int8 else tensor
            model.load_state_dict(state, strict=False)
        else:
            load_safetensors_model(model, path, strict=False)
        return model.eval()
    else:
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "config" in ckpt:
            cfg = TinyDecoderConfig(**ckpt["config"])
            model = TinyDecoder(cfg).to(device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def encode_input(sp, text: str) -> list[int]:
    task, rest = text.split(" ", 1)
    tid = {"[OCR]": 4, "[INSTR]": 5, "[AUTO]": 6}[task]
    return [tid] + sp.encode(rest, out_type=int)


@torch.no_grad()
def greedy(model, sp, text: str, max_new: int, device) -> str:
    ids = torch.tensor([encode_input(sp, text)], dtype=torch.long, device=device)
    for _ in range(max_new):
        logits, _ = model(ids)
        nxt = int(torch.argmax(logits[0, -1]).item())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        if nxt == 3:
            break
    return sp.decode([x for x in ids[0].tolist()[len(encode_input(sp, text)) :] if x != 3]).strip()


@torch.no_grad()
def beam(model, sp, text: str, max_new: int, device, beam_size: int = 3) -> list[str]:
    start = encode_input(sp, text)
    beams = [(start, 0.0)]
    for _ in range(max_new):
        cand = []
        for ids, score in beams:
            if ids[-1] == 3:
                cand.append((ids, score))
                continue
            t = torch.tensor([ids], dtype=torch.long, device=device)
            logits, _ = model(t)
            logp = torch.log_softmax(logits[0, -1], dim=-1)
            vals, idxs = torch.topk(logp, beam_size)
            for v, idx in zip(vals.tolist(), idxs.tolist()):
                cand.append((ids + [int(idx)], score + float(v)))
        beams = sorted(cand, key=lambda x: x[1] / max(1, len(x[0]) - len(start)), reverse=True)[:beam_size]
    return [sp.decode([x for x in ids[len(start) :] if x != 3]).strip() for ids, _ in beams]


def ocr_metrics(model, sp, data_dir, device):
    ds = concatenate_datasets([load_from_disk(os.path.join(data_dir, "stage3", n, "test")) for n in ["cord-v2", "sroie", "funsd"]])
    tp = pred_n = gold_n = 0
    for row in tqdm(ds, desc="eval ocr"):
        pred = greedy(model, sp, row["input_text"], 200, device)
        try:
            pobj = json.loads(pred)
        except Exception:
            pobj = {}
        gold = json.loads(row["target_text"].replace("<eos>", "").strip())
        pset = {(norm(k), norm(v)) for k, v in pobj.items()}
        gset = {(norm(k), norm(v)) for k, v in gold.items()}
        tp += len(pset & gset)
        pred_n += len(pset)
        gold_n += len(gset)
    prec = tp / pred_n if pred_n else 0.0
    rec = tp / gold_n if gold_n else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "F1": f1}


def parse_instr(text: str):
    parts = text.strip().split()
    action = ""
    slots = set()
    for p in parts:
        if p.startswith("ACTION:"):
            action = p.split(":", 1)[1]
        elif ":" in p:
            k, v = p.split(":", 1)
            slots.add(f"{k.lower()}:{v.lower().replace(' ', '_')}")
    return action, slots


def instr_metrics(model, sp, data_dir, device):
    ds = concatenate_datasets([load_from_disk(os.path.join(data_dir, "stage3", n, "test")) for n in ["snips", "atis", "dolly_filtered"]])
    act_ok = act_total = slot_tp = slot_p = slot_g = 0
    for row in tqdm(ds, desc="eval instr"):
        pred = greedy(model, sp, row["input_text"], 64, device)
        pa, ps = parse_instr(pred)
        ga, gs = parse_instr(row["target_text"].replace("<eos>", ""))
        if ga:
            act_total += 1
            act_ok += int(pa == ga)
        slot_tp += len(ps & gs)
        slot_p += len(ps)
        slot_g += len(gs)
    return {
        "action_accuracy": act_ok / act_total if act_total else 0.0,
        "slot_precision": slot_tp / slot_p if slot_p else 0.0,
        "slot_recall": slot_tp / slot_g if slot_g else 0.0,
    }


def auto_metrics(model, sp, data_dir, device):
    ds = load_from_disk(os.path.join(data_dir, "stage3", "msmarco_auto", "test"))
    top1 = top3 = total = 0
    for row in tqdm(ds, desc="eval auto"):
        gold = row["target_text"].replace("<eos>", "").strip().split()[0].lower()
        outs = beam(model, sp, row["input_text"], 5, device, 3)
        words = [o.split()[0].lower() if o.split() else "" for o in outs]
        top1 += int(words and words[0] == gold)
        top3 += int(gold in words[:3])
        total += 1
    return {"top1_accuracy": top1 / total if total else 0.0, "top3_accuracy": top3 / total if total else 0.0}


def print_table(rows):
    print("+---------+------------------+---------+")
    print("| Task    | Metric           | Score   |")
    print("+---------+------------------+---------+")
    for task, metric, score in rows:
        print(f"| {task:<7} | {metric:<16} | {score:>7.3f} |")
    print("+---------+------------------+---------+")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--task", choices=["ocr", "instr", "auto", "all"], required=True)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    model = load_model(args.checkpoint, device)
    rows = []
    if args.task in {"ocr", "all"}:
        rows.extend(("[OCR]", k, v) for k, v in ocr_metrics(model, sp, args.data_dir, device).items())
    if args.task in {"instr", "all"}:
        rows.extend(("[INSTR]", k, v) for k, v in instr_metrics(model, sp, args.data_dir, device).items())
    if args.task in {"auto", "all"}:
        rows.extend(("[AUTO]", k, v) for k, v in auto_metrics(model, sp, args.data_dir, device).items())
    print_table(rows)
    print("=== DONE: evaluate.py ===")


if __name__ == "__main__":
    main()
