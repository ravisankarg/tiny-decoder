import argparse
import math
import os
import string

import numpy as np
import sentencepiece as spm
import torch
from safetensors.torch import load_file, load_model as load_safetensors_model

from model import TinyDecoder
from model_config import TinyDecoderConfig


TASKS = {"ocr": ("[OCR]", 4), "instr": ("[INSTR]", 5), "auto": ("[AUTO]", 6)}
JSON_CHARS = set('{}:,"\\\'.-_/ ' + string.digits + string.ascii_letters)


def load_model(path: str, device: torch.device) -> TinyDecoder:
    cfg_path = os.path.splitext(path)[0] + "_config.json"
    cfg = TinyDecoderConfig.load_json(cfg_path) if os.path.exists(cfg_path) else TinyDecoderConfig()
    model = TinyDecoder(cfg).to(device)
    if path.endswith(".pt"):
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "config" in ckpt:
            cfg = TinyDecoderConfig(**ckpt["config"])
            model = TinyDecoder(cfg).to(device)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        model.eval()
        return model
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
    model.eval()
    return model


def encode(sp, task: str, text: str) -> list[int]:
    return [TASKS[task][1]] + sp.encode(text, out_type=int)


def decode_tokens(sp, tokens: list[int]) -> str:
    return sp.decode([t for t in tokens if t != 3]).strip()


def token_allowed(sp, idx: int) -> bool:
    piece = sp.id_to_piece(int(idx)).replace("▁", " ")
    return bool(piece) and all(ch in JSON_CHARS for ch in piece)


@torch.no_grad()
def greedy(model, sp, prompt_ids: list[int], max_new: int, device, constrain_ocr: bool) -> str:
    ids = list(prompt_ids)
    generated = []
    opened = False
    allowed = None
    if constrain_ocr:
        allowed = torch.tensor([token_allowed(sp, i) for i in range(sp.get_piece_size())], dtype=torch.bool, device=device)
    for _ in range(max_new):
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(inp)
        next_logits = logits[0, -1].float()
        if constrain_ocr and opened and allowed is not None:
            next_logits = next_logits.masked_fill(~allowed, -1e9)
        nxt = int(torch.argmax(next_logits).item())
        ids.append(nxt)
        generated.append(nxt)
        piece = sp.id_to_piece(nxt).replace("▁", " ")
        if "{" in piece:
            opened = True
        if nxt == 3:
            break
    return decode_tokens(sp, generated)


@torch.no_grad()
def beam(model, sp, prompt_ids: list[int], max_new: int, device, beam_size: int = 3):
    beams = [(list(prompt_ids), [], 0.0)]
    for _ in range(max_new):
        candidates = []
        for ids, gen, score in beams:
            if gen and gen[-1] == 3:
                candidates.append((ids, gen, score))
                continue
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            logits, _ = model(inp)
            logp = torch.log_softmax(logits[0, -1].float(), dim=-1)
            vals, idxs = torch.topk(logp, beam_size)
            for val, idx in zip(vals.tolist(), idxs.tolist()):
                candidates.append((ids + [int(idx)], gen + [int(idx)], score + float(val)))
        beams = sorted(candidates, key=lambda x: x[2] / max(1, len(x[1])), reverse=True)[:beam_size]
    return [(decode_tokens(sp, gen), score / max(1, len(gen))) for _, gen, score in beams]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--task", choices=["ocr", "instr", "auto"], required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--max_new_tokens", type=int)
    args = p.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = 200 if args.task == "ocr" else 64 if args.task == "instr" else 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    model = load_model(args.model_path, device)
    prompt = encode(sp, args.task, args.input)
    if args.task == "auto":
        for i, (text, score) in enumerate(beam(model, sp, prompt, args.max_new_tokens, device, 3), 1):
            print(f"{i}. {text} score={score:.4f}")
    else:
        print(greedy(model, sp, prompt, args.max_new_tokens, device, args.task == "ocr"))
    print("=== DONE: inference_demo.py ===")


if __name__ == "__main__":
    main()
