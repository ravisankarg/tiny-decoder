import argparse
import json
import os

import torch
from safetensors.torch import save_file

from model import TinyDecoder
from model_config import TinyDecoderConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--config_path", default="model_config.json")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    if os.path.exists(args.output_path):
        print(f"{args.output_path} exists; skipping")
        print("=== DONE: export_safetensors.py ===")
        return
    cfg = TinyDecoderConfig.load_json(args.config_path)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = TinyDecoder(cfg)
    model.load_state_dict(state, strict=False)
    quantized = {}
    for key, tensor in model.state_dict().items():
        if key == "lm_head.weight":
            continue
        cpu = tensor.detach().cpu()
        if cpu.is_floating_point():
            max_abs = float(cpu.abs().max().item())
            scale = max(max_abs / 127.0, 1e-8)
            quantized[key] = torch.clamp(torch.round(cpu / scale), -127, 127).to(torch.int8)
            quantized[key + "__scale"] = torch.tensor([scale], dtype=torch.float32)
        else:
            quantized[key] = cpu
    save_file(quantized, args.output_path, metadata={"format": "tiny_decoder_int8_safetensors"})
    cfg_path = os.path.splitext(args.output_path)[0] + "_config.json"
    cfg.save_json(cfg_path)
    mb = os.path.getsize(args.output_path) / (1024 * 1024)
    print(f"Exported safetensors size: {mb:.1f} MB")
    if mb > 25:
        raise RuntimeError(f"Exported safetensors is {mb:.1f} MB, above the 25 MB limit")
    print("=== DONE: export_safetensors.py ===")


if __name__ == "__main__":
    main()
