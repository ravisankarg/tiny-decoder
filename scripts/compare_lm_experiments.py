#!/usr/bin/env python3
import csv
import glob
import json
import os


def read_last_csv(path: str) -> dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def main() -> None:
    rows = []
    for meta_path in sorted(glob.glob("checkpoints/*/metadata.json")):
        out_dir = os.path.dirname(meta_path)
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        last = read_last_csv(os.path.join(out_dir, "training_log.csv"))
        rows.append({
            "experiment": meta.get("experiment_name", os.path.basename(out_dir)),
            "params": meta.get("param_count", ""),
            "dataset": (meta.get("dataset") or {}).get("dataset_name", ""),
            "epochs": meta.get("epochs", ""),
            "batch": meta.get("batch_size", ""),
            "accum": meta.get("grad_accum_steps", ""),
            "tokens_per_step": meta.get("effective_batch_tokens", ""),
            "last_step": last.get("step", ""),
            "tokens_seen": last.get("tokens_seen", ""),
            "train_loss": last.get("train_loss", ""),
            "val_loss": last.get("val_loss", ""),
        })
    if not rows:
        print("No experiment metadata found under checkpoints/*/metadata.json")
        return
    headers = list(rows[0])
    widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in headers}
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(str(row[h]).ljust(widths[h]) for h in headers))


if __name__ == "__main__":
    main()
