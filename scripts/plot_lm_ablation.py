#!/usr/bin/env python3
import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


def read_rows(checkpoint_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in sorted(checkpoint_dir.glob("lm_*/training_log.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("experiment_name") == "experiment_name":
                    continue
                try:
                    row["_param_count"] = int(row["param_count"])
                    row["_step"] = int(row["step"])
                    row["_epoch"] = int(row["epoch"])
                    row["_tokens_seen"] = int(row["tokens_seen"])
                    row["_val_loss"] = float(row["val_loss"])
                    row["_lr"] = float(row["lr"])
                    row["_timestamp"] = datetime.fromisoformat(row["timestamp"])
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append(row)
    return sorted(rows, key=lambda r: (r["experiment_name"], r["_step"]))


def grouped(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(row["experiment_name"], []).append(row)
    return dict(sorted(out.items(), key=lambda item: item[1][0]["_param_count"]))


def label(name: str, rows: list[dict[str, str]]) -> str:
    return f"{name} ({rows[0]['_param_count'] / 1_000_000:.1f}M)"


def plot_val_loss_by_tokens(groups: dict[str, list[dict[str, str]]], out_path: Path) -> None:
    plt.figure(figsize=(9, 5.2), dpi=160)
    for name, rows in groups.items():
        xs = [r["_tokens_seen"] / 1_000_000 for r in rows]
        ys = [r["_val_loss"] for r in rows]
        plt.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=label(name, rows))
    plt.xlabel("Tokens Seen (millions)")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs Tokens Seen")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_val_loss_by_epoch(groups: dict[str, list[dict[str, str]]], out_path: Path) -> None:
    plt.figure(figsize=(9, 5.2), dpi=160)
    for name, rows in groups.items():
        xs = [r["_epoch"] + (r["_step"] % 2500) / 2500 for r in rows]
        ys = [r["_val_loss"] for r in rows]
        plt.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=label(name, rows))
    plt.xlabel("Epoch Progress")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs Epoch")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_summary(groups: dict[str, list[dict[str, str]]], out_path: Path) -> None:
    fieldnames = [
        "experiment",
        "params",
        "first_val_loss",
        "final_val_loss",
        "val_loss_drop",
        "tokens_seen",
        "tokens_per_param",
        "minutes",
        "tokens_per_second",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, rows in groups.items():
            first = rows[0]
            last = rows[-1]
            seconds = max(1.0, (last["_timestamp"] - first["_timestamp"]).total_seconds())
            writer.writerow({
                "experiment": name,
                "params": last["_param_count"],
                "first_val_loss": f"{first['_val_loss']:.4f}",
                "final_val_loss": f"{last['_val_loss']:.4f}",
                "val_loss_drop": f"{first['_val_loss'] - last['_val_loss']:.4f}",
                "tokens_seen": last["_tokens_seen"],
                "tokens_per_param": f"{last['_tokens_seen'] / last['_param_count']:.2f}",
                "minutes": f"{seconds / 60:.1f}",
                "tokens_per_second": f"{last['_tokens_seen'] / seconds:.0f}",
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--out_dir", default="docs/plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(Path(args.checkpoint_dir))
    if not rows:
        raise SystemExit("No training_log.csv rows found")
    groups = grouped(rows)
    plot_val_loss_by_tokens(groups, out_dir / "lm_val_loss_vs_tokens.png")
    plot_val_loss_by_epoch(groups, out_dir / "lm_val_loss_vs_epoch.png")
    write_summary(groups, out_dir / "lm_ablation_summary.csv")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
