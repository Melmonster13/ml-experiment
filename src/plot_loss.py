"""Plot training/validation loss curves from an experiment's metrics.csv.

Usage:
    python src/plot_loss.py
    python src/plot_loss.py --experiment experiments/2026-04-30-1200
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
DEFAULT_SAVE_EVERY = 100


def latest_experiment() -> Path:
    runs = sorted(p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir())
    if not runs:
        raise SystemExit(f"No experiments found in {EXPERIMENTS_DIR}")
    return runs[-1]


def load_metrics(path: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    train: list[tuple[int, float]] = []
    val: list[tuple[int, float]] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            it = int(row["iter"])
            if row.get("train_loss"):
                train.append((it, float(row["train_loss"])))
            if row.get("val_loss"):
                val.append((it, float(row["val_loss"])))
    return train, val


def get_save_every(exp_dir: Path) -> int:
    for name in ("lora_config.json", "lora_config.yaml"):
        p = exp_dir / name
        if not p.exists():
            continue
        try:
            if name.endswith(".json"):
                cfg = json.loads(p.read_text())
            else:
                import yaml
                cfg = yaml.safe_load(p.read_text())
            if cfg and "save_every" in cfg:
                return int(cfg["save_every"])
        except Exception:
            pass
    return DEFAULT_SAVE_EVERY


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=None)
    args = parser.parse_args()

    exp_dir = args.experiment or latest_experiment()
    metrics_path = exp_dir / "metrics.csv"
    if not metrics_path.exists():
        raise SystemExit(f"No metrics.csv in {exp_dir}")

    train, val = load_metrics(metrics_path)
    if not train:
        raise SystemExit("No training rows found in metrics.csv")

    save_every = get_save_every(exp_dir)
    max_iter = max(it for it, _ in train + val) if (train or val) else 0
    checkpoints = list(range(save_every, max_iter + 1, save_every))

    train_x = [it for it, _ in train]
    train_y = [loss for _, loss in train]
    val_x = [it for it, _ in val]
    val_y = [loss for _, loss in val]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: train loss
    ax1.plot(train_x, train_y, color="C0", linewidth=1.5, label="Train loss")
    for c in checkpoints:
        ax1.axvline(c, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: val points over faded train line
    ax2.plot(train_x, train_y, color="C0", linewidth=1.0, alpha=0.25, label="Train loss (faded)")
    if val_x:
        ax2.plot(val_x, val_y, "o-", color="C3", linewidth=1.5, markersize=6, label="Val loss")
    for c in checkpoints:
        ax2.axvline(c, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss")
    ax2.set_title("Validation loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(f"Loss curves — {exp_dir.name}")
    fig.tight_layout()

    out_path = exp_dir / "loss_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
