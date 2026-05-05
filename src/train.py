"""Run mlx-lm LoRA fine-tuning on Phi-2 using configs/lora_config.json.

Adapter weights are saved to experiments/<YYYY-MM-DD-HHMM>/.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

TRAIN_RE = re.compile(
    r"Iter\s+(\d+):\s+Train loss\s+([0-9.]+),\s+Learning Rate\s+([0-9.eE+\-]+),\s+It/sec\s+([0-9.]+)"
)
VAL_RE = re.compile(
    r"Iter\s+(\d+):\s+Val loss\s+([0-9.]+),\s+Val took\s+([0-9.]+)"
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "lora_config.json"
DATA_DIR = ROOT / "data"
EXPERIMENTS_DIR = ROOT / "experiments"


def main() -> None:
    with CONFIG_PATH.open() as f:
        cfg = json.load(f)

    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    out_dir = EXPERIMENTS_DIR / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist a copy of the JSON config alongside the run for reproducibility
    (out_dir / "lora_config.json").write_text(json.dumps(cfg, indent=2))

    # mlx_lm.lora doesn't expose --lora-rank / --lora-alpha / --lora-dropout /
    # --lora-scale / --lora-keys as CLI flags — they're only read from a YAML
    # config under `lora_parameters`. So we materialize a YAML config that
    # carries the full config (top-level + lora_parameters) and pass --config.
    yaml_cfg = {
        "model": cfg["model"],
        "train": True,
        "data": str(DATA_DIR),
        "seed": cfg["seed"],
        "num_layers": cfg["num_layers"],
        "batch_size": cfg["batch_size"],
        "iters": cfg["iters"],
        "val_batches": cfg["val_batches"],
        "learning_rate": cfg["learning_rate"],
        "steps_per_report": cfg["steps_per_report"],
        "steps_per_eval": cfg["steps_per_eval"],
        "save_every": cfg["save_every"],
        "adapter_path": str(out_dir),
        "max_seq_length": cfg["max_seq_length"],
        "grad_checkpoint": bool(cfg.get("grad_checkpoint", False)),
        "lora_parameters": cfg["lora_parameters"],
    }
    yaml_path = out_dir / "lora_config.yaml"
    with yaml_path.open("w") as f:
        yaml.safe_dump(yaml_cfg, f, sort_keys=False)

    cmd = [sys.executable, "-m", "mlx_lm.lora", "--config", str(yaml_path)]

    print("Running:", " ".join(cmd))
    print(f"Adapters → {out_dir}")

    metrics_path = out_dir / "metrics.csv"
    rows: list[dict] = []

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

            m = TRAIN_RE.search(line)
            if m:
                rows.append({
                    "iter": int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "val_loss": "",
                    "learning_rate": float(m.group(3)),
                    "its_per_sec": float(m.group(4)),
                })
                continue

            m = VAL_RE.search(line)
            if m:
                rows.append({
                    "iter": int(m.group(1)),
                    "train_loss": "",
                    "val_loss": float(m.group(2)),
                    "learning_rate": "",
                    "its_per_sec": "",
                })
    finally:
        ret = proc.wait()
        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["iter", "train_loss", "val_loss", "learning_rate", "its_per_sec"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {metrics_path} ({len(rows)} rows)")

    if ret != 0:
        raise SystemExit(ret)


if __name__ == "__main__":
    main()
