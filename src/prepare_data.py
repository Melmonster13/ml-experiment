"""Prepare an instruction dataset for mlx-lm LoRA fine-tuning.

Currently loads `iamtarun/python_code_instructions_18k_alpaca` (Alpaca-style
Python instructions); swap the `load_dataset` call for `tatsu-lab/alpaca` to
fine-tune on the general Alpaca set instead.

mlx-lm expects a `data/` directory containing train.jsonl, valid.jsonl, test.jsonl,
each with one JSON object per line in `{"text": "..."}` format (completions style).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAMPLE_PATH = DATA_DIR / "alpaca_sample.jsonl"

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def format_example(ex: dict) -> dict:
    if ex.get("input"):
        prompt = PROMPT_WITH_INPUT.format(instruction=ex["instruction"], input=ex["input"])
    else:
        prompt = PROMPT_NO_INPUT.format(instruction=ex["instruction"])
    return {"text": prompt + ex["output"]}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    formatted = [format_example(ex) for ex in ds]

    rng = random.Random(0)
    rng.shuffle(formatted)

    write_jsonl(SAMPLE_PATH, formatted[:500])
    print(f"Wrote {SAMPLE_PATH} ({500} examples)")

    # 90/5/5 train/valid/test split
    n = len(formatted)
    n_val = max(500, n // 20)
    n_test = max(500, n // 20)
    train = formatted[: n - n_val - n_test]
    valid = formatted[n - n_val - n_test : n - n_test]
    test = formatted[n - n_test :]

    write_jsonl(DATA_DIR / "train.jsonl", train)
    write_jsonl(DATA_DIR / "valid.jsonl", valid)
    write_jsonl(DATA_DIR / "test.jsonl", test)
    print(f"Wrote train/valid/test: {len(train)}/{len(valid)}/{len(test)}")


if __name__ == "__main__":
    main()
