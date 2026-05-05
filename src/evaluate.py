"""Compare a base model vs its LoRA-tuned adapter on a prompt.

Defaults to microsoft/phi-2 + the most recent run in experiments/, but the
model and adapter can be pinned via --model / --adapter for any run
(e.g. StarCoder2-3B + a Python-instruction adapter).

Usage:
    python src/evaluate.py --adapter experiments/2026-04-30-1200 "Your prompt here"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from mlx_lm import generate, load

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "microsoft/phi-2"

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def latest_experiment() -> Path | None:
    exp = ROOT / "experiments"
    if not exp.exists():
        return None
    runs = sorted(p for p in exp.iterdir() if p.is_dir())
    return runs[-1] if runs else None


def run(model_path: str, adapter_path: str | None, prompt: str, max_tokens: int) -> str:
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Instruction to send to the model")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None,
                        help="Path to adapter dir (defaults to latest in experiments/)")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    adapter = args.adapter or (str(latest_experiment()) if latest_experiment() else None)
    if adapter is None:
        raise SystemExit("No adapter found. Pass --adapter or run training first.")

    formatted = PROMPT_TEMPLATE.format(instruction=args.prompt)

    print("=" * 60)
    print("BASE MODEL")
    print("=" * 60)
    base = run(args.model, None, formatted, args.max_tokens)
    print(base)

    print()
    print("=" * 60)
    print(f"FINE-TUNED ({adapter})")
    print("=" * 60)
    tuned = run(args.model, adapter, formatted, args.max_tokens)
    print(tuned)


if __name__ == "__main__":
    main()
