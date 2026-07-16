# ml-experiment

LoRA fine-tuning of small open-weight LLMs (`microsoft/phi-2`, `bigcode/starcoder2-3b`) on instruction datasets, running locally with MLX on Apple Silicon. Project is in a finished/portfolio state — pipeline works end-to-end, hyperparameter sweep is complete, results are committed.

## Stack & hardware

- **Hardware:** M4 MacBook Pro, 16 GB unified memory
- **Python:** 3.11, arm64 native, venv at `.venv`
- **Frameworks:** `mlx`, `mlx-lm`, `datasets`, `transformers`, `huggingface-hub`, `matplotlib`, `pyyaml`
- **Models exercised:** `microsoft/phi-2` (~2.7B), `bigcode/starcoder2-3b` (~3B)
- **Datasets exercised:** `tatsu-lab/alpaca` (~52k instruction-response pairs), `iamtarun/python_code_instructions_18k_alpaca` (~18k Python instructions)

## Folder layout

- `src/` — `prepare_data.py`, `train.py`, `evaluate.py`, `plot_loss.py`, `plot_comparison.py` (cross-run README charts)
- `assets/` — README charts, regenerated with `python src/plot_comparison.py`
- `data/` — formatted dataset (gitignored)
- `configs/` — `lora_config.json`, the active hyperparameters for the next run
- `experiments/` — one dated subfolder per run (`YYYY-MM-DD-HHMM/`) holding adapter weights, a copy of the config, `metrics.csv`, and `loss_curves.png`. Adapter weights are gitignored; `metrics.csv` and `loss_curves.png` are kept.
- `notebooks/` — exploration

## Run order

```bash
# 1. Format the dataset (writes train/valid/test + a 500-row sample)
python src/prepare_data.py

# 2. Fine-tune; adapters land in experiments/<timestamp>/
python src/train.py

# 3. Plot loss curves for the latest run
python src/plot_loss.py

# 4. Compare base vs fine-tuned on a prompt (latest experiment by default)
python src/evaluate.py "Explain gradient descent in two sentences."
python src/evaluate.py --model bigcode/starcoder2-3b --adapter experiments/2026-05-04-2240 "..."
```

## Final best configs

**Phi-2 / Alpaca — best val loss 0.81**
- `num_layers: 16`, `lora_parameters.rank: 8`, `lora_parameters.alpha: 16`
- `iters: 200`, `learning_rate: 1e-4`, `batch_size: 1`, `max_seq_length: 1024`, `grad_checkpoint: true`

**StarCoder2-3B / Python instructions — best val loss 0.57**
- `num_layers: 4`, `lora_parameters.rank: 8`, `lora_parameters.alpha: 16`
- `iters: 200`, `learning_rate: 1e-4`, `batch_size: 1`, `max_seq_length: 256`, `grad_checkpoint: true`
- LoRA targets: `self_attn.{q,k,v}_proj`, `self_attn.dense`

## Key decisions

- **Alpaca prompt format** — standard "Below is an instruction…" template with a separate `### Input` block when the example has one. Stored in `{"text": ...}` lines (completions style), which is what `mlx_lm.lora` expects in `--data`.
- **Adapter-only outputs** — we don't fuse weights at the end of training. `evaluate.py` loads base + adapter at inference; fuse later with `mlx_lm.fuse` if a single distributable artifact is needed.
- **Dated experiment dirs** — every run writes to `experiments/<YYYY-MM-DD-HHMM>/`, including a snapshot of the config, so runs are reproducible and comparable.
- **YAML wrapping in `train.py`** — `mlx_lm.lora` only reads `lora_parameters` (rank/alpha/dropout/scale/keys) from a YAML config, not from CLI flags. `train.py` materializes a YAML next to the run and passes `--config`.

## Lessons from the hyperparameter search

- **Layer count mattered more than iteration count.** 8 → 16 trainable LoRA layers improved Phi-2's best val loss from 0.88 to 0.81 *with fewer iterations* (200 vs. 300).
- **Higher rank wasn't better.** At 16 layers, rank 8 (val 0.82) beat rank 16 (val 0.86) — the larger adapter overfit faster.
- **More iterations hurt.** 600 iters reached the same best val (0.88 at iter 200) as 300 iters at the same rank/layers, then drifted past the best checkpoint — val loss was back above 1.0 by iter 600. Keep `iters` modest and watch the val curve in `loss_curves.png`.
- **Narrower domains converge fast.** StarCoder2 on Python instructions reached 0.57 with only 4 LoRA layers and 200 iters.

## Known memory limitations

- 16 GB unified memory is the hard ceiling. Training requires `batch_size: 1` and `grad_checkpoint: true` for both models tested.
- **StarCoder2-3B forces `max_seq_length: 256`.** Many examples in the Python-instruction dataset exceed this and get truncated, so the model sees a clipped view of longer programs. Raising the cap OOMs on this hardware.
- Phi-2 fits fine at `max_seq_length: 1024`.
- For larger models or longer contexts, expect to reduce `num_layers`, reduce `max_seq_length`, or move off this machine.

## How to add a new experiment

1. Edit `configs/lora_config.json` — change `model`, `num_layers`, `iters`, `lora_parameters.rank/alpha`, or `max_seq_length` as needed. Keep `batch_size: 1` and `grad_checkpoint: true` unless freeing memory elsewhere.
2. If switching dataset, edit the `load_dataset(...)` call in `src/prepare_data.py` and re-run it so `data/` reflects the new dataset.
3. Run `python src/train.py`. A fresh `experiments/<YYYY-MM-DD-HHMM>/` folder is created automatically with a snapshot of the config, `adapters.safetensors`, and `metrics.csv`.
4. Run `python src/plot_loss.py` (defaults to the latest run) to render `loss_curves.png` into the same folder.
5. Sanity-check generations with `python src/evaluate.py "<prompt>"` — pin `--model` and `--adapter` if not running on the latest Phi-2 adapter.
6. Commit `metrics.csv` and `loss_curves.png` (allowed by `.gitignore`); add a row to the experiments table in `README.md`. If the run belongs in the README charts, add it to the run lists in `src/plot_comparison.py` and re-run that script.
