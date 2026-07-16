"""Render the README comparison charts from committed experiments/*/metrics.csv.

Usage:
    python src/plot_comparison.py

Writes assets/phi2-sweep-val-loss.png and assets/sweep-best-val-loss.png.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
ASSETS_DIR = ROOT / "assets"

BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Distinct Phi-2 configs only: the 200/250-iter runs of the winning config
# (2026-05-04-0014/2135) trace the 300-iter curve point-for-point.
SWEEP_SERIES = [
    ("2026-05-03-1733", "8 layers · rank 8", BLUE),
    ("2026-05-03-2339", "16 layers · rank 8", AQUA),
    ("2026-05-03-2358", "16 layers · rank 16", YELLOW),
]

# Mirrors the README experiments table. 2026-05-04-2214/2228/2240 are excluded:
# empty or truncated metrics from aborted StarCoder2 runs.
SUMMARY_ROWS = [
    ("2026-05-03-1733", "phi2", "8 layers · rank 8 · 600 iters"),
    ("2026-05-03-1745", "phi2", "8 layers · rank 8 · 300 iters"),
    ("2026-05-03-2358", "phi2", "16 layers · rank 16 · 300 iters"),
    ("2026-05-03-2339", "phi2", "16 layers · rank 8 · 300 iters"),
    ("2026-05-04-0014", "phi2", "16 layers · rank 8 · 200 iters"),
    ("2026-05-04-2233", "sc2", "4 layers · rank 8 · 200 iters"),
]
GROUP_HEADERS = {"phi2": "Phi-2 · Alpaca", "sc2": "StarCoder2-3B · Python instructions"}
GROUP_COLORS = {"phi2": BLUE, "sc2": AQUA}
WINNERS = {"2026-05-04-0014", "2026-05-04-2233"}


def val_points(run: str) -> tuple[list[int], list[float]]:
    with (EXPERIMENTS_DIR / run / "metrics.csv").open() as f:
        pts = [(int(r["iter"]), float(r["val_loss"]))
               for r in csv.DictReader(f) if r.get("val_loss")]
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_sweep_curves() -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=200)
    for run, label, color in SWEEP_SERIES:
        x, y = val_points(run)
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=label, zorder=3)
        ax.annotate(label, (x[-1], y[-1]), xytext=(8, 0),
                    textcoords="offset points", fontsize=9.5, color=INK2,
                    va="center")

    ax.scatter([200], [0.817], s=150, facecolors="none", edgecolors=INK,
               linewidths=1.4, zorder=4)
    ax.annotate("best 0.82 — final run\n(16 · r8) stops here", (200, 0.817),
                xytext=(258, 0.715), textcoords="data", fontsize=9.5,
                color=INK2, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=BASELINE, lw=1))

    ax.set_xlim(-10, 700)
    ax.set_ylim(0.7, 2.05)
    ax.set_xticks([1, 100, 200, 300, 400, 500, 600])
    ax.set_xlabel("Iteration", fontsize=10, color=MUTED)
    ax.set_ylabel("Validation loss", fontsize=10, color=MUTED)
    ax.set_title("Phi-2 / Alpaca — validation loss across the LoRA sweep",
                 fontsize=13, color=INK, loc="left", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5, labelcolor=INK2)
    fig.tight_layout()

    out = ASSETS_DIR / "phi2-sweep-val-loss.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_best_val_summary() -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=200)

    ys: list[float] = []
    y, prev_group = 0.0, None
    for _, group, _ in SUMMARY_ROWS:
        if prev_group is not None and group != prev_group:
            y -= 0.6
        y -= 1.0
        ys.append(y)
        prev_group = group

    seen: set[str] = set()
    for (run, group, label), yy in zip(SUMMARY_ROWS, ys):
        v = min(val_points(run)[1])
        win = run in WINNERS
        ax.hlines(yy, 0.5, v, color=GRID, linewidth=1, zorder=2)
        ax.scatter([v], [yy], s=110 if win else 80, color=GROUP_COLORS[group],
                   edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.annotate(f"{v:.2f}", (v, yy), xytext=(10, 0),
                    textcoords="offset points", fontsize=10,
                    color=INK if win else INK2, va="center",
                    fontweight="bold" if win else "normal")
        ax.annotate(("★ " if win else "") + label, (0.5, yy), xytext=(-8, 0),
                    textcoords="offset points", fontsize=10,
                    color=INK if win else INK2, ha="right", va="center",
                    fontweight="bold" if win else "normal")
        if group not in seen:
            seen.add(group)
            ax.annotate(GROUP_HEADERS[group], (0.5, yy + 0.75), xytext=(-8, 0),
                        textcoords="offset points", fontsize=10.5, color=INK,
                        fontweight="bold", ha="right", va="center")

    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(min(ys) - 0.7, 0.4)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.set_xlabel("Best validation loss  (lower is better)", fontsize=10,
                  color=MUTED)
    ax.set_title("Best validation loss by configuration", fontsize=13,
                 color=INK, loc="left", pad=14)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=10)
    fig.tight_layout()

    out = ASSETS_DIR / "sweep-best-val-loss.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out


def main() -> None:
    ASSETS_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
    })
    for out in (plot_sweep_curves(), plot_best_val_summary()):
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
