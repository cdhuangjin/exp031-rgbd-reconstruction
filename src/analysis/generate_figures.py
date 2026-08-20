"""Generate publication-ready summary figures from verified window aggregates."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).with_name("figure_summary_data.json")
OUT = ROOT / "04_结果" / "figures"
METHODS = ["Baseline", "M1", "M2", "M3", "Full"]
COLORS = ["#000000", "#0072B2", "#E69F00", "#009E73", "#D55E00"]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _style()
    data = json.loads(DATA.read_text(encoding="utf-8"))
    scenes = list(data["scenes"])

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharey=True)
    x = np.arange(len(METHODS))
    for index, (ax, scene) in enumerate(zip(axes, scenes)):
        means = [data["scenes"][scene][m][0] for m in METHODS]
        errors = [data["scenes"][scene][m][1] for m in METHODS]
        ax.bar(x, means, yerr=errors, capsize=2.5, color=COLORS, edgecolor="white", linewidth=0.5)
        ax.set_title(scene, fontsize=9)
        ax.set_xticks(x, METHODS, rotation=35, ha="right")
        ax.set_ylim(0, max(means) + max(errors) + 0.08)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        if index == 0:
            ax.set_ylabel("Held-out depth RMSE (m)")
        ax.text(-0.12, 1.08, "ABC"[index], transform=ax.transAxes, fontweight="bold", fontsize=10)
    fig.suptitle("Cross-scene ablation across three expanded sampling windows", y=1.02, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig1_rmse_ablation")

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    deltas = [data["full_minus_baseline"][s][0] for s in scenes]
    bars = ax.bar(np.arange(len(scenes)), deltas, color=["#56B4E9", "#009E73", "#0072B2"], width=0.62)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(np.arange(len(scenes)), scenes, rotation=25, ha="right")
    ax.set_ylabel("Full − Baseline RMSE (m)")
    ax.set_title("Full-method change on unseen scenes")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for bar, scene in zip(bars, scenes):
        improved, total = data["full_minus_baseline"][scene][1:]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 0.72,
            f"{improved}/{total} windows",
            ha="center",
            va="center",
            fontsize=7,
        )
    fig.tight_layout()
    _save(fig, "fig2_full_delta")

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    baseline_cov = [data["coverage"][s][0] for s in scenes]
    full_cov = [data["coverage"][s][1] for s in scenes]
    positions = np.arange(len(scenes))
    width = 0.34
    ax.bar(positions - width / 2, baseline_cov, width, label="Baseline", color="#999999")
    ax.bar(positions + width / 2, full_cov, width, label="Full", color="#D55E00")
    ax.set_xticks(positions, scenes, rotation=25, ha="right")
    ax.set_ylabel("Valid depth coverage")
    ax.set_title("Coverage trade-off in expanded windows")
    ax.set_ylim(0, max(baseline_cov) * 1.18)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig3_coverage_tradeoff")


if __name__ == "__main__":
    main()
