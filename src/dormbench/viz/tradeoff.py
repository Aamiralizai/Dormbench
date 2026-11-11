"""Figure — compartment invariance versus in vitro discrimination."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .theme import GRID, INK, MUTED, apply_theme, save_svg

HOT, COOL, GREEN = "#c1272d", "#1b6ca8", "#2a9d8f"


def plot_tradeoff(res: pd.DataFrame, out_path: str | Path):
    apply_theme()
    import matplotlib.pyplot as plt

    d = res.dropna(subset=["auroc"]).copy()
    # unconstrained row has no cap; place it at the left as "none"
    d["cap_label"] = d["max_specificity"].map(
        lambda v: "none" if pd.isna(v) else f"{v:g}")
    d = d.reset_index(drop=True)
    x = np.arange(len(d))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

    # --- panel 1: AUROC as the filter tightens --------------------------
    ax = axes[0]
    ax.axhspan(0.45, 0.55, color=GREEN, alpha=0.10, lw=0)
    ax.axhline(0.5, color=INK, lw=1.2, ls="--")
    ax.plot(x, d["auroc"], "-o", color=HOT, lw=2.2, ms=7,
            mfc="white", mew=1.8, zorder=3)
    best = int(d["invariance"].idxmin())
    ax.plot([x[best]], [d.loc[best, "auroc"]], "o", color=HOT, ms=12,
            mfc=HOT, mec="white", mew=1.6, zorder=4)
    ax.annotate(f"cap {d.loc[best, 'cap_label']}\nAUROC {d.loc[best,'auroc']:.3f}",
                (x[best], d.loc[best, "auroc"]), xytext=(10, -34),
                textcoords="offset points", fontsize=7.8, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.text(0.02, 0.52, "compartment-neutral", transform=ax.transAxes,
            fontsize=7.4, color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(d["cap_label"], fontsize=8)
    ax.set_xlabel("maximum compartment specificity allowed", fontsize=9, color=INK)
    ax.set_ylabel("AUROC  cancer > stroma", fontsize=9, color=INK)
    ax.set_title("Excluding compartment-specific genes\nmoves the signature to neutral",
                 fontsize=9.6, color=INK, loc="left", pad=8)
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)

    # --- panel 2: the cost, on both axes --------------------------------
    ax = axes[1]
    ax.plot(x, d["cv_accuracy"], "-s", color=COOL, lw=2.0, ms=6,
            mfc="white", mew=1.6, label="in vitro CV accuracy")
    ax.plot(x, 1 - d["invariance"] * 2, "-o", color=HOT, lw=2.0, ms=6,
            mfc="white", mew=1.6, label="compartment invariance")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(d["cap_label"], fontsize=8)
    ax.set_xlabel("maximum compartment specificity allowed", fontsize=9, color=INK)
    ax.set_ylabel("scaled to 0–1", fontsize=9, color=INK)
    ax.set_title("Invariance improves; in vitro accuracy does not move\n"
                 "(the training task is saturated)",
                 fontsize=9.6, color=INK, loc="left", pad=8)
    ax.legend(fontsize=7.6, frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)

    # --- panel 3: how many genes it costs -------------------------------
    ax = axes[2]
    base = float(d["genes_available"].max())
    removed = base - d["genes_available"]
    ax.bar(x, removed, color=MUTED, alpha=0.85, width=0.62)
    for xi, (r, n) in enumerate(zip(removed, d["genes_available"])):
        ax.text(xi, r, f"{int(r)}", ha="center", va="bottom",
                fontsize=7.6, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(d["cap_label"], fontsize=8)
    ax.set_xlabel("maximum compartment specificity allowed", fontsize=9, color=INK)
    ax.set_ylabel("genes excluded", fontsize=9, color=INK)
    ax.set_title(f"Cost in genes (of {int(base)} available)",
                 fontsize=9.6, color=INK, loc="left", pad=8)
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Figure — a dormancy signature can be made compartment-neutral by "
        "excluding a small minority of genes at derivation",
        fontsize=11, color=INK, x=0.005, ha="left", y=0.998)
    fig.text(0.005, 0.004,
             "Cross-validated accuracy is 1.00 at every threshold: the in vitro "
             "task is saturated, so no accuracy cost is measurable rather than "
             "no cost existing.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.030, 1, 0.950])

    path = save_svg(fig, out_path)
    plt.close(fig)
    return path
