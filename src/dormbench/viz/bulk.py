"""Figure 7 — signature scores versus compartment content in bulk cohorts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .theme import INK, MUTED, apply_theme, save_svg

CTRL, HOT, COOL = "#2a9d8f", "#c1272d", "#1b6ca8"


def plot_bulk(corr: pd.DataFrame, out_path: str | Path):
    apply_theme()
    import matplotlib.pyplot as plt

    up = corr[corr["score"].str.endswith("|up")].copy()
    up["sig"] = up["score"].str.split("|").str[0]
    cohorts = sorted(up["cohort"].unique())

    fig, axes = plt.subplots(1, max(2, len(cohorts) + 1),
                             figsize=(5.4 * max(2, len(cohorts) + 1), 4.6),
                             squeeze=False)
    axes = axes[0]

    for ax, coh in zip(axes, cohorts):
        d = (up[up["cohort"] == coh].groupby("sig")[["r_stroma", "r_cancer"]]
             .mean().sort_values("r_stroma"))
        y = np.arange(len(d))
        cols = [CTRL if "E2F" in s else (HOT if "TRADITIOM" in s else COOL)
                for s in d.index]
        ax.barh(y, d["r_stroma"].values, color=cols, height=0.68, alpha=0.9)
        ax.axvline(0, color=INK, lw=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(d.index, fontsize=7.6)
        ax.set_xlabel("Spearman r  with stromal content", fontsize=9, color=INK)
        ax.set_title(coh, fontsize=10, color=INK, loc="left", pad=8)
        ax.grid(axis="x", alpha=0.5)
        ax.set_axisbelow(True)

    # final panel: stroma vs cancer correlation, all cohorts
    ax = axes[len(cohorts)] if len(axes) > len(cohorts) else axes[-1]
    for coh, mk in zip(cohorts, ("o", "s", "^")):
        d = up[up["cohort"] == coh]
        cols = [CTRL if "E2F" in s else (HOT if "TRADITIOM" in s else COOL)
                for s in d["sig"]]
        ax.scatter(d["r_cancer"], d["r_stroma"], c=cols, s=52, marker=mk,
                   edgecolor="white", lw=1.0, label=coh, zorder=3)
    lo = float(np.nanmin([up["r_cancer"].min(), up["r_stroma"].min()])) - 0.1
    hi = float(np.nanmax([up["r_cancer"].max(), up["r_stroma"].max()])) + 0.1
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1.0, ls="--", alpha=0.6)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("r  with cancer-cell content", fontsize=9, color=INK)
    ax.set_ylabel("r  with stromal content", fontsize=9, color=INK)
    ax.set_title("Above the diagonal = composition-driven", fontsize=10,
                 color=INK, loc="left", pad=8)
    ax.legend(fontsize=7.4, frameon=False)
    ax.grid(alpha=0.5)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Figure 7 — in bulk tumours, dormancy scores track cell-type "
        "composition; the cell-cycle control does not",
        fontsize=11, color=INK, x=0.005, ha="left", y=0.998)
    fig.text(0.005, 0.004,
             "Green = E2F_control (no cell-type-marker content). "
             "Red = TRADITIOM dormancy. Compartment abundance estimated from "
             "Wu atlas marker genes.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.030, 1, 0.950])

    path = save_svg(fig, out_path)
    plt.close(fig)
    return path
