"""Figure 1 — dormancy signature scores by tumour compartment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .theme import GRID, INK, MUTED, apply_theme, save_svg

CANCER = "Cancer Epithelial"
ORDER = ["Cancer Epithelial", "Normal Epithelial", "CAFs", "PVL", "Endothelial",
         "T-cells", "Myeloid", "B-cells", "Plasmablasts"]
CANCER_COL, STROMA_COL, IMMUNE_COL = "#c1272d", "#1b6ca8", "#9aa0ad"

GROUP_COL = {
    "Cancer Epithelial": CANCER_COL,
    "Normal Epithelial": "#e08a8a",
    "CAFs": STROMA_COL,
    "PVL": "#4a8fc0",
    "Endothelial": "#7fb3d5",
}


def plot_compartment_scores(df: pd.DataFrame, out_path: str | Path):
    apply_theme()
    import matplotlib.pyplot as plt

    score_cols = [c for c in df.columns if c != "celltype_major"]
    if not score_cols:
        raise ValueError("no score columns")

    present = [c for c in ORDER if c in set(df["celltype_major"])]
    present += [c for c in sorted(set(df["celltype_major"])) if c not in present]

    n = len(score_cols)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.9 * nrow),
                             squeeze=False)

    for ax, col in zip(axes.ravel(), score_cols):
        data = [df.loc[df["celltype_major"] == g, col].dropna().values
                for g in present]
        parts = ax.violinplot(data, showextrema=False, widths=0.82)
        for pc, g in zip(parts["bodies"], present):
            pc.set_facecolor(GROUP_COL.get(g, IMMUNE_COL))
            pc.set_alpha(0.72)
            pc.set_edgecolor("none")

        for i, d in enumerate(data, start=1):
            if len(d):
                med = float(np.median(d))
                ax.plot([i - 0.28, i + 0.28], [med, med], color="white",
                        lw=2.2, zorder=4, solid_capstyle="butt")
                ax.plot([i - 0.28, i + 0.28], [med, med], color=INK,
                        lw=1.0, zorder=5, solid_capstyle="butt")

        if CANCER in present:
            ci = present.index(CANCER) + 1
            ax.axvspan(ci - 0.5, ci + 0.5, color="#f5c518", alpha=0.10,
                       lw=0, zorder=0)

        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels(present, rotation=38, ha="right", fontsize=7.6)
        ax.set_title(col.replace("|", "  ·  "), fontsize=9.5, color=INK,
                     loc="left", pad=8)
        ax.grid(axis="y", alpha=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "Figure 1 — dormancy signature scores by tumour compartment  "
        "(Wu et al. breast atlas, 100,064 cells)",
        fontsize=11, color=INK, x=0.005, ha="left", y=0.998,
    )
    fig.text(0.005, 0.004,
             "White bar = median.  Yellow = cancer epithelial, the compartment "
             "dormancy signatures are assumed to measure.",
             fontsize=7.6, color=MUTED)
    fig.tight_layout(rect=[0, 0.022, 1, 0.972])

    path = save_svg(fig, out_path)
    plt.close(fig)
    return path
