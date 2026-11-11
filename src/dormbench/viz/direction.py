"""Figure 3 — direction-resolved attribution: the paper's central figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .theme import INK, MUTED, apply_theme, save_svg

CANCER = "Cancer Epithelial"
ORDER = ["Cancer Epithelial", "Normal Epithelial", "CAFs", "PVL", "Endothelial",
         "T-cells", "Myeloid", "B-cells", "Plasmablasts"]
COL = {"Cancer Epithelial": "#c1272d", "Normal Epithelial": "#e08a8a",
       "CAFs": "#1b6ca8", "PVL": "#4a8fc0", "Endothelial": "#7fb3d5"}
OTHER = "#9aa0ad"
UP_C, DOWN_C = "#2a9d8f", "#c1272d"


def plot_direction_resolved(res: pd.DataFrame, df: pd.DataFrame,
                            ct: np.ndarray, keep: np.ndarray,
                            out_path: str | Path):
    apply_theme()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # panel 1: corr with G2M, up vs down
    ax = axes[0]
    agg = (res.groupby(["signature", "direction"])["corr_g2m"].mean()
           .unstack(fill_value=np.nan))
    sigs = list(agg.index)
    y = np.arange(len(sigs))
    if "up" in agg:
        ax.barh(y - 0.19, agg["up"].values, height=0.36, color=UP_C,
                alpha=0.85, label="up genes")
    if "down" in agg:
        ax.barh(y + 0.19, agg["down"].values, height=0.36, color=DOWN_C,
                alpha=0.85, label="down genes")
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(sigs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("correlation with G2M score", fontsize=9, color=INK)
    ax.set_title("The proliferation module lives in the 'down' half",
                 fontsize=10, color=INK, loc="left", pad=8)
    ax.legend(fontsize=7.6, frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.5)
    ax.set_axisbelow(True)

    # panel 2: cancer vs best stromal, up direction only
    ax = axes[1]
    up = (res[res["direction"] == "up"]
          .groupby("signature")[["auroc_cancer_vs_stroma_nc"]].mean()
          .sort_values("auroc_cancer_vs_stroma_nc"))
    y2 = np.arange(len(up))
    vals = up["auroc_cancer_vs_stroma_nc"].values
    ax.barh(y2, vals, color=[COL[CANCER] if v > 0.5 else "#1b6ca8" for v in vals],
            alpha=0.85, height=0.6)
    ax.axvline(0.5, color=INK, lw=1.1, ls="--")
    ax.set_yticks(y2)
    ax.set_yticklabels(up.index, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUROC  cancer epithelium > top stromal", fontsize=9, color=INK)
    ax.set_title("Up-programmes only, non-cycling cells", fontsize=10,
                 color=INK, loc="left", pad=8)
    ax.text(0.52, len(up) - 0.4, "cancer-dominant", fontsize=7.4,
            color=COL[CANCER], va="top")
    ax.text(0.48, len(up) - 0.4, "stroma-dominant", fontsize=7.4,
            color="#1b6ca8", ha="right", va="top")
    ax.grid(axis="x", alpha=0.5)
    ax.set_axisbelow(True)

    # panel 3: compartment means for the up score of the first signature
    ax = axes[2]
    up_cols = [c for c in df.columns if c.endswith("|up")
               and "G2M" not in c and c.startswith("AUCell")]
    col = up_cols[0] if up_cols else None
    if col:
        present = [g for g in ORDER if g in set(ct)]
        data = [df[col].values[keep & (ct == g)] for g in present]
        parts = ax.violinplot(data, showextrema=False, widths=0.82)
        for pc, g in zip(parts["bodies"], present):
            pc.set_facecolor(COL.get(g, OTHER))
            pc.set_alpha(0.75)
            pc.set_edgecolor("none")
        for i, d in enumerate(data, start=1):
            if len(d):
                m = float(np.median(d))
                ax.plot([i - 0.28, i + 0.28], [m, m], color="white", lw=2.2, zorder=4)
                ax.plot([i - 0.28, i + 0.28], [m, m], color=INK, lw=1.0, zorder=5)
        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels(present, rotation=38, ha="right", fontsize=7.6)
        ax.set_title(col.replace("|", "  ·  "), fontsize=9.5, color=INK,
                     loc="left", pad=8)
        ax.grid(axis="y", alpha=0.55)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Figure 3 — direction-resolved attribution: signed dormancy scores "
        "encode proliferation, not compartment",
        fontsize=11, color=INK, x=0.005, ha="left", y=0.998)
    fig.text(0.005, 0.004,
             "Non-cycling cells only (cycling and top-quartile G2M removed). "
             "AUROC 0.5 = no separation between cancer epithelium and stroma.",
             fontsize=7.6, color=MUTED)
    fig.tight_layout(rect=[0, 0.028, 1, 0.952])

    path = save_svg(fig, out_path)
    plt.close(fig)
    return path
