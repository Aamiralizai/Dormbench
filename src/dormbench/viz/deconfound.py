"""Figure 2 — is the compartment effect proliferation?"""

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


def plot_deconfounding(df: pd.DataFrame, ct: np.ndarray, g2m: np.ndarray,
                       score_cols: list[str], keep: np.ndarray,
                       out_path: str | Path):
    apply_theme()
    import matplotlib.pyplot as plt

    col = score_cols[0]
    present = [c for c in ORDER if c in set(ct)]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    # --- panel 1: score vs G2M, coloured by compartment ------------------
    ax = axes[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), min(12000, len(df)), replace=False)
    for g in present:
        m = (ct[idx] == g)
        if m.sum() == 0:
            continue
        ax.scatter(g2m[idx][m], df[col].values[idx][m], s=2.5, alpha=0.28,
                   color=COL.get(g, OTHER), lw=0, label=g if g in COL else None)
    r = float(np.corrcoef(df[col].values, g2m)[0, 1])
    ax.set_xlabel("G2M score", fontsize=9, color=INK)
    ax.set_ylabel(col.split("|")[1] + " score", fontsize=9, color=INK)
    ax.set_title(f"Dormancy score vs proliferation   r = {r:+.2f}",
                 fontsize=10, color=INK, loc="left", pad=8)
    ax.legend(fontsize=7, frameon=False, markerscale=4, loc="upper right")
    ax.grid(alpha=0.5)
    ax.set_axisbelow(True)

    # --- panel 2: compartment means, all vs cycle-matched ----------------
    ax = axes[1]
    x = np.arange(len(present))
    allm = [np.nanmean(df[col].values[ct == g]) for g in present]
    matm = [np.nanmean(df[col].values[(ct == g) & keep]) for g in present]
    ax.barh(x - 0.2, allm, height=0.38, color=MUTED, alpha=0.75, label="all cells")
    ax.barh(x + 0.2, matm, height=0.38,
            color=[COL.get(g, OTHER) for g in present], label="non-cycling only")
    ax.set_yticks(x)
    ax.set_yticklabels(present, fontsize=7.8)
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("mean score", fontsize=9, color=INK)
    ax.set_title("Compartment ranking before / after cycle matching",
                 fontsize=10, color=INK, loc="left", pad=8)
    ax.legend(fontsize=7.4, frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.5)
    ax.set_axisbelow(True)

    # --- panel 3: within non-cycling cancer ------------------------------
    ax = axes[2]
    v = df[col].values[(ct == CANCER) & keep]
    ax.hist(v, bins=60, color=COL[CANCER], alpha=0.8, lw=0)
    ax.axvline(float(np.median(v)), color=INK, lw=1.2, ls="--")
    ax.set_xlabel(col.split("|")[1] + " score", fontsize=9, color=INK)
    ax.set_ylabel("cells", fontsize=9, color=INK)
    ax.set_title(f"Non-cycling cancer epithelium  (n = {len(v):,})",
                 fontsize=10, color=INK, loc="left", pad=8)
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Figure 2 — is the compartment effect dormancy, or proliferation?",
        fontsize=11, color=INK, x=0.005, ha="left", y=0.998)
    fig.text(0.005, 0.004,
             "A unimodal distribution in panel 3 means the signature does not "
             "identify a discrete dormant subpopulation among cancer cells.",
             fontsize=7.6, color=MUTED)
    fig.tight_layout(rect=[0, 0.028, 1, 0.955])

    path = save_svg(fig, out_path)
    plt.close(fig)
    return path
