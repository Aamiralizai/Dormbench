"""Figure 5 — what drives the compartment preference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .theme import INK, MUTED, apply_theme, save_svg

CANCER = "Cancer Epithelial"
HOT, COOL = "#c1272d", "#1b6ca8"


def plot_drivers(summary: pd.DataFrame, lead: pd.DataFrame,
                 means: pd.DataFrame, out_path: str | Path):
    apply_theme()
    import matplotlib.pyplot as plt

    up = summary[summary["direction"] == "up"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # panel 1: concentration curve
    ax = axes[0]
    y = np.arange(len(up))
    up = up.sort_values("gap_share_top50")
    ax.barh(y - 0.19, up["gap_share_top10"].values * 100, height=0.36,
            color=HOT, alpha=0.85, label="top 10 genes")
    ax.barh(y + 0.19, up["gap_share_top50"].values * 100, height=0.36,
            color=COOL, alpha=0.85, label="top 50 genes")
    ax.axvline(60, color=INK, lw=1.0, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['signature']}\n(n={int(r['n_genes'])})"
                        for _, r in up.iterrows()], fontsize=7.2)
    ax.set_xlabel("% of compartment gap carried", fontsize=9, color=INK)
    ax.set_title("Is the preference distributed or dragged?",
                 fontsize=9.8, color=INK, loc="left", pad=8)
    ax.legend(fontsize=7.4, frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.5)
    ax.set_axisbelow(True)

    # panel 2: leading edge of the largest signature
    ax = axes[1]
    if len(lead):
        big = up.iloc[-1]["signature"] if len(up) else lead["signature"].iat[0]
        sel = lead[(lead["signature"] == big) & (lead["direction"] == "up")]
        if sel.empty:
            sel = lead[lead["direction"] == "up"]
        sel = sel.sort_values("log2_ratio", ascending=True).tail(20)
        y2 = np.arange(len(sel))
        ax.barh(y2, sel["log2_ratio"].values, color=HOT, alpha=0.85, height=0.66)
        ax.set_yticks(y2)
        ax.set_yticklabels(sel["gene"], fontsize=7.2)
        ax.axvline(0, color=INK, lw=0.9)
        tgt = sel["target"].mode().iat[0] if len(sel) else "target"
        ax.set_xlabel(f"log2  {tgt} / cancer epithelium", fontsize=9, color=INK)
        ax.set_title(f"Leading edge — {big}", fontsize=9.8, color=INK,
                     loc="left", pad=8)
        ax.grid(axis="x", alpha=0.5)
        ax.set_axisbelow(True)

    # panel 3: distribution of per-gene ratios
    ax = axes[2]
    for i, (_, r) in enumerate(up.iterrows()):
        sig = r["signature"]
        sel = lead[(lead["signature"] == sig) & (lead["direction"] == "up")]
        if sel.empty:
            continue
        ax.scatter(sel["expr_cancer"], sel["expr_target"], s=26, alpha=0.7,
                   label=sig, lw=0, edgecolor="none")
    lim = ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 1
    ax.plot([0, lim], [0, lim], color=INK, lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("mean expression, cancer epithelium", fontsize=9, color=INK)
    ax.set_ylabel("mean expression, preferred compartment", fontsize=9, color=INK)
    ax.set_title("Leading-edge genes: where are they expressed?",
                 fontsize=9.8, color=INK, loc="left", pad=8)
    ax.legend(fontsize=6.8, frameon=False, markerscale=1.4)
    ax.grid(alpha=0.5)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Figure 5 — the genes responsible for each signature's compartment "
        "preference",
        fontsize=11, color=INK, x=0.005, ha="left", y=0.998)
    fig.text(0.005, 0.004,
             "Non-cycling cells only. A signature whose top 50 genes carry most "
             "of the gap is not a mislocalised programme — it is a score dragged "
             "by a minority of its members.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.030, 1, 0.950])

    path = save_svg(fig, out_path)
    plt.close(fig)
    return path
