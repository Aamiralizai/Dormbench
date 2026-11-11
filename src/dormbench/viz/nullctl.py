"""Figure 4 — observed statistics against size- and expression-matched nulls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .theme import INK, MUTED, apply_theme, save_svg

SIG_C, NS_C = "#c1272d", "#9aa0ad"
UP_C, DOWN_C = "#2a9d8f", "#c1272d"


def plot_null_control(res: pd.DataFrame, out_path: str | Path):
    apply_theme()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # ---- panel 1: observed vs null, forest style -----------------------
    ax = axes[0]
    d = res.sort_values(["direction", "auroc_real"]).reset_index(drop=True)
    y = np.arange(len(d))
    for i, r in d.iterrows():
        lo = r["auroc_null_mean"] - 2 * r["auroc_null_sd"]
        hi = r["auroc_null_mean"] + 2 * r["auroc_null_sd"]
        ax.plot([lo, hi], [i, i], color=MUTED, lw=5, alpha=0.35,
                solid_capstyle="butt", zorder=1)
        ax.plot([r["auroc_null_mean"]], [i], "|", color=MUTED, ms=9, zorder=2)
        col = SIG_C if abs(r["auroc_z"]) > 2 else NS_C
        ax.plot([r["auroc_real"]], [i], "o", color=col, ms=7,
                mec="white", mew=1.2, zorder=3)
    ax.axvline(0.5, color=INK, lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['signature']}  {r['direction']}  (n={r['n_genes']})"
                        for _, r in d.iterrows()], fontsize=7.4)
    ax.set_xlabel("AUROC  cancer epithelium > stroma", fontsize=9, color=INK)
    ax.set_title("Observed vs matched null (grey = null mean ± 2 SD)",
                 fontsize=9.8, color=INK, loc="left", pad=8)
    ax.grid(axis="x", alpha=0.5)
    ax.set_axisbelow(True)

    # ---- panel 2: the power limit --------------------------------------
    ax = axes[1]
    ax.scatter(d["n_genes"], d["auroc_null_sd"], s=54,
               color=[SIG_C if abs(z) > 2 else NS_C for z in d["auroc_z"]],
               edgecolor="white", lw=1.1, zorder=3)
    for _, r in d.iterrows():
        ax.annotate(r["signature"][:14], (r["n_genes"], r["auroc_null_sd"]),
                    fontsize=6.6, color=MUTED,
                    xytext=(5, 3), textcoords="offset points")
    xs = np.linspace(max(5, d["n_genes"].min()), d["n_genes"].max(), 100)
    ax.plot(xs, 0.39 / np.sqrt(xs), color=INK, lw=1.0, ls=":", alpha=0.6,
            label=r"$\propto 1/\sqrt{n}$")
    ax.set_xscale("log")
    ax.set_xlabel("genes in signature", fontsize=9, color=INK)
    ax.set_ylabel("null SD of AUROC", fontsize=9, color=INK)
    ax.set_title("Small signatures cannot reach significance",
                 fontsize=9.8, color=INK, loc="left", pad=8)
    ax.legend(fontsize=7.4, frameon=False)
    ax.grid(alpha=0.5)
    ax.set_axisbelow(True)

    # ---- panel 3: bimodality -------------------------------------------
    ax = axes[2]
    up = res[res["direction"] == "up"].sort_values("bimodality_real")
    y3 = np.arange(len(up))
    ax.barh(y3 - 0.19, up["bimodality_null_mean"].values, height=0.36,
            color=MUTED, alpha=0.6, label="matched null")
    ax.barh(y3 + 0.19, up["bimodality_real"].values, height=0.36,
            color=[SIG_C if p >= 0.95 else NS_C
                   for p in up["bimodality_percentile"]],
            label="observed")
    ax.axvline(0.555, color=INK, lw=1.0, ls="--")
    ax.text(0.558, len(up) - 0.5, "Sarle 0.555", fontsize=6.8, color=INK,
            rotation=90, va="top")
    ax.set_yticks(y3)
    ax.set_yticklabels(up["signature"], fontsize=7.8)
    ax.set_xlabel("bimodality coefficient, non-cycling cancer cells",
                  fontsize=9, color=INK)
    ax.set_title("Sub-structure within cancer epithelium",
                 fontsize=9.8, color=INK, loc="left", pad=8)
    ax.legend(fontsize=7.4, frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.5)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Figure 4 — signature statistics against size- and expression-matched "
        "null gene sets",
        fontsize=11, color=INK, x=0.005, ha="left", y=0.998)
    fig.text(0.005, 0.004,
             "Red = departs from null (|z| > 2, or bimodality percentile ≥ 0.95). "
             "Grey = indistinguishable from random gene sets of matched size and "
             "expression — absence of evidence, not evidence of absence.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.030, 1, 0.950])

    path = save_svg(fig, out_path)
    plt.close(fig)
    return path
