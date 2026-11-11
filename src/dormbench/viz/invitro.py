"""Figure 6 — leading-edge genes across the in vitro adaptation arc."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .theme import GRID, INK, MUTED, apply_theme, save_svg

ORDER = ["T0", "Untreated", "Latency", "Dormancy", "Awakening", "TEP"]
LE_C, REST_C, UNT_C = "#c1272d", "#9aa0ad", "#1b6ca8"


def plot_invitro(expr: pd.DataFrame, grp: pd.Series, le_genes: list,
                 rest: list, per_df: pd.DataFrame, signature: str,
                 out_path: str | Path):
    apply_theme()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    def series(genes):
        xs, ys, lo, hi = [], [], [], []
        for i, g in enumerate(ORDER):
            cols = grp.index[grp == g]
            if len(cols) == 0 or len(genes) == 0:
                continue
            v = expr.loc[genes, cols].apply(pd.to_numeric, errors="coerce")
            per_sample = v.mean(axis=0)
            xs.append(i); ys.append(per_sample.mean())
            lo.append(per_sample.min()); hi.append(per_sample.max())
        return xs, ys, lo, hi

    # panel 1: trajectory
    ax = axes[0]
    for genes, col, lab in ((le_genes, LE_C, "leading edge"),
                            (rest, REST_C, "other 'up' genes")):
        xs, ys, lo, hi = series(genes)
        if xs:
            ax.fill_between(xs, lo, hi, color=col, alpha=0.14, lw=0)
            ax.plot(xs, ys, "-o", color=col, ms=5.5, lw=2.1, mfc="white",
                    mew=1.7, label=lab, zorder=3)
    ax.axvspan(0.5, 1.5, color=UNT_C, alpha=0.08, lw=0, zorder=0)
    ax.text(1, ax.get_ylim()[1], "time-matched\ncontrol", fontsize=6.8,
            color=UNT_C, ha="center", va="top")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, rotation=25, ha="right", fontsize=7.8)
    ax.set_ylabel("mean log2 expression", fontsize=9, color=INK)
    ax.set_title(f"{signature} up genes across the arc", fontsize=9.8,
                 color=INK, loc="left", pad=8)
    ax.legend(fontsize=7.4, frameon=False)
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)

    # panel 2: dormancy vs untreated induction, per gene
    ax = axes[1]
    ax.scatter(per_df["untr_minus_T0"], per_df["dorm_minus_T0"], s=42,
               color=LE_C, alpha=0.8, edgecolor="white", lw=1.0, zorder=3)
    lim = [min(per_df["untr_minus_T0"].min(), per_df["dorm_minus_T0"].min()) - 0.3,
           max(per_df["untr_minus_T0"].max(), per_df["dorm_minus_T0"].max()) + 0.3]
    ax.plot(lim, lim, color=INK, lw=1.0, ls="--", alpha=0.7)
    ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
    for _, r in per_df.head(8).iterrows():
        ax.annotate(r["gene"], (r["untr_minus_T0"], r["dorm_minus_T0"]),
                    fontsize=6.6, color=MUTED, xytext=(4, 3),
                    textcoords="offset points")
    ax.set_xlabel("untreated − T0  (culture drift)", fontsize=9, color=INK)
    ax.set_ylabel("dormancy − T0  (induction)", fontsize=9, color=INK)
    ax.set_title("Above the diagonal = dormancy-specific", fontsize=9.8,
                 color=INK, loc="left", pad=8)
    ax.grid(alpha=0.5)
    ax.set_axisbelow(True)

    # panel 3: ranked dormancy-vs-untreated
    ax = axes[2]
    d = per_df.sort_values("dorm_minus_untr")
    y = np.arange(len(d))
    ax.barh(y, d["dorm_minus_untr"].values,
            color=[LE_C if v > 0 else REST_C for v in d["dorm_minus_untr"]],
            height=0.72)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(d["gene"], fontsize=6.4)
    ax.set_xlabel("dormancy − untreated (log2)", fontsize=9, color=INK)
    ax.set_title("Dormancy-specific induction, per gene", fontsize=9.8,
                 color=INK, loc="left", pad=8)
    ax.grid(axis="x", alpha=0.5)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Figure 6 — are the tissue leading-edge genes actually induced in the "
        "source experiment?",
        fontsize=11, color=INK, x=0.005, ha="left", y=0.998)
    fig.text(0.005, 0.004,
             "Untreated samples (d30/120/170) are time-matched proliferating "
             "controls: they separate dormancy biology from long-term culture "
             "drift.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.030, 1, 0.950])

    path = save_svg(fig, out_path)
    plt.close(fig)
    return path
