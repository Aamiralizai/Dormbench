"""Figure: repressive histone marks across the endocrine-therapy adaptation arc."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .theme import GRID, INK, MUTED, apply_theme, save_svg

STAGE_ORDER = ["T0", "Lat7", "Lat14", "Dorm30", "Dorm60", "Awake", "TEP"]
STAGE_LABELS = ["POT\n(naive)", "Lat\nd7", "Lat\nd14", "Dorm\nd30",
                "Dorm\nd60", "Awake", "TEP"]

PANELS = [
    ("H4K20me3", "THE SIGNAL — large rise, sharp fall", True),
    ("H3K27me3", "Erodes EARLY (peaks d30, gone by d60)", False),
    ("H3K9me2", "Real but small dynamic range", False),
    ("H3K9me3", "NEGATIVE CONTROL — flat throughout", False),
]

AI_COL, TAM_COL, ACCENT = "#c1272d", "#1b6ca8", "#c1272d"


def plot_erosion(mcf7: pd.DataFrame, meta: pd.DataFrame,
                 t47d: pd.DataFrame, t_stage: pd.Series,
                 out_path: str | Path):
    apply_theme()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, len(PANELS), figsize=(15.5, 4.3))

    t47_map = {"T0": "T0", "Dorm30": "Dorm", "TEP": "TEP"}

    for ax, (mark, note, is_key) in zip(axes, PANELS):
        if mark not in mcf7.index:
            ax.set_visible(False)
            continue

        for arm, col in (("AI", AI_COL), ("TAM", TAM_COL)):
            xs, ys, lo, hi = [], [], [], []
            for i, s in enumerate(STAGE_ORDER):
                m = meta.stage.eq(s) & (meta.arm.eq(arm) | meta.arm.eq("POT"))
                v = mcf7.loc[mark, meta.index[m]].dropna()
                if len(v):
                    xs.append(i)
                    ys.append(v.mean())
                    lo.append(v.min())
                    hi.append(v.max())
            if xs:
                ax.fill_between(xs, lo, hi, color=col, alpha=0.13, lw=0)
                ax.plot(xs, ys, "-o", color=col, ms=5, lw=2.1,
                        mfc="white", mew=1.8, zorder=3)

        if mark in t47d.index:
            xs2, ys2 = [], []
            for i, s in enumerate(STAGE_ORDER):
                if s in t47_map:
                    v = t47d.loc[mark, t_stage.index[t_stage == t47_map[s]]].dropna()
                    if len(v):
                        xs2.append(i)
                        ys2.append(v.mean())
            if xs2:
                ax.plot(xs2, ys2, "--s", color=MUTED, ms=4.5, lw=1.5,
                        mfc="white", mew=1.4, zorder=2)

        ax.axhline(1.0, color=GRID, lw=1, ls=":", zorder=0)
        ax.axvspan(2.5, 4.5, color="#f5c518", alpha=0.10, lw=0, zorder=0)
        ax.set_xticks(range(len(STAGE_ORDER)))
        ax.set_xticklabels(STAGE_LABELS, fontsize=7.5, color=MUTED)
        ax.set_title(mark, fontsize=12, color=INK, pad=9,
                     fontweight="bold", loc="left")
        ax.text(0, 1.005, note, transform=ax.transAxes, fontsize=7.4,
                color=ACCENT if is_key else MUTED, va="bottom", style="italic")
        ax.tick_params(labelsize=8, length=3)
        ax.grid(axis="y", alpha=0.6)
        ax.set_axisbelow(True)
        if is_key:
            ax.set_ylabel("L/H ratio (relative abundance)", fontsize=9, color=INK)

    axes[0].legend(
        handles=[
            Line2D([], [], color=AI_COL, marker="o", mfc="white", mew=1.6,
                   lw=2, label="MCF7 −E2 (AI)"),
            Line2D([], [], color=TAM_COL, marker="o", mfc="white", mew=1.6,
                   lw=2, label="MCF7 TAM"),
            Line2D([], [], color=MUTED, marker="s", mfc="white", mew=1.4,
                   lw=1.5, ls="--", label="T47D −E2"),
        ],
        fontsize=7.6, frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.90),
    )

    fig.suptitle(
        "Repressive histone marks across the ET adaptation arc  —  "
        "Rosano et al. 2024, Table S8",
        fontsize=10.5, color=INK, x=0.006, ha="left", y=0.995,
    )
    fig.text(0.006, 0.012,
             "Shaded band = min–max across replicates.  Yellow = dormancy window.  "
             "Dotted line = L/H 1.0 (no change vs internal standard).",
             fontsize=7.3, color=MUTED)
    fig.tight_layout(rect=[0, 0.035, 1, 0.945])

    path = save_svg(fig, out_path)
    import matplotlib.pyplot as plt2
    plt2.close(fig)
    return path
