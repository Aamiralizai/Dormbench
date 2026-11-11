"""Donor-level figure for WP8: one observation per donor.

The reviewer asked for a main figure that shows the donor as the unit of
analysis rather than tens of thousands of cells. This draws, for the primary
signature in ER-positive donors, the paired per-donor pseudobulk score in
cancer epithelium against each stromal compartment, with a connecting line per
donor so the within-donor difference is what the eye follows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .theme import GRID, INK, MUTED, PALETTE, apply_theme, save_svg

CANCER = "Cancer Epithelial"


def plot_donor_level(pseudo: pd.DataFrame, signature: str,
                     comparisons: list[str], out_path: str | Path,
                     cohort_label: str = "ER-positive donors"):
    """Paired per-donor scores: cancer vs each stromal compartment.

    pseudo: long frame with donor_id, celltype_major, signature, mean_score,
            cohort (already restricted to the cohort of interest by the caller).
    """
    import matplotlib.pyplot as plt

    apply_theme()
    sub = pseudo[pseudo["signature"] == signature]
    panels = [c for c in comparisons if c in set(sub["celltype_major"])]
    if not panels:
        return None

    fig, axes = plt.subplots(
        1, len(panels), figsize=(2.7 * len(panels) + 0.6, 4.4), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, comp in zip(axes, panels):
        wide = sub.pivot_table(index="donor_id", columns="celltype_major",
                               values="mean_score")
        if CANCER not in wide.columns or comp not in wide.columns:
            continue
        pair = wide[[CANCER, comp]].dropna()
        n = len(pair)
        lower = int((pair[CANCER] < pair[comp]).sum())
        # jittered x positions for the two compartments
        rng = np.random.default_rng(0)
        for _, row in pair.iterrows():
            c_val, s_val = row[CANCER], row[comp]
            colour = PALETTE[1] if c_val < s_val else PALETTE[5]
            ax.plot([0, 1], [c_val, s_val], color=colour, lw=0.9, alpha=0.55,
                    zorder=1)
            ax.scatter([0, 1], [c_val, s_val], s=22, color=[PALETTE[0], PALETTE[2]],
                       edgecolor="white", lw=0.6, zorder=3)
        # median markers
        ax.scatter([0, 1], [pair[CANCER].median(), pair[comp].median()],
                   marker="_", s=520, color=INK, lw=2.0, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Cancer\nepithelium", comp], fontsize=9, color=INK)
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(f"lower in cancer: {lower}/{n}", fontsize=9.5,
                     color=INK, loc="left", pad=8)
        ax.grid(axis="y", alpha=0.5)

    axes[0].set_ylabel(f"{signature.replace('|', ' ')}  (per-donor mean)",
                       fontsize=9, color=INK)
    fig.suptitle(
        f"Donor-level scores in {cohort_label}  (one line per donor)",
        fontsize=11, color=INK, x=0.01, ha="left", y=1.0)
    fig.text(0.01, -0.02,
             "Each line is one donor. Red = lower in cancer than in the "
             "stromal compartment; grey = higher. Bar = median.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return save_svg(fig, out_path)
