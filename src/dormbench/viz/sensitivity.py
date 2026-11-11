"""Figure 8 — leave-one-out and resampling sensitivity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .theme import INK, MUTED, apply_theme, save_svg

HOT, COOL, GREY = "#c1272d", "#1b6ca8", "#9aa0ad"


def plot_sensitivity(summ: pd.DataFrame, donors: pd.DataFrame,
                     out_path: str | Path):
    apply_theme()
    import matplotlib.pyplot as plt

    d = summ.copy()
    d["label"] = d["signature"] + " " + d["direction"]
    d = d.sort_values("observed")
    y = np.arange(len(d))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # panel 1: observed with bootstrap CI and LOO range
    ax = axes[0]
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot([r["loo_min"], r["loo_max"]], [i, i], color=GREY, lw=5,
                alpha=0.4, solid_capstyle="butt", zorder=1)
        ax.plot([r["boot_ci_low"], r["boot_ci_high"]], [i, i], color=COOL,
                lw=2.0, solid_capstyle="butt", zorder=2)
        ax.plot([r["observed"]], [i], "o", color=HOT, ms=6, mec="white",
                mew=1.1, zorder=3)
    ax.axvline(0.5, color=INK, lw=1.0, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=7.4)
    ax.set_xlabel("AUROC  cancer > stroma", fontsize=9, color=INK)
    ax.set_title("Grey = leave-one-gene range, blue = bootstrap 95% CI",
                 fontsize=9.6, color=INK, loc="left", pad=8)
    ax.grid(axis="x", alpha=0.5)
    ax.set_axisbelow(True)

    # panel 2: magnitude of single-gene vs single-donor effects
    ax = axes[1]
    ax.scatter(d["max_gene_delta"], d["max_donor_delta"], s=54, color=HOT,
               edgecolor="white", lw=1.1, zorder=3)
    lim = max(float(d["max_gene_delta"].max()),
              float(d["max_donor_delta"].max())) * 1.15 + 1e-3
    ax.plot([0, lim], [0, lim], color=INK, lw=1.0, ls="--", alpha=0.7)
    for _, r in d.iterrows():
        ax.annotate(r["signature"][:16],
                    (r["max_gene_delta"], r["max_donor_delta"]),
                    fontsize=6.6, color=MUTED, xytext=(4, 3),
                    textcoords="offset points")
    ax.set_xlabel("largest shift from dropping one gene", fontsize=9, color=INK)
    ax.set_ylabel("largest shift from dropping one donor", fontsize=9, color=INK)
    ax.set_title("Above the diagonal = patient-driven, not gene-driven",
                 fontsize=9.6, color=INK, loc="left", pad=8)
    ax.grid(alpha=0.5)
    ax.set_axisbelow(True)

    # panel 3: donor-LOO spread per signature
    ax = axes[2]
    if len(donors):
        dd = donors.copy()
        dd["label"] = dd["signature"] + " " + dd["direction"]
        order = [l for l in d["label"] if l in set(dd["label"])]
        data = [dd.loc[dd["label"] == l, "delta"].values for l in order]
        if data:
            bp = ax.boxplot(data, vert=False, widths=0.62, patch_artist=True,
                            showfliers=True,
                            flierprops=dict(marker="o", ms=2.6, mfc=HOT,
                                            mec="none", alpha=0.7))
            for patch in bp["boxes"]:
                patch.set_facecolor(COOL)
                patch.set_alpha(0.45)
                patch.set_edgecolor("none")
            for med in bp["medians"]:
                med.set_color(INK)
            ax.set_yticks(range(1, len(order) + 1))
            ax.set_yticklabels(order, fontsize=7.4)
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_xlabel("change in AUROC when a donor is removed", fontsize=9,
                  color=INK)
    ax.set_title("Per-donor influence (26 patients)", fontsize=9.6,
                 color=INK, loc="left", pad=8)
    ax.grid(axis="x", alpha=0.5)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Figure 8 — sensitivity: no single gene or patient drives the "
        "compartment attribution",
        fontsize=11, color=INK, x=0.005, ha="left", y=0.998)
    fig.text(0.005, 0.004,
             "The atlas has 100,064 cells but only 26 donors, so the effective "
             "sample size is donors. Panel 3 is the meaningful precision check.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.030, 1, 0.950])

    path = save_svg(fig, out_path)
    plt.close(fig)
    return path
