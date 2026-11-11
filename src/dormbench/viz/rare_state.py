"""Figure 13 - is there a reproducible rare TRADITIOM-high malignant state?

Panels
    A  TRADITIOM-up distribution among non-cycling malignant cells, per ER+
       donor (the population the tail is drawn from).
    B  tail versus remainder, per donor: median score gap at the headline
       threshold. A tail that barely separates is not a state.
    C  pathway AUROC of tail versus remainder, one dot per donor, with the
       same-size random-subset null as a grey band. 0.5 = indistinguishable.
    D  mean effect (AUROC - 0.5) by pathway and threshold: the stability check.
    E  stromal-marker panel AUROC (only when the marker panel was computed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .theme import GRID, INK, MUTED, PALETTE, apply_theme, save_svg

HOT, COOL, GREEN = "#c1272d", "#1b6ca8", "#2a9d8f"
PRIMARY_TAG = "TRADITIOM_dormancy|up"


def plot_rare_state(df: pd.DataFrame, donors: list[str], pool_mask,
                    primary: str, donor_table: pd.DataFrame,
                    composition: pd.DataFrame,
                    summary: pd.DataFrame, thresholds, headline_q: float,
                    marker: pd.DataFrame | None,
                    marker_table: pd.DataFrame | None,
                    specificity: pd.DataFrame | None = None,
                    out_path: str | Path = "fig13_rare_state.svg"):
    apply_theme()
    import matplotlib.pyplot as plt

    ncols = 3
    fig, axes = plt.subplots(2, ncols, figsize=(16.5, 8.6))
    axes = axes.ravel()
    prim = pd.to_numeric(df[primary], errors="coerce").values
    donor_arr = df["donor_id"].astype(str).values

    # ---- A: the distribution the tail is drawn from ----------------------
    ax = axes[0]
    data, labels = [], []
    for d in donors:
        m = pool_mask & (donor_arr == d)
        v = prim[m]
        v = v[np.isfinite(v)]
        if len(v) >= 30:
            data.append(v)
            labels.append(d)
    if data:
        parts = ax.violinplot(data, showextrema=False, widths=0.85)
        for i, b in enumerate(parts["bodies"]):
            b.set_facecolor(PALETTE[i % len(PALETTE)])
            b.set_alpha(0.55)
            b.set_edgecolor("none")
        for i, v in enumerate(data, start=1):
            ax.plot([i], [np.median(v)], "_", color=INK, ms=13, mew=2.0)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("TRADITIOM-up score (AUCell)", fontsize=9, color=INK)
    ax.set_title("A  Non-cycling malignant cells, by ER+ donor",
                 fontsize=9.6, color=INK, loc="left", pad=8)
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)

    # ---- B: does the tail actually separate? ----------------------------
    ax = axes[1]
    comp = composition[composition["threshold"] == headline_q]
    if len(comp):
        x = np.arange(len(comp))
        ax.vlines(x, comp["median_primary_rest"], comp["median_primary_tail"],
                  color=GRID, lw=6, zorder=1)
        ax.plot(x, comp["median_primary_rest"], "o", color=MUTED, ms=7,
                label="remainder", zorder=3)
        ax.plot(x, comp["median_primary_tail"], "o", color=HOT, ms=7,
                label=f"top {headline_q:.0%}", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(comp["donor_id"], rotation=60, ha="right", fontsize=7)
        ax.legend(fontsize=7.4, frameon=False, loc="upper right")
    ax.set_ylabel("median TRADITIOM-up score", fontsize=9, color=INK)
    ax.set_title(f"B  Tail versus remainder at top {headline_q:.0%}",
                 fontsize=9.6, color=INK, loc="left", pad=8)
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)

    # ---- C: pathway separation with the same-size null ------------------
    ax = axes[2]
    hz = donor_table[donor_table["threshold"] == headline_q]
    order = (hz.groupby("pathway")["auroc"].median()
             .sort_values().index.tolist())
    for i, label in enumerate(order):
        blk = hz[hz["pathway"] == label]
        lo = float(np.nanmedian(blk["null_lo"]))
        hi = float(np.nanmedian(blk["null_hi"]))
        ax.plot([lo, hi], [i, i], color=GRID, lw=5, solid_capstyle="round",
                zorder=1)
        ax.scatter(blk["auroc"], np.full(len(blk), i), s=22,
                   color=[HOT if v > 0.5 else COOL for v in blk["auroc"]],
                   alpha=0.85, zorder=3, linewidths=0)
        ax.plot([np.nanmedian(blk["auroc"])], [i], "D", color=INK, ms=5,
                zorder=4)
    ax.axvline(0.5, color=INK, lw=1.2, ls="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7.4)
    ax.set_xlabel("AUROC  tail > remainder   (grey = same-size null)",
                  fontsize=9, color=INK)
    ax.set_title("C  Pathway separation at the headline threshold",
                 fontsize=9.6, color=INK, loc="left", pad=8)
    ax.grid(axis="x", alpha=0.5)
    ax.set_axisbelow(True)

    # ---- D: threshold stability heatmap ---------------------------------
    ax = axes[3]
    paths = sorted(summary["pathway"].unique())
    grid = np.full((len(paths), len(thresholds)), np.nan)
    for j, q in enumerate(thresholds):
        blk = summary[summary["threshold"] == q].set_index("pathway")
        for i, p in enumerate(paths):
            if p in blk.index:
                grid[i, j] = blk.loc[p, "mean_effect"]
    lim = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"{q:.0%}" for q in thresholds], fontsize=8)
    ax.set_yticks(range(len(paths)))
    ax.set_yticklabels(paths, fontsize=7.2)
    ax.set_xlabel("tail threshold", fontsize=9, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cb.set_label("mean AUROC − 0.5", fontsize=7.6, color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_title("D  Effect by pathway and threshold",
                 fontsize=9.6, color=INK, loc="left", pad=8)

    # ---- E: is the profile specific to this signature? ------------------
    ax = axes[4]
    if specificity is not None and len(specificity):
        st = specificity[specificity["is_state_pathway"]]
        piv = (st.groupby(["defining_signature", "pathway"])["auroc"]
               .median().unstack())
        if PRIMARY_TAG in piv.index and len(piv) > 1:
            order = piv.loc[PRIMARY_TAG].sort_values().index
            xs = np.arange(len(order))
            others = [t for t in piv.index if t != PRIMARY_TAG]
            for j, tag in enumerate(others):
                ax.plot(xs, piv.loc[tag, order].values, "o", ms=5,
                        color=PALETTE[j % len(PALETTE)], alpha=0.8,
                        label=tag.replace("|up", ""), linewidth=0)
            ax.plot(xs, piv.loc[PRIMARY_TAG, order].values, "D", ms=8,
                    color=HOT, label="TRADITIOM-up", zorder=4)
            ax.axhline(0.5, color=INK, lw=1.2, ls="--")
            ax.set_xticks(xs)
            ax.set_xticklabels(order, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel("median AUROC of the tail", fontsize=9, color=INK)
            ax.legend(fontsize=6.8, frameon=False, loc="upper left")
            ax.set_title("E  Is the profile specific? Tails from other\n"
                         "signatures show the same programme",
                         fontsize=9.6, color=INK, loc="left", pad=8)
            ax.grid(axis="y", alpha=0.5)
            ax.set_axisbelow(True)
        else:
            ax.axis("off")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "specificity controls\nnot computed",
                ha="center", va="center", fontsize=9, color=MUTED)

    # ---- F: verdict summary text ----------------------------------------
    ax = axes[5]
    ax.axis("off")
    n_don = int(hz["donor_id"].nunique()) if len(hz) else 0
    sig = summary[(summary["threshold"] == headline_q)
                  & (summary["is_state_pathway"]) & (summary["q"] < 0.05)]
    lines = [f"ER+ donors: {n_don}",
             f"headline threshold: top {headline_q:.0%}",
             f"state pathways with q < 0.05: {len(sig)}"]
    if len(sig):
        for _, r in sig.sort_values("median_auroc").iterrows():
            lines.append(f"   {r['pathway']}: AUROC {r['median_auroc']:.2f}"
                         f", {int(r['n_above_half'])}/{int(r['n_donors'])} donors up")
    if marker_table is not None and len(marker_table):
        mh = marker_table[marker_table["threshold"] == headline_q]
        if len(mh):
            lines.append("")
            lines.append("stromal markers in the tail (AUROC):")
            for p, blk in mh.groupby("panel"):
                lines.append(f"   {p}: {blk['auroc'].median():.2f}")
    ax.text(0.0, 0.95, "\n".join(lines), va="top", ha="left", fontsize=8,
            color=INK, linespacing=1.6)
    ax.set_title("F  What this figure supports", fontsize=9.6, color=INK,
                 loc="left", pad=8)

    banner_text = ("Compartment averages can hide a rare state. The tail is "
                   "compared with the rest of the same donor's malignant cells "
                   "and with a random subset of the same size; the donor is the "
                   "replicate.")
    fig.text(0.005, -0.01, banner_text, fontsize=7.4, color=MUTED)
    fig.tight_layout()
    return save_svg(fig, out_path)
