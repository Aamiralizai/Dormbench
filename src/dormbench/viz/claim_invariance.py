"""Figure 7 - claim-invariance analyses.

Panels match the legend in Section 3.19 / Method S18:
    A  compartment AUROC for the dormancy panel under the nine cell-cycle
       filtering rules
    B  the same for the E2F and G2M controls, which shows that control
       neutrality is a property of the filter rather than of the gene set
    C  the fifty MSigDB Hallmark modules, with each dormancy signature placed
       by percentile
    D  held-out donor AUROC from the fully nested redesign, and the same folds
       rescored with the fitted linear predictor
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .theme import GRID, INK, MUTED, PALETTE, apply_theme, save_svg

HOT, COOL, GREEN = "#c1272d", "#1b6ca8", "#2a9d8f"

SCHEME_ORDER = ["none", "annotation", "top10pct", "top20pct", "top25pct",
                "top30pct", "top40pct", "residualised", "g2m_matched"]
SCHEME_LABEL = {"none": "none", "annotation": "cycling\nremoved",
                "top10pct": "10%", "top20pct": "20%", "top25pct": "25%\n(default)",
                "top30pct": "30%", "top40pct": "40%",
                "residualised": "resid.", "g2m_matched": "G2M\nmatched"}


def plot_claim_invariance(multiverse_csv, hallmark_csv, module_null_csv,
                          folds_csv, v_score_csv, out_path: str | Path,
                          png_path: str | Path | None = None):
    apply_theme()
    import matplotlib.pyplot as plt

    mv = pd.read_csv(multiverse_csv)
    hm = pd.read_csv(hallmark_csv)
    nd = pd.read_csv(module_null_csv)
    folds = pd.read_csv(folds_csv)
    vs = pd.read_csv(v_score_csv)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    axes = axes.ravel()
    x = np.arange(len(SCHEME_ORDER))

    # ---- A: dormancy panel across filtering rules -----------------------
    ax = axes[0]
    dorm = ["TRADITIOM_dormancy", "PreAdapted", "EMT_hallmark",
            "Quiescence_core"]
    for i, sig in enumerate(dorm):
        w = mv[mv["signature"] == sig].set_index("scheme")
        y = [w["auroc"].get(s, np.nan) for s in SCHEME_ORDER]
        ax.plot(x, y, "-o", ms=5, lw=1.8, color=PALETTE[i % len(PALETTE)],
                label=sig.replace("_", " "), mfc="white", mew=1.4)
    ax.axhline(0.5, color=INK, ls="--", lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([SCHEME_LABEL[s] for s in SCHEME_ORDER], fontsize=7)
    ax.set_ylabel("AUROC  cancer vs stroma", fontsize=9, color=INK)
    ax.set_title("A  Dormancy panel across filtering rules",
                 fontsize=9.6, color=INK, loc="left", pad=8)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.text(0.98, 0.96, "below 0.5 = stromal", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color=MUTED)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)

    # ---- B: controls ----------------------------------------------------
    ax = axes[1]
    ctrl = ["E2F_control", "G2M_control", "G2M_hallmark_control"]
    for i, sig in enumerate(ctrl):
        w = mv[mv["signature"] == sig].set_index("scheme")
        if w.empty:
            continue
        y = [w["auroc"].get(s, np.nan) for s in SCHEME_ORDER]
        ax.plot(x, y, "-s", ms=5, lw=1.8, color=(HOT if i == 0 else
                                                 (COOL if i == 1 else GREEN)),
                label=sig.replace("_", " "), mfc="white", mew=1.4)
        ax.annotate(f"{y[0]:.2f}", (x[0], y[0]), xytext=(-6, 6),
                    textcoords="offset points", fontsize=7, color=INK)
    ax.axhline(0.5, color=INK, ls="--", lw=1.1)
    ax.axhspan(0.4, 0.6, color=GREEN, alpha=0.10, lw=0)
    ax.set_xticks(x)
    ax.set_xticklabels([SCHEME_LABEL[s] for s in SCHEME_ORDER], fontsize=7)
    ax.set_ylabel("AUROC  cancer vs stroma", fontsize=9, color=INK)
    ax.set_title("B  The controls are neutral only after filtering",
                 fontsize=9.6, color=INK, loc="left", pad=8)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.text(0.98, 0.96, "shaded = 0.45-0.55", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color=MUTED)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)

    # ---- C: Hallmark comparator ----------------------------------------
    ax = axes[2]
    rng = np.random.default_rng(0)
    yj = rng.uniform(-0.22, 0.22, len(hm))
    ax.scatter(hm["auroc"], yj, s=26,
               color=[PALETTE[4] if p else MUTED for p in hm["proliferative"]],
               alpha=0.75, linewidths=0)
    ax.axvline(float(hm["auroc"].median()), color=INK, ls="--", lw=1.0)
    ax.text(float(hm["auroc"].median()), 0.34,
            f"Hallmark median {hm['auroc'].median():.2f}",
            fontsize=7, color=INK, ha="center")
    up = nd[(nd["direction"] == "up")
            & nd["signature"].isin(["TRADITIOM_dormancy", "PreAdapted",
                                    "NR2F1_axis", "BHLHE41_dormancy",
                                    "Quiescence_core"])]
    short = {"TRADITIOM_dormancy": "TRADITIOM", "PreAdapted": "PreAdapted",
             "NR2F1_axis": "NR2F1", "BHLHE41_dormancy": "BHLHE41",
             "Quiescence_core": "Quiescence"}
    for k, (_, r) in enumerate(up.sort_values("auroc").iterrows()):
        y_lab = -0.30 if k % 2 == 0 else -0.58
        ax.scatter([r["auroc"]], [y_lab], s=75, marker="D",
                   color=HOT if r["signature"] == "TRADITIOM_dormancy"
                   else COOL, zorder=5, linewidths=0)
        ax.annotate(short.get(r["signature"], r["signature"]),
                    (r["auroc"], y_lab), xytext=(0, -11),
                    textcoords="offset points", fontsize=7, color=INK,
                    ha="center")
    ax.set_ylim(-0.85, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("AUROC  cancer vs stroma", fontsize=9, color=INK)
    ax.set_title("C  Dormancy signatures sit inside the Hallmark distribution",
                 fontsize=9.6, color=INK, loc="left", pad=8)
    ax.grid(axis="x", alpha=0.4)
    ax.set_axisbelow(True)

    # ---- D: nested folds, baseline vs linear predictor ------------------
    ax = axes[3]
    order = np.argsort(folds["test_auroc"].values)
    yb = folds["test_auroc"].values[order]
    donors = folds["held_out_donor"].values[order]
    xs = np.arange(len(yb))
    vm = vs.set_index("held_out_donor")["auroc_model"]
    ym = np.array([vm.get(d, np.nan) for d in donors])
    for i in xs:
        ax.plot([i, i], [yb[i], ym[i]], color=GRID, lw=1.4, zorder=1)
    ax.scatter(xs, yb, s=34, color=COOL, zorder=3, linewidths=0,
               label="unweighted gene mean")
    ax.scatter(xs, ym, s=34, color=HOT, marker="D", zorder=3, linewidths=0,
               label="fitted linear predictor")
    ax.axhline(0.5, color=INK, ls="--", lw=1.1)
    ax.axhspan(0.45, 0.55, color=GREEN, alpha=0.10, lw=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(d)[:7] for d in donors], rotation=90, fontsize=5.6)
    ax.set_ylabel("held-out donor AUROC", fontsize=9, color=INK)
    ax.set_title("D  Nested folds are biased in both directions",
                 fontsize=9.6, color=INK, loc="left", pad=8)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = save_svg(fig, out_path)
    if png_path is not None:
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    return out
