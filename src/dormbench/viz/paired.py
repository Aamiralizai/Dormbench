"""Figure 9 — paired on-treatment biopsies: state change or composition change."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .theme import GRID, INK, MUTED, apply_theme, save_svg

CANCER = "Cancer Epithelial"
HOT, COOL, CTRL, GREY = "#c1272d", "#1b6ca8", "#2a9d8f", "#9aa0ad"
ORDER = ["Cancer Epithelial", "Normal Epithelial", "CAFs", "PVL", "Endothelial",
         "T-cells", "Myeloid", "B-cells", "Plasmablasts"]


def _cls(name: str) -> str:
    if "E2F" in name or "G2M" in name:
        return "control"
    if name.startswith(("TRADITIOM_dormancy|up", "PreAdapted|up",
                        "NR2F1", "BHLHE41", "Quiescence_core|up")):
        return "dormancy"
    return "other"


def plot_paired(pairs: pd.DataFrame, deltas: pd.DataFrame,
                out_path: str | Path, focus: str = "TRADITIOM_dormancy|up"):
    apply_theme()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))

    # ---- panel 1: what happened to the tumour --------------------------
    ax = axes[0]
    ccols = [c for c in pairs.columns if c.startswith("d_comp_")]
    names = [c.replace("d_comp_", "") for c in ccols]
    means = [float(pairs[c].mean()) for c in ccols]
    order = [n for n in ORDER if n in names] + [n for n in names if n not in ORDER]
    idx = [names.index(n) for n in order]
    vals = [means[i] for i in idx]
    cols = [HOT if order[k] == CANCER else (COOL if vals[k] > 0 else GREY)
            for k in range(len(order))]
    y = np.arange(len(order))
    ax.barh(y, vals, color=cols, height=0.68, alpha=0.9)
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlabel("mean change on treatment (z)", fontsize=9, color=INK)
    ax.set_title("Endocrine therapy removes cancer cells;\nstroma fills the space",
                 fontsize=9.6, color=INK, loc="left", pad=8)
    ax.grid(axis="x", alpha=0.5)
    ax.set_axisbelow(True)

    # ---- panel 2: per patient, score change vs composition change ------
    ax = axes[1]
    col = f"d_{focus}"
    if col in pairs.columns:
        x = pairs["d_stroma"].values.astype(float)
        yv = pairs[col].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(yv)
        ax.scatter(x[ok], yv[ok], s=46, color=HOT, alpha=0.75,
                   edgecolor="white", lw=1.0, zorder=3)
        if ok.sum() > 3:
            b, a = np.polyfit(x[ok], yv[ok], 1)
            xs = np.linspace(x[ok].min(), x[ok].max(), 50)
            ax.plot(xs, a + b * xs, color=INK, lw=1.6, zorder=4)
            ax.plot([0], [a], "o", color="white", ms=9, mec=INK, mew=1.6,
                    zorder=5)
            ax.annotate(f"Δ at Δstroma=0\n= {a:+.3f}", (0, a),
                        xytext=(14, -26), textcoords="offset points",
                        fontsize=7.6, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
            from scipy import stats as _st
            r = _st.spearmanr(x[ok], yv[ok])[0]
            ax.text(0.03, 0.95, f"r = {r:+.2f}   n = {int(ok.sum())}",
                    transform=ax.transAxes, fontsize=8.4, color=INK,
                    va="top")
        ax.axhline(0, color=GRID, lw=1)
        ax.axvline(0, color=GRID, lw=1)
        ax.set_xlabel("change in stromal content", fontsize=9, color=INK)
        ax.set_ylabel(f"change in {focus.split('|')[0]} score", fontsize=9,
                      color=INK)
        ax.set_title("Each point is one patient, before vs on treatment",
                     fontsize=9.6, color=INK, loc="left", pad=8)
        ax.grid(alpha=0.5)
        ax.set_axisbelow(True)

    # ---- panel 3: how much survives adjustment --------------------------
    ax = axes[2]
    d = deltas.copy()
    if "fraction_retained" not in d.columns:
        d["fraction_retained"] = d["delta_adjusted"] / d["delta_treated"]
    d = d[np.isfinite(d["fraction_retained"])]
    d["class"] = d["score"].map(_cls)
    d = d.sort_values("fraction_retained")
    y3 = np.arange(len(d))
    cmap = {"control": CTRL, "dormancy": HOT, "other": GREY}
    ax.barh(y3, d["fraction_retained"].clip(-0.2, 1.2).values,
            color=[cmap[c] for c in d["class"]], height=0.68, alpha=0.9)
    ax.axvline(1.0, color=INK, lw=1.0, ls="--")
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_yticks(y3)
    ax.set_yticklabels(d["score"], fontsize=7.0)
    ax.set_xlabel("fraction of change surviving adjustment", fontsize=9,
                  color=INK)
    ax.set_title("1.0 = unaffected by composition, 0 = entirely composition",
                 fontsize=9.6, color=INK, loc="left", pad=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=CTRL, label="proliferation control"),
                       Patch(color=HOT, label="dormancy"),
                       Patch(color=GREY, label="other programmes")],
              fontsize=7.2, frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.5)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Figure 9 — in paired biopsies on endocrine therapy, much of the "
        "apparent dormancy induction is a composition shift",
        fontsize=11, color=INK, x=0.005, ha="left", y=0.998)
    fig.text(0.005, 0.004,
             "Proliferation controls retain most of their change — a real drug "
             "effect — confirming the adjustment does not simply flatten "
             "everything.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.030, 1, 0.950])

    path = save_svg(fig, out_path)
    plt.close(fig)
    return path
