"""Figure 14 - survival analysis: does dormancy add to CTS5?

Panels
    A  Kaplan-Meier recurrence-free survival from the 5-year landmark, by CTS5
       risk group (the incumbent's own stratification).
    B  The same, by dormancy tertile. If the signature carried late-recurrence
       information the curves should separate.
    C  Calibration at the horizon: predicted versus observed risk per quantile,
       for each nested model.
    D  Discrimination: C-index and time-dependent AUC per model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .theme import GRID, INK, MUTED, PALETTE, apply_theme, save_svg

HOT, COOL, GREEN = "#c1272d", "#1b6ca8", "#2a9d8f"


def _km_curve(times, events):
    """Kaplan-Meier step curve, returned as (times, survival)."""
    order = np.argsort(times)
    t = np.asarray(times, float)[order]
    e = np.asarray(events, float)[order]
    n = len(t)
    surv, out_t, out_s = 1.0, [0.0], [1.0]
    for i in range(n):
        if e[i]:
            surv *= (1 - 1.0 / (n - i))
            out_t.append(t[i])
            out_s.append(surv)
    out_t.append(max(t.max() if n else 0, 0.0))
    out_s.append(surv)
    return np.array(out_t), np.array(out_s)


def _km_plot(ax, df, group_col, title, colors):
    for i, (name, g) in enumerate(df.groupby(group_col, observed=True)):
        if len(g) < 5:
            continue
        t, s = _km_curve(g["T"].values, g["E"].values)
        ax.step(t, s, where="post", color=colors[i % len(colors)], lw=1.9,
                label=f"{name} (n={len(g)})")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("years from 5-year landmark", fontsize=9, color=INK)
    ax.set_ylabel("recurrence-free survival", fontsize=9, color=INK)
    ax.set_title(title, fontsize=9.6, color=INK, loc="left", pad=8)
    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, loc="lower left")


def plot_survival(lm: pd.DataFrame, fits, models: pd.DataFrame,
                  calibration, horizon_years: float,
                  timevarying: pd.DataFrame, out_path: str | Path):
    apply_theme()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
    axes = axes.ravel()

    # ---- A: CTS5 risk groups --------------------------------------------
    ax = axes[0]
    order = {"low": 0, "intermediate": 1, "high": 2}
    lm = lm.copy()
    lm["cts5_group"] = pd.Categorical(lm["cts5_group"],
                                      categories=["low", "intermediate",
                                                  "high"], ordered=True)
    _km_plot(ax, lm.dropna(subset=["cts5_group"]), "cts5_group",
             "A  Recurrence after 5 years, by CTS5 risk group",
             [GREEN, COOL, HOT])

    # ---- B: dormancy tertiles -------------------------------------------
    ax = axes[1]
    lm["dormancy_tertile"] = pd.qcut(lm["dormancy"], 3,
                                     labels=["low", "middle", "high"],
                                     duplicates="drop")
    _km_plot(ax, lm.dropna(subset=["dormancy_tertile"]), "dormancy_tertile",
             "B  The same, by dormancy-score tertile", [COOL, MUTED, HOT])

    # ---- C: calibration --------------------------------------------------
    ax = axes[2]
    cal = pd.read_csv(calibration) if Path(calibration).exists() else pd.DataFrame()
    if len(cal):
        ax.plot([0, 0.35], [0, 0.35], ls="--", color=INK, lw=1.0)
        for i, (name, g) in enumerate(cal.groupby("model")):
            g = g.sort_values("predicted")
            ax.plot(g["predicted"], g["observed"], "-o", ms=5,
                    color=PALETTE[i % len(PALETTE)], lw=1.6,
                    label=name.replace("_", " "))
        ax.set_xlabel(f"predicted {horizon_years:.0f}-year recurrence risk",
                      fontsize=9, color=INK)
        ax.set_ylabel("observed (Kaplan-Meier)", fontsize=9, color=INK)
        ax.legend(fontsize=7, frameon=False)
    ax.set_title("C  Calibration at the horizon", fontsize=9.6, color=INK,
                 loc="left", pad=8)
    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)

    # ---- D: discrimination ----------------------------------------------
    ax = axes[3]
    x = np.arange(len(models))
    ax.bar(x - 0.2, models["c_index"], width=0.4, color=COOL,
           label="Harrell C")
    if "auc_horizon" in models:
        ax.bar(x + 0.2, models["auc_horizon"], width=0.4, color=GREEN,
               label=f"AUC at {horizon_years:.0f} y")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in models["model"]],
                       fontsize=7.4)
    ax.set_ylim(0.5, max(0.85, float(np.nanmax(models["c_index"])) + 0.05))
    ax.axhline(0.5, color=INK, ls="--", lw=1.0)
    ax.set_ylabel("discrimination", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("D  Adding dormancy to CTS5", fontsize=9.6, color=INK,
                 loc="left", pad=8)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return save_svg(fig, out_path)
