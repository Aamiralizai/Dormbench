"""WP2c — is the compartment effect real, or is it proliferation?

WP2b found that dormancy signatures score lowest in cancer epithelium and
highest in endothelial/PVL/normal-epithelial cells. Before that can be called
a compartment effect, one alternative has to be excluded: dormancy scores are
strongly anti-correlated with cell cycle (r ~ -0.6), and cancer epithelium is
the proliferative compartment. An inverse-proliferation score would rank the
compartments exactly this way with no dormancy biology involved at all.

Four tests, each removing a different part of the confound:

  A. Cycle-matched comparison. Drop every cycling cell, then re-rank
     compartments. If the ordering survives, proliferation composition was not
     driving it.

  B. Residualisation. Regress each dormancy score on the G2M score and re-rank
     on the residual. Removes the linear proliferation component within cells,
     not just between compartments.

  C. Depth confound. Correlate each score with nFeature_RNA. Cancer cells are
     typically deeper-sequenced; mean-based scores (scanpy) are sensitive to
     this in a way rank-based scores (AUCell) are not.

  D. Within-compartment structure. Restricted to non-cycling cancer epithelium,
     does the score identify a distinct subpopulation, or is it unimodal noise?
     A dormancy signature that cannot find dormant cells among cancer cells is
     not measuring dormancy.

Signed scores are also decomposed. `combined = up - down` hides which half
carries the signal; when the down set is several times larger than the up set
the combined score is effectively the negated down set.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import banner, get_logger

CANCER = "Cancer Epithelial"
CYCLING_LABELS = ("Cancer Cycling", "Cycling T-cells", "Cycling")


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _residualise(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residual of y after removing the linear component of x."""
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 10:
        return y
    b, a = np.polyfit(x[ok], y[ok], 1)
    return y - (a + b * x)


def _dip_like(v: np.ndarray) -> float:
    """Crude bimodality index: |skew|-corrected excess kurtosis proxy.

    Values > 0.55 suggest bimodality (Sarle's bimodality coefficient).
    """
    from scipy.stats import kurtosis, skew

    v = v[np.isfinite(v)]
    if len(v) < 50:
        return np.nan
    n = len(v)
    g, k = skew(v), kurtosis(v, fisher=True)
    denom = k + 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    return float((g**2 + 1) / denom) if denom else np.nan


def run(cfg: Config) -> dict:
    log = get_logger("wp2c", cfg.paths.logs)
    cfg.paths.ensure_outputs()

    report: dict = {
        "step": "wp2c",
        "generated": datetime.now().isoformat(timespec="seconds"),
    }

    banner(log, "WP2c — CELL-CYCLE DECONFOUNDING")

    path = cfg.paths.results / "wp2b_cell_scores.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `dormbench wp2b` first.")
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df):,} cells from {path.name}")

    ct = df["celltype_major"].astype(str)
    minor = (df["celltype_minor"].astype(str) if "celltype_minor" in df
             else pd.Series(["?"] * len(df)))

    g2m_col = next((c for c in df.columns if "G2M" in c and c.startswith("AUCell")),
                   next((c for c in df.columns if "G2M" in c), None))
    if g2m_col is None:
        raise ValueError("No G2M score column found; re-run wp2b with the G2M control.")
    log.info(f"Proliferation reference: {g2m_col}")

    combined = [c for c in df.columns if c.endswith("|combined") and "G2M" not in c]
    updown = [c for c in df.columns
              if (c.endswith("|up") or c.endswith("|down")) and "G2M" not in c]
    log.info(f"Dormancy scores: {len(combined)} combined, {len(updown)} directional")
    log.info("")

    # ---- A. cycle-matched ------------------------------------------------
    banner(log, "A. CYCLE-MATCHED COMPARTMENT RANKING")
    is_cycling = minor.str.contains("|".join(CYCLING_LABELS), case=False, na=False)
    g2m = df[g2m_col].values
    hi_g2m = g2m > np.nanquantile(g2m, 0.75)
    keep = ~(is_cycling.values | hi_g2m)
    log.info(f"Dropping {int((~keep).sum()):,} cycling / high-G2M cells "
             f"({100*(~keep).mean():.0f}%), keeping {int(keep.sum()):,}")
    log.info("")

    rows_a = []
    for col in combined:
        sub = df.loc[keep, col].values
        sct = ct[keep].values
        means = {l: float(np.nanmean(sub[sct == l])) for l in np.unique(sct)}
        top = max(means, key=means.get)
        rank = sorted(means, key=means.get, reverse=True).index(CANCER) + 1 \
            if CANCER in means else np.nan
        rows_a.append({"score": col, "top": top, "cancer_rank": rank,
                       "n_compartments": len(means),
                       "cancer_mean": means.get(CANCER, np.nan)})
        log.info(f"{col}")
        log.info(f"   top = {top:20s}   cancer rank {rank}/{len(means)}")
    report["cycle_matched"] = rows_a
    log.info("")

    # ---- B. residualised -------------------------------------------------
    banner(log, "B. RESIDUALISED ON G2M")
    rows_b = []
    for col in combined:
        resid = _residualise(df[col].values, g2m)
        means = {l: float(np.nanmean(resid[ct.values == l])) for l in ct.unique()}
        top = max(means, key=means.get)
        order = sorted(means, key=means.get, reverse=True)
        rank = order.index(CANCER) + 1 if CANCER in means else np.nan
        var_expl = 1 - np.nanvar(resid) / np.nanvar(df[col].values)
        rows_b.append({"score": col, "top": top, "cancer_rank": rank,
                       "variance_explained_by_g2m": float(var_expl)})
        log.info(f"{col}")
        log.info(f"   G2M explains {100*var_expl:5.1f}% of variance")
        log.info(f"   after removal: top = {top:20s}   cancer rank {rank}/{len(means)}")
    report["residualised"] = rows_b
    log.info("")

    # ---- C. sequencing depth --------------------------------------------
    banner(log, "C. SEQUENCING-DEPTH CONFOUND")
    depth_col = next((c for c in ("nFeature_RNA", "nCount_RNA") if c in df.columns),
                     None)
    if depth_col is None:
        log.info("   nFeature_RNA / nCount_RNA not carried through wp2b — skipped.")
        log.info("   (Not fatal: AUCell is rank-based and largely depth-robust.)")
        report["depth"] = "unavailable"
    else:
        dep = df[depth_col].values
        d = {}
        for col in combined:
            r = float(np.corrcoef(df[col].values, dep)[0, 1])
            d[col] = r
            log.info(f"   corr({col}, {depth_col}) = {r:+.3f}")
        report["depth"] = d
    log.info("")

    # ---- D. within cancer epithelium ------------------------------------
    banner(log, "D. WITHIN NON-CYCLING CANCER EPITHELIUM")
    log.info("If a dormancy signature cannot find a subpopulation among cancer")
    log.info("cells, it is not identifying dormant cancer cells.")
    log.info("")
    sel = (ct.values == CANCER) & keep
    log.info(f"n = {int(sel.sum()):,} non-cycling cancer epithelial cells")
    log.info("")
    rows_d = []
    for col in combined:
        v = df.loc[sel, col].values
        bc = _dip_like(v)
        # how separable are the top and bottom deciles
        lo, hi = np.nanquantile(v, [0.1, 0.9])
        spread = float((hi - lo) / (np.nanstd(v) + 1e-9))
        rows_d.append({"score": col, "bimodality_coef": bc,
                       "decile_spread_sd": spread,
                       "cv": float(np.nanstd(v) / (abs(np.nanmean(v)) + 1e-9))})
        flag = "  <== bimodal" if (bc is not np.nan and bc > 0.55) else ""
        log.info(f"   {col:42s} bimodality {bc:.3f}{flag}")
    report["within_cancer"] = rows_d
    log.info("")

    # ---- decomposition ---------------------------------------------------
    banner(log, "E. UP vs DOWN DECOMPOSITION")
    log.info("A combined score dominated by a much larger 'down' set is an")
    log.info("inverted proliferation score, not a dormancy score.")
    log.info("")
    rows_e = []
    for col in sorted(updown):
        r = float(np.corrcoef(df[col].values, g2m)[0, 1])
        rows_e.append({"score": col, "corr_with_g2m": r})
        log.info(f"   corr({col:42s}, G2M) = {r:+.3f}")
    report["decomposition"] = rows_e
    log.info("")

    # ---- verdict ---------------------------------------------------------
    banner(log, "VERDICT")
    a_cancer_last = sum(1 for r in rows_a
                        if r["cancer_rank"] == r["n_compartments"])
    b_top_cancer = sum(1 for r in rows_b if r["top"] == CANCER)
    mean_var = float(np.nanmean([r["variance_explained_by_g2m"] for r in rows_b]))

    log.info(f"Cycle-matched: cancer still last in {a_cancer_last}/{len(rows_a)} scores")
    log.info(f"Residualised : cancer is top in {b_top_cancer}/{len(rows_b)} scores")
    log.info(f"G2M explains on average {100*mean_var:.1f}% of dormancy-score variance")
    log.info("")

    if mean_var > 0.30 and a_cancer_last < len(rows_a):
        verdict = "PROLIFERATION_CONFOUND"
        log.info("The apparent compartment effect is substantially explained by")
        log.info("proliferation. These signatures behave as inverse-proliferation")
        log.info("scores; cancer epithelium ranks last because it is the")
        log.info("proliferative compartment, not because dormancy is stromal.")
        log.info("")
        log.info("This is still a publishable result -- arguably a better one --")
        log.info("but the claim changes. Frame the paper as: dormancy signatures")
        log.info("are not separable from proliferation in bulk or single-cell")
        log.info("scoring, which explains why bulk dormancy scores fail to")
        log.info("predict relapse. Do NOT claim 'dormancy signal is stromal'.")
    elif a_cancer_last == len(rows_a) and mean_var < 0.30:
        verdict = "COMPARTMENT_EFFECT_ROBUST"
        log.info("The compartment effect survives cycle matching and")
        log.info("residualisation. The signatures genuinely score outside cancer")
        log.info("epithelium. This is the strong version of claim C1.")
    else:
        verdict = "PARTIAL"
        log.info("Mixed. Proliferation explains part but not all of the effect.")
        log.info("Report both the raw and deconfounded rankings; the honest claim")
        log.info("is that dormancy signatures conflate two things.")

    report["verdict"] = verdict
    report["g2m_variance_explained"] = mean_var
    log.info("")
    log.info(f"VERDICT: {verdict}")

    out = cfg.paths.results / "wp2c_deconfounding.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {out}")

    pd.DataFrame(rows_b).to_csv(
        cfg.paths.results / "wp2c_residualised_ranking.csv", index=False)

    try:
        from ..viz.deconfound import plot_deconfounding
        fig = plot_deconfounding(df, ct.values, g2m, combined, keep,
                                 cfg.paths.figures / "fig2_deconfounding.svg")
        log.info(f"Written: {fig}")
        report["figure"] = str(fig)
    except Exception as e:  # noqa: BLE001
        log.info(f"[!] Figure skipped: {type(e).__name__}: {e}")

    return report
