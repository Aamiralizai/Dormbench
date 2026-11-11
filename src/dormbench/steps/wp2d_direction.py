"""WP2d — direction-resolved compartment attribution.

WP2c section E showed that the 'down' halves of published dormancy signatures
are proliferation modules (corr with G2M up to +0.73), while the 'up' halves are
nearly orthogonal to cell cycle (-0.02 to -0.19). A signed score computed as
`up - down` therefore constructs an inverse-proliferation score by arithmetic,
regardless of what the up genes contain.

This step redoes the compartment attribution on each direction separately, so
that the dormancy programme is tested without its embedded proliferation module.

It also fixes a defect in WP2b: signatures with only an 'up' direction produce
no combined score and were silently dropped from attribution. That excluded
NR2F1_axis -- the signature most directly relevant to the published finding
that NR2F1 is predominantly expressed in CAFs rather than cancer cells. Every
signature is tested here on its up score.

Outputs
-------
results/wp2d_direction_attribution.csv   per signature x direction x method
figures/fig3_direction_resolved.svg      the paper's central figure
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import banner, get_logger

CANCER = "Cancer Epithelial"
CYCLING = ("Cancer Cycling", "Cycling T-cells", "Cycling")
STROMA = ("CAFs", "PVL", "Endothelial")


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _bimodality(v: np.ndarray) -> float:
    from scipy.stats import kurtosis, skew

    v = v[np.isfinite(v)]
    if len(v) < 50:
        return np.nan
    n = len(v)
    g, k = skew(v), kurtosis(v, fisher=True)
    denom = k + 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    return float((g**2 + 1) / denom) if denom else np.nan


def run(cfg: Config) -> dict:
    log = get_logger("wp2d", cfg.paths.logs)
    cfg.paths.ensure_outputs()

    report: dict = {"step": "wp2d",
                    "generated": datetime.now().isoformat(timespec="seconds")}

    banner(log, "WP2d — DIRECTION-RESOLVED COMPARTMENT ATTRIBUTION")

    path = cfg.paths.results / "wp2b_cell_scores.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `dormbench wp2b` first.")
    df = pd.read_csv(path)
    ct = df["celltype_major"].astype(str).values
    minor = (df["celltype_minor"].astype(str)
             if "celltype_minor" in df else pd.Series(["?"] * len(df)))
    log.info(f"Loaded {len(df):,} cells")

    g2m_col = next((c for c in df.columns
                    if "G2M" in c and c.startswith("AUCell")), None)
    if g2m_col is None:
        g2m_col = next(c for c in df.columns if "G2M" in c)
    g2m = df[g2m_col].values

    # non-cycling mask, as in wp2c
    is_cyc = minor.str.contains("|".join(CYCLING), case=False, na=False).values
    keep = ~(is_cyc | (g2m > np.nanquantile(g2m, 0.75)))
    log.info(f"Non-cycling subset: {int(keep.sum()):,} cells "
             f"({100*keep.mean():.0f}%)")
    log.info("")

    # every directional score, excluding the proliferation controls
    cols = [c for c in df.columns
            if (c.endswith("|up") or c.endswith("|down"))
            and "G2M" not in c and "E2F" not in c]
    sigs = sorted({c.split("|")[1] for c in cols})
    log.info(f"Signatures tested: {sigs}")
    log.info("")

    rows = []
    for col in sorted(cols):
        method, sig, direction = col.split("|")
        v = df[col].values

        means_all = {l: float(np.nanmean(v[ct == l])) for l in np.unique(ct)}
        means_nc = {l: float(np.nanmean(v[keep & (ct == l)])) for l in np.unique(ct)}
        order_nc = sorted(means_nc, key=means_nc.get, reverse=True)

        cancer_nc = v[keep & (ct == CANCER)]
        best_str = max((s for s in STROMA if s in means_nc),
                       key=lambda s: means_nc[s])
        stroma_nc = v[keep & (ct == best_str)]

        rows.append({
            "method": method, "signature": sig, "direction": direction,
            "corr_g2m": float(np.corrcoef(v, g2m)[0, 1]),
            "top_all": max(means_all, key=means_all.get),
            "top_noncycling": order_nc[0],
            "cancer_rank_noncycling": order_nc.index(CANCER) + 1,
            "n_compartments": len(order_nc),
            "cancer_mean_nc": means_nc[CANCER],
            "best_stromal": best_str,
            "best_stromal_mean_nc": means_nc[best_str],
            "auroc_cancer_vs_stroma_nc": _auroc(cancer_nc, stroma_nc),
            "bimodality_in_cancer_nc": _bimodality(cancer_nc),
        })

    res = pd.DataFrame(rows)
    res.to_csv(cfg.paths.results / "wp2d_direction_attribution.csv", index=False)

    # ---- report by direction --------------------------------------------
    for direction in ("up", "down"):
        banner(log, f"{direction.upper()} SCORES  (non-cycling cells only)")
        sub = res[res["direction"] == direction]
        if sub.empty:
            log.info("   none")
            continue
        log.info(f"{'signature':22s} {'method':8s} {'r(G2M)':>7s} "
                 f"{'top compartment':20s} {'rank':>6s} {'AUROC':>7s}")
        for _, r in sub.sort_values(["signature", "method"]).iterrows():
            log.info(f"{r['signature']:22s} {r['method']:8s} "
                     f"{r['corr_g2m']:+7.3f} {r['top_noncycling']:20s} "
                     f"{r['cancer_rank_noncycling']:>3.0f}/{r['n_compartments']:<2.0f} "
                     f"{r['auroc_cancer_vs_stroma_nc']:7.3f}")
        log.info("")

    # ---- the key contrast ------------------------------------------------
    banner(log, "KEY CONTRAST — does the direction change the conclusion?")
    up = res[res["direction"] == "up"]
    down = res[res["direction"] == "down"]
    mean_r_up = float(up["corr_g2m"].mean())
    mean_r_dn = float(down["corr_g2m"].mean()) if len(down) else np.nan
    log.info(f"mean corr with G2M:   up {mean_r_up:+.3f}   down {mean_r_dn:+.3f}")
    log.info("")

    up_cancer_top = int((up["top_noncycling"] == CANCER).sum())
    log.info(f"UP scores with cancer epithelium top: {up_cancer_top}/{len(up)}")
    for _, r in up.iterrows():
        mark = "  <== cancer-dominant" if r["top_noncycling"] == CANCER else ""
        log.info(f"   {r['signature']:22s} {r['method']:8s} "
                 f"top={r['top_noncycling']}{mark}")
    log.info("")

    # per-signature AUROC averaged over methods, up direction only
    banner(log, "PER-SIGNATURE VERDICT (up direction, non-cycling)")
    log.info("AUROC > 0.5 means cancer epithelium scores above the top stromal")
    log.info("compartment. Averaged across AUCell and scanpy.")
    log.info("")
    per_sig = (up.groupby("signature")
               .agg(auroc=("auroc_cancer_vs_stroma_nc", "mean"),
                    corr_g2m=("corr_g2m", "mean"),
                    bimod=("bimodality_in_cancer_nc", "mean"))
               .sort_values("auroc", ascending=False))
    for sig, r in per_sig.iterrows():
        tag = "cancer-dominant" if r["auroc"] > 0.5 else "stroma-dominant"
        log.info(f"   {sig:22s} AUROC {r['auroc']:.3f}  ({tag})   "
                 f"r(G2M) {r['corr_g2m']:+.3f}   bimodality {r['bimod']:.3f}")
    report["per_signature"] = per_sig.reset_index().to_dict("records")
    log.info("")

    # ---- verdict ---------------------------------------------------------
    banner(log, "VERDICT")
    n_cancer_dom = int((per_sig["auroc"] > 0.5).sum())
    log.info(f"Cancer-dominant signatures (up direction): "
             f"{n_cancer_dom}/{len(per_sig)}")
    log.info("")
    log.info("Interpretation:")
    if abs(mean_r_dn) > abs(mean_r_up) + 0.15:
        log.info(f"  - the 'down' sets carry a proliferation module "
                 f"(mean r(G2M) = {mean_r_dn:+.3f}) while the 'up' sets are")
        log.info(f"    comparatively cycle-independent ({mean_r_up:+.3f})")
        log.info("  - scoring as up-minus-down therefore builds an inverse-")
        log.info("    proliferation score by arithmetic, whatever the up genes hold")
    elif abs(mean_r_up) > abs(mean_r_dn) + 0.15:
        log.info(f"  - unexpectedly the 'up' sets are more cycle-linked "
                 f"({mean_r_up:+.3f}) than the 'down' sets ({mean_r_dn:+.3f})")
        log.info("  - check the direction labels in signature_panel.csv before")
        log.info("    interpreting further; this is the reverse of the usual case")
    else:
        log.info(f"  - both directions carry similar cycle association "
                 f"(up {mean_r_up:+.3f}, down {mean_r_dn:+.3f}); the signed score")
        log.info("    is not obviously dominated by one half")
    log.info("")

    if n_cancer_dom == 0:
        verdict = "STROMAL_ROBUST"
        log.info("Even the cycle-clean 'up' programmes score outside cancer")
        log.info("epithelium. The compartment effect is not a proliferation")
        log.info("artefact. This is the strong version of claim C1.")
    elif n_cancer_dom == len(per_sig):
        verdict = "ARTEFACT"
        log.info("Every 'up' programme is cancer-dominant. The apparent stromal")
        log.info("signal in wp2b was entirely an artefact of the signed score.")
        log.info("The paper's claim becomes methodological: signed dormancy")
        log.info("scores are invalid, and the fix is to score directions apart.")
    else:
        verdict = "SIGNATURE_DEPENDENT"
        log.info("Signatures disagree. 'Dormancy signature' is not one thing:")
        log.info("some measure a composition-adjusted programme, others measure")
        log.info("stromal or quiescence biology. Compartment attribution is")
        log.info("therefore a necessary step before any of them is used, which")
        log.info("is precisely the paper's argument.")

    report["verdict"] = verdict
    report["n_cancer_dominant"] = n_cancer_dom
    report["mean_corr_g2m_up"] = mean_r_up
    report["mean_corr_g2m_down"] = mean_r_dn
    log.info("")
    log.info(f"VERDICT: {verdict}")

    out = cfg.paths.results / "wp2d_direction.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {out}")

    try:
        from ..viz.direction import plot_direction_resolved
        fig = plot_direction_resolved(
            res, df, ct, keep, cfg.paths.figures / "fig3_direction_resolved.svg")
        log.info(f"Written: {fig}")
        report["figure"] = str(fig)
    except Exception as e:  # noqa: BLE001
        log.info(f"[!] Figure skipped: {type(e).__name__}: {e}")

    return report
