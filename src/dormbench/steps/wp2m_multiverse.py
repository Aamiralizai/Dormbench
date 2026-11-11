"""WP2m - cell-cycle filtering multiverse.

Review point 9. Every downstream step (wp2c, wp2d, wp2e, wp2f, wp7) drops
cycling-labelled cells AND the top G2M quartile before comparing compartments.
That 25% threshold was chosen once and never varied. It is not a neutral choice:
G2M score is mechanistically entangled with the signatures under test, cancer
cells are removed disproportionately by it, and the E2F negative control's
behaviour is sensitive to exactly this cut. A reviewer can argue the filter
manufactures the negative control.

The defence is not an argument, it is a curve. This step recomputes the headline
compartment statistic under every filtering rule a reviewer might propose:

    none                no filtering at all
    annotation          drop cells labelled as cycling, nothing else
    top-10/20/25/30/40  annotation filter plus that G2M quantile
    residualised        no cells dropped; the score is residualised on G2M
    g2m-matched         cancer and stromal cells matched on G2M decile, so the
                        two compartments have the same proliferation profile
                        by construction rather than by exclusion

The claim that has to survive is TRADITIOM's, and the control that has to stay
near 0.5 under at least the reported setting is E2F. If TRADITIOM's direction
flips across the grid, the manuscript cannot report a single threshold. If E2F
is neutral only at 25%, the negative control is an artefact of the cut and must
be described that way.

Input is results/wp2b_cell_scores.csv.gz, so the atlas is not re-loaded.
Output is results/wp2m_multiverse.csv and results/wp2m_multiverse.json.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import banner, get_logger

CANCER = "Cancer Epithelial"
STROMA = ("CAFs", "PVL", "Endothelial")
CYCLING = ("Cancer Cycling", "Cycling T-cells", "Cycling")
QUANTILES = (0.10, 0.20, 0.25, 0.30, 0.40)
# Reported separately because they carry the manuscript's claims.
HEADLINE = ("TRADITIOM_dormancy", "PreAdapted", "E2F_control", "G2M_control")


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _residualise(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 10:
        return y
    b, a = np.polyfit(x[ok], y[ok], 1)
    return y - (a + b * x)


def _g2m_matched(g2m: np.ndarray, pos: np.ndarray, neg: np.ndarray,
                 rng, n_bins: int = 10):
    """Subsample so cancer and stromal cells share a G2M decile profile.

    Excluding proliferative cells removes them from the analysis. Matching
    keeps them but equalises the two compartments' proliferation distribution,
    which tests the same confound without conditioning on a variable that is
    part of the biology under study.
    """
    edges = np.nanquantile(g2m, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.digitize(g2m, edges[1:-1])
    keep_p, keep_n = [], []
    for k in range(n_bins):
        ip = np.where(pos & (b == k))[0]
        in_ = np.where(neg & (b == k))[0]
        m = min(len(ip), len(in_))
        if m == 0:
            continue
        keep_p.append(rng.choice(ip, m, replace=False))
        keep_n.append(rng.choice(in_, m, replace=False))
    if not keep_p:
        return None, None
    return np.concatenate(keep_p), np.concatenate(keep_n)


def run(cfg: Config, method: str = "AUCell") -> dict:
    log = get_logger("wp2m", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    rng = np.random.default_rng(cfg.seed)
    report: dict = {"step": "wp2m",
                    "generated": datetime.now().isoformat(timespec="seconds"),
                    "params": {"method": method, "quantiles": list(QUANTILES)}}

    banner(log, "WP2m - CELL-CYCLE FILTERING MULTIVERSE")
    log.info("The pipeline's default filter drops cycling cells and the top")
    log.info("G2M quartile. This recomputes the compartment AUROC under every")
    log.info("plausible alternative, so the conclusion is shown not to depend")
    log.info("on the threshold that was chosen.")
    log.info("")

    path = cfg.paths.results / "wp2b_cell_scores.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `dormbench wp2b` first.")
    df = pd.read_csv(path)
    ct = df["celltype_major"].astype(str).values
    minor = (df["celltype_minor"].astype(str)
             if "celltype_minor" in df else pd.Series(["?"] * len(df)))
    log.info(f"Loaded {len(df):,} cells")

    g2m_col = next((c for c in df.columns
                    if "G2M" in c and c.startswith(method)),
                   next((c for c in df.columns if "G2M" in c), None))
    if g2m_col is None:
        raise ValueError("no G2M score column; re-run wp2b with the G2M control")
    g2m = df[g2m_col].values
    log.info(f"Proliferation reference: {g2m_col}")

    cols = [c for c in df.columns
            if c.startswith(f"{method}|") and c.endswith("|up")]
    if not cols:
        raise ValueError(f"no {method}|*|up score columns in {path.name}")
    log.info(f"Signatures: {len(cols)} up-direction scores")

    is_cyc = minor.str.contains("|".join(CYCLING), case=False, na=False).values
    best_stroma = max(STROMA, key=lambda s: int((ct == s).sum()))
    log.info(f"Comparison: {CANCER} vs {best_stroma}")
    log.info("")

    # ---- build the grid of filtering rules -------------------------------
    schemes: list[tuple[str, np.ndarray | None, bool]] = [
        ("none", np.ones(len(df), dtype=bool), False),
        ("annotation", ~is_cyc, False),
    ]
    for q in QUANTILES:
        schemes.append((f"top{int(q*100)}pct",
                        ~(is_cyc | (g2m > np.nanquantile(g2m, 1 - q))), False))
    schemes.append(("residualised", np.ones(len(df), dtype=bool), True))

    rows = []
    for name, keep, resid in schemes:
        n_kept = int(keep.sum())
        frac_cancer_lost = 1 - (keep & (ct == CANCER)).sum() / max(
            (ct == CANCER).sum(), 1)
        frac_stroma_lost = 1 - (keep & (ct == best_stroma)).sum() / max(
            (ct == best_stroma).sum(), 1)
        for col in cols:
            v = _residualise(df[col].values, g2m) if resid else df[col].values
            pos = v[keep & (ct == CANCER)]
            neg = v[keep & (ct == best_stroma)]
            rows.append({
                "scheme": name, "score": col,
                "signature": col.split("|")[1],
                "n_cells": n_kept,
                "frac_cancer_removed": float(frac_cancer_lost),
                "frac_stroma_removed": float(frac_stroma_lost),
                "auroc": _auroc(pos, neg)})

    # ---- G2M-matched scheme ---------------------------------------------
    pos_m = (ct == CANCER)
    neg_m = (ct == best_stroma)
    ip, in_ = _g2m_matched(g2m, pos_m, neg_m, rng)
    if ip is not None:
        log.info(f"G2M-matched subsample: {len(ip)} cancer and {len(in_)} "
                 f"{best_stroma} cells, matched on G2M decile")
        for col in cols:
            v = df[col].values
            rows.append({
                "scheme": "g2m_matched", "score": col,
                "signature": col.split("|")[1],
                "n_cells": int(len(ip) + len(in_)),
                "frac_cancer_removed": float(1 - len(ip) / max(pos_m.sum(), 1)),
                "frac_stroma_removed": float(1 - len(in_) / max(neg_m.sum(), 1)),
                "auroc": _auroc(v[ip], v[in_])})
        log.info("")

    res = pd.DataFrame(rows)
    res.to_csv(cfg.paths.results / "wp2m_multiverse.csv", index=False)

    # ---- report ----------------------------------------------------------
    banner(log, "AUROC (cancer vs stroma) ACROSS FILTERING SCHEMES")
    log.info("0.5 = no compartment preference. Below 0.5 = stroma-dominant.")
    log.info("")
    wide = res.pivot_table(index="signature", columns="scheme", values="auroc")
    order = [s for s in ["none", "annotation", "top10pct", "top20pct",
                         "top25pct", "top30pct", "top40pct", "residualised",
                         "g2m_matched"] if s in wide.columns]
    wide = wide[order]
    head = [s for s in HEADLINE if s in wide.index]
    log.info("Headline signatures:")
    log.info(wide.loc[head].round(3).to_string())
    log.info("")
    others = [s for s in wide.index if s not in head]
    if others:
        log.info("All other signatures:")
        log.info(wide.loc[others].round(3).to_string())
        log.info("")

    cost = res[res["scheme"] != "none"].groupby("scheme")[
        ["frac_cancer_removed", "frac_stroma_removed"]].first()
    log.info("Cells removed by each scheme (the selection effect itself):")
    log.info((100 * cost).round(1).to_string())
    log.info("")

    # ---- stability of each claim ----------------------------------------
    banner(log, "DOES THE CONCLUSION DEPEND ON THE THRESHOLD?")
    stab = []
    for sig in wide.index:
        v = wide.loc[sig].dropna()
        if v.empty:
            continue
        side = np.sign(v - 0.5)
        stab.append({"signature": sig, "auroc_min": float(v.min()),
                     "auroc_max": float(v.max()),
                     "auroc_range": float(v.max() - v.min()),
                     "direction_stable": bool(len(set(side[side != 0])) <= 1),
                     "n_schemes": int(len(v))})
    stab = pd.DataFrame(stab).sort_values("auroc_range", ascending=False)
    log.info(f"{'signature':28s} {'min':>7s} {'max':>7s} {'range':>7s} "
             f"{'direction stable':>17s}")
    for _, r in stab.iterrows():
        log.info(f"{r['signature']:28s} {r['auroc_min']:7.3f} "
                 f"{r['auroc_max']:7.3f} {r['auroc_range']:7.3f} "
                 f"{str(bool(r['direction_stable'])):>17s}")
    log.info("")
    report["stability"] = stab.to_dict("records")
    report["grid"] = wide.round(4).reset_index().to_dict("records")

    # ---- verdict ---------------------------------------------------------
    banner(log, "VERDICT")
    trad = stab[stab["signature"] == "TRADITIOM_dormancy"]
    e2f = wide.loc["E2F_control"] if "E2F_control" in wide.index else None

    trad_stable = bool(trad["direction_stable"].iloc[0]) if len(trad) else False
    if len(trad):
        t = trad.iloc[0]
        log.info(f"TRADITIOM AUROC across schemes: {t['auroc_min']:.3f} to "
                 f"{t['auroc_max']:.3f}, direction stable: {trad_stable}")
    e2f_neutral_everywhere = None
    if e2f is not None:
        near = (e2f - 0.5).abs() < 0.1
        e2f_neutral_everywhere = bool(near.all())
        log.info(f"E2F control within 0.1 of 0.5 in "
                 f"{int(near.sum())}/{len(near)} schemes")
        if not near.all():
            bad = ", ".join(near.index[~near])
            log.info(f"   [!] E2F is NOT neutral under: {bad}")
    log.info("")
    report["traditiom_direction_stable"] = trad_stable
    report["e2f_neutral_everywhere"] = e2f_neutral_everywhere

    if trad_stable and e2f_neutral_everywhere:
        verdict = "FILTER_INDEPENDENT"
        log.info("The compartment result holds under every filtering rule and")
        log.info("the negative control stays neutral throughout. The 25%")
        log.info("threshold is a presentational choice, not a load-bearing one.")
        log.info("Report this grid as supplementary evidence and the objection")
        log.info("is closed.")
    elif trad_stable:
        verdict = "CLAIM_STABLE_CONTROL_THRESHOLD_SENSITIVE"
        log.info("The TRADITIOM conclusion survives every filtering rule, but")
        log.info("the E2F control is neutral only under some of them. The")
        log.info("primary claim is safe; the negative control must be described")
        log.info("as threshold-dependent rather than as a clean null.")
    else:
        verdict = "THRESHOLD_SENSITIVE"
        log.info("[!] The compartment conclusion changes direction across")
        log.info("    filtering schemes. It cannot be reported at a single")
        log.info("    threshold. Either report the full grid as the result, or")
        log.info("    move to the G2M-matched comparison, which equalises")
        log.info("    proliferation without excluding cells.")
    report["verdict"] = verdict
    log.info("")
    log.info(f"VERDICT: {verdict}")

    with open(cfg.paths.results / "wp2m_multiverse.json", "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {cfg.paths.results / 'wp2m_multiverse.csv'}")
    return report
