"""WP9 - independent purity validation and variance partition.

The central claim is that dormancy scores in bulk tissue reflect
compartment-associated expression rather than a cancer-cell state. Elsewhere
this is shown by correlating the dormancy score with a stromal-marker score,
but both are computed from the same expression matrix, so the correlation
could be circular. This step tests the claim against composition estimators
that were not defined by this study - ESTIMATE and MCP-counter - and asks how
much dormancy-score variance each independent covariate explains.

For each cohort it fits

    dormancy_score ~ estimate_purity + proliferation + subtype(+ grade/stage)

and reports the partial (semi-partial) R2 of each term: the drop in model R2
when that term alone is removed. If independent purity explains more variance
than proliferation or subtype, "composition" is the right word; if not, the
honest statement is that the score tracks a stromal-marker axis, which is what
the manuscript now says.

The estimators live in dormbench.purity and run on the bulk matrices already
loaded by WP3, so no new data are required.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import banner, get_logger
from ..purity import all_purity_scores

PROLIF_GENES = ["MKI67", "PCNA", "MCM2", "CCNB1", "CDK1", "TOP2A", "BIRC5",
                "AURKA", "CCNB2", "UBE2C"]
DORMANCY = "TRADITIOM_dormancy|up"


def _zscore(expr: pd.DataFrame, genes: list[str]) -> pd.Series:
    present = [g for g in genes if g in expr.index]
    if len(present) < 3:
        return pd.Series(np.nan, index=expr.columns)
    sub = expr.loc[present]
    z = sub.sub(sub.mean(axis=1), axis=0).div(
        sub.std(axis=1).replace(0, np.nan), axis=0)
    return z.mean(axis=0)


def _partial_r2(df: pd.DataFrame, y: str, terms: list[str]) -> dict:
    """Semi-partial R2 for each term: full model R2 minus R2 without that term.

    OLS via numpy; categorical terms are one-hot expanded by the caller.
    """
    import numpy as np

    def r2(cols):
        X = df[cols].to_numpy(float)
        X = np.column_stack([np.ones(len(df)), X])
        yy = df[y].to_numpy(float)
        ok = np.isfinite(yy) & np.isfinite(X).all(axis=1)
        if ok.sum() < len(cols) + 5:
            return np.nan, int(ok.sum())
        b, *_ = np.linalg.lstsq(X[ok], yy[ok], rcond=None)
        pred = X[ok] @ b
        ss_res = float(((yy[ok] - pred) ** 2).sum())
        ss_tot = float(((yy[ok] - yy[ok].mean()) ** 2).sum())
        return (1 - ss_res / ss_tot if ss_tot > 0 else np.nan), int(ok.sum())

    full, n = r2(terms)
    out = {"_full_r2": full, "_n": n}
    for t in terms:
        reduced = [c for c in terms if c != t]
        red_r2, _ = r2(reduced) if reduced else (0.0, n)
        out[t] = (full - red_r2) if np.isfinite(full) and np.isfinite(red_r2) \
            else np.nan
    return out


def _cohort(name: str, expr: pd.DataFrame, dormancy_genes: list[str],
            subtype: pd.Series | None, log) -> dict:
    banner(log, f"{name}  ({expr.shape[1]} samples)")
    pur = all_purity_scores(expr)
    dorm = _zscore(expr, dormancy_genes)
    prolif = _zscore(expr, PROLIF_GENES)

    df = pd.DataFrame({"dormancy": dorm, "purity": pur["estimate_purity_proxy"],
                       "estimate_stromal": pur["estimate_stromal"],
                       "mcp_fibroblast": pur["mcp_fibroblast"],
                       "mcp_endothelial": pur["mcp_endothelial"],
                       "proliferation": prolif})
    # how independent is the ESTIMATE purity from the internal stromal axis?
    r_int = df[["dormancy", "estimate_stromal"]].corr().iloc[0, 1]
    r_mcp = df[["dormancy", "mcp_fibroblast"]].corr().iloc[0, 1]
    log.info(f"   dormancy vs ESTIMATE stromal   r = {r_int:+.3f}")
    log.info(f"   dormancy vs MCP fibroblast     r = {r_mcp:+.3f}")
    log.info(f"   dormancy vs MCP endothelial    r = "
             f"{df[['dormancy','mcp_endothelial']].corr().iloc[0,1]:+.3f}")

    terms = ["purity", "proliferation"]
    if subtype is not None:
        sub = subtype.reindex(df.index).astype(str)
        # one-hot, drop first level as reference
        dummies = pd.get_dummies(sub, prefix="sub", drop_first=True)
        df = df.join(dummies)
        terms += list(dummies.columns)
    pr2 = _partial_r2(df, "dormancy", terms)

    log.info("")
    log.info(f"   variance partition (semi-partial R2), full R2 = "
             f"{pr2['_full_r2']:.3f}, n = {pr2['_n']}")
    for t in terms:
        lab = "subtype" if t.startswith("sub_") else t
        log.info(f"      {lab:16s} {pr2[t]:+.4f}")
    # collapse subtype dummies into one number
    sub_r2 = sum(v for k, v in pr2.items()
                 if k.startswith("sub_") and np.isfinite(v))
    result = {"cohort": name, "n": pr2["_n"], "full_r2": pr2["_full_r2"],
              "partial_r2_purity": pr2.get("purity"),
              "partial_r2_proliferation": pr2.get("proliferation"),
              "partial_r2_subtype": sub_r2 if subtype is not None else None,
              "r_dormancy_estimate_stromal": float(r_int),
              "r_dormancy_mcp_fibroblast": float(r_mcp)}
    log.info("")
    return result, df[["dormancy", "purity", "proliferation",
                       "estimate_stromal", "mcp_fibroblast",
                       "mcp_endothelial"]].assign(cohort=name)


def _tumour_only(expr: pd.DataFrame, log) -> pd.DataFrame:
    """Drop TCGA normal-adjacent samples and deduplicate patients.

    TCGA barcodes encode sample type in the fourth field: -01 (primary tumour),
    -06 (metastatic) are tumour; -11 (solid tissue normal) is not. Reviewer
    item: WP9 must use the same tumour-only cohort as the main analysis, or the
    purity relationship is inflated by normal tissue.
    """
    cols = list(expr.columns)
    # sample-type code is characters 13-15 of a full TCGA barcode
    def is_tumour(bc: str) -> bool:
        parts = str(bc).split("-")
        if len(parts) >= 4 and len(parts[3]) >= 2:
            return parts[3][:2] in {"01", "03", "06", "09"}
        return True  # keep if we cannot parse (non-TCGA cohorts)

    def patient(bc: str) -> str:
        return "-".join(str(bc).split("-")[:3])

    # select by position to survive duplicate column labels
    keep_pos, seen = [], set()
    for i, c in enumerate(cols):
        if not is_tumour(c):
            continue
        pid = patient(c)
        if pid in seen:
            continue
        seen.add(pid); keep_pos.append(i)
    out = expr.iloc[:, keep_pos]
    dropped = sum(1 for c in cols if not is_tumour(c))
    dedup = len(cols) - dropped - len(keep_pos)
    if dropped or dedup:
        log.info(f"   tumour-only filter: dropped {dropped} normal-adjacent, "
                 f"{dedup} duplicate-patient; {out.shape[1]} tumours retained")
    return out


def run(cfg: Config) -> dict:
    """Independent purity validation across the bulk cohorts."""
    from ..io import (DataNotFound, load_metabric, load_scanb, load_tcga,
                      tidy_metabric_expression)

    log = get_logger("wp9", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    banner(log, "WP9 - INDEPENDENT PURITY VALIDATION")
    report = {"step": "wp9", "generated": datetime.now().isoformat()}

    panel = pd.read_csv(cfg.paths.results / "signature_panel.csv")
    dgenes = panel[(panel["signature"] == "TRADITIOM_dormancy")
                   & (panel["direction"] == "up")]["gene"].tolist()
    log.info(f"   dormancy signature: {len(dgenes)} genes")
    log.info("   estimators: ESTIMATE (stromal+immune), MCP-counter "
             "(fibroblast, endothelial)")
    log.info("   these are independent of the internal stromal-marker score")
    log.info("")

    rows, frames = [], []
    # optionally export each cohort's matrix (genes x samples, symbols as the
    # index) for the official ESTIMATE / MCP-counter R script. Written to
    # results/ as wp9_expr_<COHORT>.csv so scripts/run_official_purity.R can
    # read them without any path-hunting.
    export_dir = cfg.paths.results

    def _as_expr(obj):
        """Coerce a loader return (DataFrame or dict) to the expression frame."""
        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, dict):
            for k in ("expr", "expression", "matrix", "data", "X"):
                if k in obj and isinstance(obj[k], pd.DataFrame):
                    return obj[k]
            # else the first DataFrame value
            for v in obj.values():
                if isinstance(v, pd.DataFrame):
                    return v
        raise TypeError(f"cannot find expression frame in {type(obj)}")

    def _export(name: str, expr: pd.DataFrame):
        if not getattr(cfg, "_wp9_export", True):
            return
        if not isinstance(expr, pd.DataFrame):
            log.info(f"   [!] {name} export skipped: not a DataFrame")
            return
        p = export_dir / f"wp9_expr_{name}.csv"
        expr.to_csv(p)
        log.info(f"   exported matrix for official purity tools: {p.name} "
                 f"({expr.shape[0]} genes x {expr.shape[1]} samples)")

    # METABRIC
    try:
        mb = load_metabric(cfg.paths.metabric_dir)
        expr = tidy_metabric_expression(mb["expr"])
        sub = None
        if "sample" in mb:
            s = mb["sample"]
            col = next((c for c in s.columns if "PAM50" in c.upper()
                        or "SUBTYPE" in c.upper()), None)
            idc = next((c for c in s.columns if c.upper() in
                        ("SAMPLE_ID", "PATIENT_ID")), None)
            if col and idc:
                sub = s.set_index(idc)[col]
        _export("METABRIC", expr)
        r, f = _cohort("METABRIC", expr, dgenes, sub, log)
        rows.append(r); frames.append(f)
    except (DataNotFound, Exception) as e:  # noqa: BLE001
        log.info(f"   METABRIC skipped: {type(e).__name__}: {e}")

    # SCAN-B
    try:
        sb = load_scanb(cfg.paths.scanb_dir)
        expr = _as_expr(sb)
        _export("SCANB", expr)
        r, f = _cohort("SCAN-B", expr, dgenes, None, log)
        rows.append(r); frames.append(f)
    except (DataNotFound, Exception) as e:  # noqa: BLE001
        log.info(f"   SCAN-B skipped: {type(e).__name__}: {e}")

    # TCGA
    try:
        tc = load_tcga(cfg.paths.tcga_dir, log)
        expr = tc["expr"] if "expr" in tc else tc
        expr = _tumour_only(expr, log)
        report["tcga_n_tumour_only"] = int(expr.shape[1])
        _export("TCGA", expr)
        r, f = _cohort("TCGA-BRCA", expr, dgenes, None, log)
        rows.append(r); frames.append(f)
    except (DataNotFound, Exception) as e:  # noqa: BLE001
        log.info(f"   TCGA skipped: {type(e).__name__}: {e}")

    if not rows:
        raise RuntimeError("no bulk cohort could be loaded for WP9")

    res = pd.DataFrame(rows)
    res.to_csv(cfg.paths.results / "wp9_variance_partition.csv", index=False)
    if frames:
        pd.concat(frames).to_csv(
            cfg.paths.results / "wp9_purity_scores.csv.gz", index=True)

    banner(log, "VERDICT")
    # is dormancy still associated with INDEPENDENT stromal content?
    mean_r = float(np.nanmean([r["r_dormancy_estimate_stromal"] for r in rows]))
    mean_pur = float(np.nanmean([r["partial_r2_purity"] for r in rows
                                 if r["partial_r2_purity"] is not None]))
    mean_pro = float(np.nanmean([r["partial_r2_proliferation"] for r in rows
                                 if r["partial_r2_proliferation"] is not None]))
    log.info(f"   mean r(dormancy, independent ESTIMATE stromal) = {mean_r:+.3f}")
    log.info(f"   mean partial R2  purity = {mean_pur:.3f},  "
             f"proliferation = {mean_pro:.3f}")
    report["mean_r_independent_stromal"] = mean_r
    report["mean_partial_r2_purity"] = mean_pur
    report["mean_partial_r2_proliferation"] = mean_pro
    report["cohorts"] = rows
    if mean_r > 0.3 and mean_pur >= mean_pro:
        verdict = "INDEPENDENT_PURITY_CONFIRMS_COMPOSITION"
        log.info("")
        log.info("   The dormancy score tracks an INDEPENDENT purity estimate,")
        log.info("   and purity explains at least as much variance as")
        log.info("   proliferation. Composition is a defensible description.")
    elif mean_r > 0.3:
        verdict = "INDEPENDENT_STROMAL_ASSOCIATION_CONFIRMED"
        log.info("")
        log.info("   The dormancy score is associated with an independent")
        log.info("   stromal/purity estimate, but proliferation explains as")
        log.info("   much or more variance. Report a stromal-marker/purity")
        log.info("   association rather than composition dominance.")
    else:
        verdict = "INDEPENDENT_ASSOCIATION_WEAK"
        log.info("")
        log.info("   [!] The association with an independent purity estimate is")
        log.info("   weak; the internal stromal-marker correlation may be")
        log.info("   partly circular. State this limitation explicitly.")
    report["verdict"] = verdict
    log.info("")

    with open(cfg.paths.results / "wp9_purity.json", "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {cfg.paths.results / 'wp9_variance_partition.csv'}")
    return report
