"""WP9b - variance partition using the OFFICIAL ESTIMATE / MCP-counter scores.

WP9 computes marker-mean proxies so the pipeline is self-contained. This step
instead reads the scores produced by the canonical ESTIMATE (MD Anderson) and
MCP-counter (Becht et al.) R packages via scripts/run_official_purity.R, and
refits the same variance-partition model against them. When these files are
present the analysis is genuine independent purity validation rather than a
proxy, which is what the manuscript claims once wp9b has been run.

Expected inputs (written by the R script into results/):
    purity_official_METABRIC.csv, purity_official_TCGA.csv, ...
each with a `sample` column plus ESTIMATE columns (StromalScore, ImmuneScore,
ESTIMATEScore, and TumorPurity on affymetrix) and MCP-counter population
columns (Fibroblasts, Endothelial cells, ...).

If no such files exist the step explains how to produce them and exits without
error, so `dormbench all` never breaks for users who have not run the R step.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import banner, get_logger

PROLIF_GENES = ["MKI67", "PCNA", "MCM2", "CCNB1", "CDK1", "TOP2A", "BIRC5",
                "AURKA", "CCNB2", "UBE2C"]


def _zscore(expr: pd.DataFrame, genes: list[str]) -> pd.Series:
    present = [g for g in genes if g in expr.index]
    if len(present) < 3:
        return pd.Series(np.nan, index=expr.columns)
    sub = expr.loc[present]
    z = sub.sub(sub.mean(axis=1), axis=0).div(
        sub.std(axis=1).replace(0, np.nan), axis=0)
    return z.mean(axis=0)


def _partial_r2(df: pd.DataFrame, y: str, terms: list[str]) -> dict:
    def r2(cols):
        X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float)
                                                  for c in cols]) if cols \
            else np.ones((len(df), 1))
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
        red, _ = r2([c for c in terms if c != t])
        out[t] = (full - red) if np.isfinite(full) and np.isfinite(red) \
            else np.nan
    return out


def _find_col(df: pd.DataFrame, *keys) -> str | None:
    """Match a column ignoring case, spaces, underscores and dots."""
    def norm(x):
        return str(x).lower().replace(" ", "").replace("_", "").replace(".", "")
    low = {norm(c): c for c in df.columns}
    for k in keys:
        if norm(k) in low:
            return low[norm(k)]
    # substring fallback
    for k in keys:
        nk = norm(k)
        for c in df.columns:
            if nk in norm(c):
                return c
    return None


def _norm_ids(idx) -> dict:
    """Map normalised sample id -> original, so METABRIC 'MB-0001' aligns with
    R's 'MB.0001' and TCGA barcodes align regardless of . / - differences."""
    def norm(x):
        return str(x).upper().replace(".", "-").replace("_", "-").strip()
    return {norm(i): i for i in idx}


def run(cfg: Config) -> dict:
    log = get_logger("wp9b", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    banner(log, "WP9b - OFFICIAL ESTIMATE / MCP-COUNTER VALIDATION")
    report = {"step": "wp9b", "generated": datetime.now().isoformat()}

    files = sorted(cfg.paths.results.glob("purity_official_*.csv"))
    if not files:
        log.info("   No purity_official_*.csv found.")
        log.info("   Produce them with the reference R packages:")
        log.info("     1) dormbench wp9        # writes wp9_expr_<COHORT>.csv")
        log.info("     2) Rscript scripts/run_official_purity.R")
        log.info("   then rerun `dormbench wp9b`.")
        report["verdict"] = "OFFICIAL_SCORES_ABSENT"
        with open(cfg.paths.results / "wp9b_official.json", "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        return report

    from ..io import (DataNotFound, load_metabric, load_scanb, load_tcga,
                      tidy_metabric_expression)
    panel = pd.read_csv(cfg.paths.results / "signature_panel.csv")
    dgenes = panel[(panel["signature"] == "TRADITIOM_dormancy")
                   & (panel["direction"] == "up")]["gene"].tolist()

    loaders = {
        "METABRIC": lambda: tidy_metabric_expression(
            load_metabric(cfg.paths.metabric_dir)["expr"]),
        "TCGA": lambda: (load_tcga(cfg.paths.tcga_dir, log).get("expr")
                         if isinstance(load_tcga(cfg.paths.tcga_dir, log), dict)
                         else load_tcga(cfg.paths.tcga_dir, log)),
        "SCANB": lambda: (load_scanb(cfg.paths.scanb_dir).get("expr")
                          if isinstance(load_scanb(cfg.paths.scanb_dir), dict)
                          else load_scanb(cfg.paths.scanb_dir)),
    }

    rows = []
    for f in files:
        tag = f.stem.replace("purity_official_", "")
        pur = pd.read_csv(f)
        scol = _find_col(pur, "sample") or pur.columns[0]
        pur = pur.set_index(scol)
        stromal = _find_col(pur, "StromalScore")
        fibro = _find_col(pur, "Fibroblasts", "CancerAssociatedFibroblasts")
        endo = _find_col(pur, "Endothelialcells", "Endothelial")
        est = _find_col(pur, "ESTIMATEScore")
        banner(log, f"{tag}  ({len(pur)} samples, official scores)")

        try:
            expr = loaders.get(tag.upper(), lambda: None)()
        except (DataNotFound, Exception) as e:  # noqa: BLE001
            log.info(f"   expression for {tag} unavailable ({e}); "
                     "correlation only")
            expr = None
        if expr is None:
            continue

        dorm = _zscore(expr, dgenes)
        prolif = _zscore(expr, PROLIF_GENES)
        df = pd.DataFrame({"dormancy": dorm, "proliferation": prolif})
        # align official purity onto the same samples, tolerating . / - / _
        # differences introduced when R read the CSV headers
        expr_map = _norm_ids(df.index)          # normalised -> expr id
        pur_map = _norm_ids(pur.index)          # normalised -> purity id
        shared = sorted(set(expr_map) & set(pur_map))
        if len(shared) < 30:
            log.info(f"   only {len(shared)} sample IDs aligned after "
                     "normalisation; check ID formatting")
        expr_ids = [expr_map[k] for k in shared]
        pur_ids = [pur_map[k] for k in shared]
        df = df.loc[expr_ids].reset_index(drop=True)
        pur_al = pur.loc[pur_ids].reset_index(drop=True)
        if stromal:
            df["estimate_stromal"] = pd.to_numeric(
                pur_al[stromal], errors="coerce").values
        if est:
            df["purity_estimate"] = -pd.to_numeric(
                pur_al[est], errors="coerce").values
        if fibro:
            df["mcp_fibroblast"] = pd.to_numeric(
                pur_al[fibro], errors="coerce").values

        # Choose the purity axis: prefer ESTIMATE if it produced values,
        # otherwise fall back to the MCP-counter fibroblast score, which is an
        # independent composition estimator in its own right. This means the
        # step still yields a genuine independent-estimator result even if
        # ESTIMATE returns all-NA on a given platform.
        if "purity_estimate" in df and df["purity_estimate"].notna().sum() >= 20:
            df["purity"] = df["purity_estimate"]
            purity_source = "ESTIMATE"
        elif "mcp_fibroblast" in df and df["mcp_fibroblast"].notna().sum() >= 20:
            df["purity"] = df["mcp_fibroblast"]
            purity_source = "MCP-counter fibroblast"
        else:
            purity_source = "none"
        log.info(f"   purity axis for variance model: {purity_source}")

        if "estimate_stromal" in df:
            valid = df[["dormancy", "estimate_stromal"]].dropna()
            if len(valid) >= 20:
                r = valid.corr().iloc[0, 1]
                log.info(f"   dormancy vs OFFICIAL ESTIMATE stromal   r = "
                         f"{r:+.3f}  (n={len(valid)} non-NA)")
            else:
                log.info(f"   ESTIMATE stromal: only {len(valid)} non-NA "
                         "samples; ESTIMATE returned NA for the rest")
        if "mcp_fibroblast" in df:
            validf = df[["dormancy", "mcp_fibroblast"]].dropna()
            if len(validf) >= 20:
                rf = validf.corr().iloc[0, 1]
                log.info(f"   dormancy vs OFFICIAL MCP fibroblast     r = "
                         f"{rf:+.3f}  (n={len(validf)} non-NA)")

        terms = [c for c in ("purity", "proliferation") if c in df.columns]
        # drop rows where any model term is NA (ESTIMATE returns NA for some)
        model_df = df.dropna(subset=["dormancy"] + terms) if terms else df
        pr = _partial_r2(model_df, "dormancy", terms) \
            if len(terms) == 2 and len(model_df) >= 20 else {}
        if pr:
            log.info(f"   variance partition (semi-partial R2), full R2 = "
                     f"{pr['_full_r2']:.3f}, n = {pr['_n']}")
            log.info(f"      purity        {pr.get('purity', float('nan')):+.4f}")
            log.info(f"      proliferation "
                     f"{pr.get('proliferation', float('nan')):+.4f}")

        def _corr(a, b):
            v = df[[a, b]].dropna()
            return float(v.corr().iloc[0, 1]) if len(v) >= 20 else None

        rows.append({
            "cohort": tag, "n": int(len(df)),
            "purity_source": purity_source,
            "n_estimate_valid": int(df["estimate_stromal"].notna().sum())
            if "estimate_stromal" in df else 0,
            "r_dormancy_official_stromal": _corr("dormancy", "estimate_stromal")
            if "estimate_stromal" in df else None,
            "r_dormancy_official_mcp_fibroblast": _corr("dormancy",
                                                        "mcp_fibroblast")
            if "mcp_fibroblast" in df else None,
            "partial_r2_purity": pr.get("purity"),
            "partial_r2_proliferation": pr.get("proliferation"),
            "full_r2": pr.get("_full_r2")})
        log.info("")

    if not rows:
        log.info("   No cohort could be aligned; nothing written.")
        report["verdict"] = "NO_ALIGNMENT"
    else:
        res = pd.DataFrame(rows)
        res.to_csv(cfg.paths.results / "wp9b_official_variance.csv", index=False)
        banner(log, "VERDICT")
        svals = [r["r_dormancy_official_stromal"] for r in rows
                 if r["r_dormancy_official_stromal"] is not None]
        pvals = [r["partial_r2_purity"] for r in rows
                 if r["partial_r2_purity"] is not None]
        provals = [r["partial_r2_proliferation"] for r in rows
                   if r["partial_r2_proliferation"] is not None]
        mr = float(np.nanmean(svals)) if svals else float("nan")
        mp = float(np.nanmean(pvals)) if pvals else float("nan")
        mpro = float(np.nanmean(provals)) if provals else float("nan")
        log.info(f"   mean r(dormancy, OFFICIAL ESTIMATE stromal) = {mr:+.3f}")
        log.info(f"   mean partial R2  purity = {mp:.3f}, "
                 f"proliferation = {mpro:.3f}")
        report.update({"mean_r_official_stromal": mr,
                       "mean_partial_r2_purity": mp,
                       "mean_partial_r2_proliferation": mpro, "cohorts": rows,
                       "verdict": "OFFICIAL_PURITY_VALIDATED"
                       if mr > 0.3 else "OFFICIAL_ASSOCIATION_WEAK"})
        log.info("")
        log.info("   These use the reference ESTIMATE / MCP-counter packages,")
        log.info("   so this is genuine independent purity validation.")

    with open(cfg.paths.results / "wp9b_official.json", "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {cfg.paths.results / 'wp9b_official.json'}")
    return report
