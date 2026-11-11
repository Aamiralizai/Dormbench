"""WP2i — CCLE cross-check: are these genes expressible in cancer cells at all?

The central claim from WP2d-WP2f is that TRADITIOM's leading-edge genes score in
endothelium because they are endothelial markers in tissue. The obvious
challenge is whether cancer cells can express them at all, and tumour tissue
cannot answer that: every measurement there is a mixture.

Cell lines can. CCLE has no stroma, so every read comes from a cancer cell.
Three questions follow, each answerable from the expression matrix alone:

  1. Baseline. Across breast cancer lines, what is the resting expression of
     MMRN2, NRP1, PODXL, CLIC5, FGD5, LMO2, CD36? If they sit near zero in
     essentially every line, then dormant MCF7 inducing them 8-60x is a
     specific event, not drift within a normal range -- and that is an
     independent platform and cohort supporting WP2g.

  2. Sanity check. TRADITIOM POT samples should resemble CCLE MCF7. A large
     discrepancy means a normalisation problem upstream, and it is better to
     find that here than in review.

  3. Mechanism, independently of WP2h. Across unperturbed breast lines, do the
     leading-edge genes co-vary with a hypoxia score? If they do, the HIF
     reading is supported without relying on the seven marker genes that the
     signature-subset file happens to contain.

Needs only the expression matrix and Model.csv. CRISPR gene effect is not used:
Chronos asks whether knocking a gene out impairs proliferation, and dormancy
genes are by definition not required for proliferation, so a null there would
mean nothing.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config
from ..logging_utils import banner, get_logger

# Well-established ER+/luminal breast lines. Verify against your release's
# metadata before using in a manuscript; DepMap does not ship receptor status.
ER_POSITIVE = {
    "MCF7", "T47D", "ZR751", "ZR7530", "CAMA1", "MDAMB134VI", "MDAMB175VII",
    "MDAMB361", "HCC1428", "BT474", "EFM19", "KPL1", "UACC812", "HCC1500",
    "MDAMB415", "EFM192A", "SUM44PE", "BT483", "HCC1419",
}

HYPOXIA_PROXY = ["VEGFA", "SLC2A1", "CA9", "LDHA", "PGK1", "BNIP3", "ADM",
                 "NDRG1", "P4HA1", "EGLN3", "ALDOA", "ENO1"]


def _strip_gene(name: str) -> str:
    """DepMap columns look like 'TSPAN6 (7105)'."""
    return re.sub(r"\s*\(\d+\)\s*$", "", str(name)).strip()


def _dedupe_columns(df: pd.DataFrame, log) -> pd.DataFrame:
    """Collapse duplicate gene columns.

    Stripping the Entrez suffix from 'TSPAN6 (7105)' can make two columns
    collide when a symbol appears under more than one gene ID. Left in place
    these turn every .loc lookup into a Series and crash downstream, so they
    are collapsed to the row-wise maximum.
    """
    dup = df.columns[df.columns.duplicated()].unique()
    if len(dup) == 0:
        return df
    log.info(f"   collapsing {len(dup)} duplicate gene columns "
             f"(e.g. {list(dup[:4])})")
    return df.T.groupby(level=0).max().T


def _key_expression(df: pd.DataFrame, model: pd.DataFrame, source: str,
                    log) -> pd.DataFrame | None:
    """Return the matrix indexed by DepMap ModelID, or None if not possible.

    DepMap matrices in the wild are keyed three ways: by ModelID in the index,
    by ModelID in an ordinary column (a parquet written after reset_index), or
    by stripped cell-line name. All three are handled here so the fast parquet
    can be used when it exists.
    """
    idx = df.index.astype(str)
    if idx.str.startswith("ACH-").any():
        return df

    for c in df.columns[:10]:
        try:
            if df[c].astype(str).str.startswith("ACH-").mean() > 0.8:
                log.info(f"   index restored from column '{c}'")
                return df.set_index(c)
        except Exception:  # noqa: BLE001
            continue

    # keyed by cell-line name? map back to ModelID via Model.csv
    id_col = next((c for c in ("ModelID", "DepMap_ID") if c in model.columns),
                  model.columns[0])
    name_col = next((c for c in ("StrippedCellLineName", "CellLineName",
                                 "stripped_cell_line_name")
                     if c in model.columns), None)
    if name_col:
        norm = lambda x: str(x).upper().replace("-", "").replace(" ", "")
        lut = {norm(n): i for n, i in zip(model[name_col], model[id_col])}
        hits = idx.map(lambda x: lut.get(norm(x)))
        frac = float(pd.notna(hits).mean())
        if frac > 0.8:
            log.info(f"   index is cell-line names; mapped {frac*100:.0f}% "
                     "to ModelID")
            out = df.copy()
            out.index = hits
            return out[pd.notna(out.index)]

    log.info(f"   [!] {source}: cannot key this matrix by ModelID.")
    log.info(f"       index looks like : {list(df.index[:3])}")
    log.info(f"       columns look like: {list(df.columns[:4])}")
    return None


def _load_expression(depmap_dir: Path, model: pd.DataFrame, log) -> pd.DataFrame:
    """Load CCLE expression indexed by ModelID, parquet preferred for speed."""
    for pq in sorted(depmap_dir.rglob("*expression*.parquet")):
        log.info(f"   reading {pq.name} (parquet)")
        df = _key_expression(pd.read_parquet(pq), model, pq.name, log)
        if df is not None:
            df.columns = [_strip_gene(c) for c in df.columns]
            return _dedupe_columns(df, log)
        log.info("       falling back to CSV")

    csvs = [p for p in depmap_dir.rglob("*.csv") if "expression" in p.name.lower()]
    csvs.sort(key=lambda p: ("protein" not in p.name.lower(), len(p.name)))
    for csv in csvs:
        log.info(f"   reading {csv.name} (csv, slower)")
        df = pd.read_csv(csv, index_col=0, low_memory=False)
        df = _key_expression(df, model, csv.name, log)
        if df is not None:
            df.columns = [_strip_gene(c) for c in df.columns]
            return _dedupe_columns(df, log)

    raise FileNotFoundError(
        f"No usable expression matrix under {depmap_dir}.")


def _load_model(depmap_dir: Path) -> pd.DataFrame:
    for pat in ("Model.csv", "Model*.csv", "sample_info*.csv"):
        hits = sorted(depmap_dir.rglob(pat))
        if hits:
            return pd.read_csv(hits[0], low_memory=False)
    raise FileNotFoundError(f"Model.csv not found under {depmap_dir}")


def _scalar(expr: pd.DataFrame, row, col) -> float:
    """Single value from .loc, tolerant of residual duplicate labels."""
    if row is None:
        return np.nan
    try:
        v = expr.loc[row, col]
    except KeyError:
        return np.nan
    if isinstance(v, (pd.Series, pd.DataFrame)):
        v = np.asarray(v).ravel()
        return float(v[0]) if v.size else np.nan
    return float(v)


def run(cfg: Config, signature: str = "TRADITIOM_dormancy",
        top_n: int = 30) -> dict:
    log = get_logger("wp2i", cfg.paths.logs)
    cfg.paths.ensure_outputs()

    report: dict = {"step": "wp2i",
                    "generated": datetime.now().isoformat(timespec="seconds")}

    banner(log, "WP2i — CCLE CROSS-CHECK (stroma-free cancer cells)")

    depmap_dir = getattr(cfg.paths, "depmap_dir", None)
    if depmap_dir is None or not str(depmap_dir) or not Path(depmap_dir).exists():
        raise FileNotFoundError(
            "depmap_dir is not set or does not exist. This step is optional; "
            "either skip it (dormbench all --skip wp2i) or point depmap_dir at "
            "your DepMap/CCLE download directory in config.yaml, e.g.:\n"
            "  paths:\n    depmap_dir: \"/path/to/depmap\"\n"
            "The directory must contain Model.csv and an expression matrix "
            "(*expression*.csv or *.parquet).")
    depmap_dir = Path(depmap_dir)

    model = _load_model(depmap_dir)
    expr = _load_expression(depmap_dir, model, log)
    log.info(f"   {expr.shape[0]} lines x {expr.shape[1]} genes")

    id_col = next((c for c in ("ModelID", "DepMap_ID") if c in model.columns),
                  model.columns[0])
    name_col = next((c for c in ("StrippedCellLineName", "CellLineName",
                                 "stripped_cell_line_name") if c in model.columns),
                    None)
    lin_col = next((c for c in ("OncotreeLineage", "lineage") if c in model.columns),
                   None)

    breast = model[model[lin_col].astype(str).str.contains("Breast", case=False,
                                                           na=False)] \
        if lin_col else model
    if expr.index.duplicated().any():
        n = int(expr.index.duplicated().sum())
        log.info(f"   collapsing {n} duplicate rows (same ModelID)")
        expr = expr.groupby(level=0).max()
    ids = [i for i in dict.fromkeys(breast[id_col]) if i in expr.index]
    log.info(f"   breast lines with expression: {len(ids)}")
    if len(ids) < 5:
        log.info("")
        log.info("[X] Too few breast lines matched between Model.csv and the")
        log.info("    expression matrix. This is an identifier mismatch, not a")
        log.info("    biological result.")
        log.info(f"    Model {id_col} examples : {list(breast[id_col][:3])}")
        log.info(f"    expression index examples: {list(expr.index[:3])}")
        log.info("    Fix the join before interpreting anything downstream.")
        raise ValueError(
            f"only {len(ids)} breast lines matched; expression index is "
            f"{expr.index.dtype} starting {list(expr.index[:2])}")

    names = (breast.set_index(id_col)[name_col].astype(str).str.upper()
             .str.replace(r"[^A-Z0-9]", "", regex=True)
             if name_col else pd.Series(dtype=str))
    er_ids = [i for i in ids if names.get(i, "") in ER_POSITIVE]
    log.info(f"   of which ER+/luminal (by name list): {len(er_ids)}")
    log.info("   NOTE: receptor status is not in DepMap metadata; this list is")
    log.info("   hand-curated and should be verified before publication.")
    log.info("")

    lead = pd.read_csv(cfg.paths.results / "wp2f_leading_edge.csv")
    le = lead[(lead["signature"] == signature) & (lead["direction"] == "up")]
    genes = [g for g in le["gene"].head(top_n) if g in expr.columns]
    log.info(f"Leading-edge genes present in CCLE: {len(genes)}/{min(top_n, len(le))}")
    log.info("")

    sub = expr.loc[ids, genes]
    sub_er = expr.loc[er_ids, genes] if er_ids else sub

    # ---- 1. baseline -----------------------------------------------------
    banner(log, "1. BASELINE EXPRESSION ACROSS BREAST LINES (log2 TPM+1)")
    mcf7_id = next((i for i in ids if names.get(i, "") == "MCF7"), None)
    log.info(f"{'gene':14s} {'median':>8s} {'q90':>7s} {'max':>7s} "
             f"{'%lines<0.5':>11s} {'MCF7':>7s}")
    rows = []
    for g in genes:
        v = sub[g].dropna()
        m = _scalar(expr, mcf7_id, g)
        frac_low = float((v < 0.5).mean())
        rows.append({"gene": g, "median": float(v.median()),
                     "q90": float(v.quantile(0.9)), "max": float(v.max()),
                     "frac_lines_below_0.5": frac_low, "MCF7": m,
                     "median_ERpos": float(sub_er[g].dropna().median())})
        log.info(f"{g:14s} {v.median():8.2f} {v.quantile(0.9):7.2f} "
                 f"{v.max():7.2f} {frac_low*100:10.0f}% {m:7.2f}")
    base = pd.DataFrame(rows)
    base.to_csv(cfg.paths.results / "wp2i_ccle_baseline.csv", index=False)

    silent = base[base["frac_lines_below_0.5"] > 0.8]
    log.info("")
    log.info(f"Genes essentially silent (<0.5 log2TPM in >80% of breast lines): "
             f"{len(silent)}/{len(base)}")
    if len(silent):
        log.info(f"   {', '.join(silent['gene'].tolist())}")
    report["n_silent"] = int(len(silent))
    report["n_tested"] = int(len(base))
    log.info("")

    # ---- 2. sanity check against TRADITIOM POT ---------------------------
    banner(log, "2. SANITY CHECK — TRADITIOM POT vs CCLE MCF7")
    tpath = cfg.paths.results / "traditiom_expression.csv"
    corr_info = None
    if tpath.exists() and mcf7_id:
        t = pd.read_csv(tpath)
        t = t.set_index(t.columns[0])
        pot = [c for c in t.columns if str(c).startswith("POT")]
        shared = [g for g in t.index if g in expr.columns]
        if pot and len(shared) > 50:
            tv = t.loc[shared, pot].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            cv = expr.loc[mcf7_id, shared]
            r, p = stats.spearmanr(tv.values, cv.values, nan_policy="omit")
            corr_info = {"n_genes": len(shared), "spearman": float(r),
                         "p": float(p)}
            log.info(f"   {len(shared)} shared genes")
            log.info(f"   Spearman rho = {r:.3f}  (p={p:.2e})")
            log.info("")
            if r > 0.6:
                log.info("   Consistent. TRADITIOM baseline matches CCLE MCF7,")
                log.info("   so the induction in WP2g is not a normalisation")
                log.info("   artefact.")
            else:
                log.info("   [!] Weak agreement. Check normalisation before")
                log.info("   relying on absolute levels in WP2g. Note the two")
                log.info("   use different pipelines, so perfect agreement is")
                log.info("   not expected -- but rho < 0.6 is low for the same")
                log.info("   cell line.")
        else:
            log.info("   insufficient overlap for the check")
    else:
        log.info("   skipped (need traditiom_expression.csv and MCF7 in CCLE)")
    report["pot_vs_ccle"] = corr_info
    log.info("")

    # ---- 3. hypoxia co-variation ----------------------------------------
    banner(log, "3. DO THESE GENES TRACK HYPOXIA ACROSS UNPERTURBED LINES?")
    hyp = [g for g in HYPOXIA_PROXY if g in expr.columns]
    log.info(f"   hypoxia proxy genes present: {len(hyp)}/{len(HYPOXIA_PROXY)}")
    if len(hyp) >= 5 and len(genes) >= 5:
        hscore = expr.loc[ids, hyp].mean(axis=1)
        cors = []
        for g in genes:
            r, _ = stats.spearmanr(expr.loc[ids, g], hscore, nan_policy="omit")
            cors.append({"gene": g, "rho_with_hypoxia": float(r)})
        cdf = pd.DataFrame(cors).sort_values("rho_with_hypoxia", ascending=False)
        cdf.to_csv(cfg.paths.results / "wp2i_hypoxia_correlation.csv", index=False)
        med = float(cdf["rho_with_hypoxia"].median())
        log.info(f"   median rho with hypoxia score: {med:+.3f}")
        log.info("")
        for _, r in cdf.head(10).iterrows():
            log.info(f"      {r['gene']:14s} rho {r['rho_with_hypoxia']:+.3f}")
        frac_pos = float((cdf["rho_with_hypoxia"] > 0).mean())
        n_lines = len(ids)
        report["median_rho_hypoxia"] = med
        report["frac_positive_rho"] = frac_pos
        log.info("")
        log.info(f"   {int((cdf['rho_with_hypoxia']>0).sum())}/{len(cdf)} genes "
                 f"positively correlated ({frac_pos*100:.0f}%)")
        # A single rho is weak at n~70 lines; consistency of DIRECTION across
        # genes is the informative quantity, so report both rather than
        # thresholding the median alone.
        if med > 0.35 and frac_pos > 0.8:
            log.info("   Strong, consistent co-variation with hypoxia in")
            log.info("   unperturbed cancer cells. Independent support for the")
            log.info("   WP2h reading.")
        elif med > 0.15 and frac_pos > 0.7:
            log.info("   Modest but directionally consistent association with")
            log.info("   hypoxia. Individual correlations are weak at this")
            log.info("   sample size, but nearly all point the same way, which")
            log.info("   is the informative part. Supports the HIF reading")
            log.info("   without settling it.")
        elif frac_pos < 0.5:
            log.info("   Genes correlate NEGATIVELY with hypoxia on balance,")
            log.info("   arguing against the WP2h reading.")
        else:
            log.info("   No coherent hypoxia association at baseline. Treat")
            log.info("   WP2h as unresolved until the full count matrix is")
            log.info("   available.")
    else:
        log.info("   insufficient genes for the correlation")
        report["median_rho_hypoxia"] = None
    log.info("")

    # ---- verdict ---------------------------------------------------------
    banner(log, "VERDICT")
    frac_silent = len(silent) / max(len(base), 1)
    if frac_silent > 0.5:
        verdict = "SILENT_IN_CANCER_LINES"
        log.info("Most leading-edge genes are essentially unexpressed across")
        log.info("breast cancer lines. Dormant MCF7 switching them on is")
        log.info("therefore a specific event, not variation within a normal")
        log.info("range. This is independent support for WP2g on a different")
        log.info("platform and cohort, and it strengthens the claim that these")
        log.info("genes are not part of the normal breast-cancer repertoire.")
    elif frac_silent < 0.2:
        verdict = "BROADLY_EXPRESSED"
        log.info("These genes are expressed across breast cancer lines at")
        log.info("baseline. Their tissue localisation to endothelium is then")
        log.info("about relative abundance rather than cancer cells being")
        log.info("unable to express them -- a weaker claim, and the text")
        log.info("should say so.")
    else:
        verdict = "MIXED"
        log.info("Some leading-edge genes are silent in cancer lines and some")
        log.info("are broadly expressed. Split them in the figure; they are")
        log.info("making different points.")

    report["verdict"] = verdict
    log.info("")
    log.info(f"VERDICT: {verdict}")

    out = cfg.paths.results / "wp2i_ccle.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {out}")
    log.info(f"Written: {cfg.paths.results / 'wp2i_ccle_baseline.csv'}")

    return report
