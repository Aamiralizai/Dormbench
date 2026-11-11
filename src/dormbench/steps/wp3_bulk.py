"""WP3 — does the compartment artefact appear in bulk clinical data?

WP2b-WP2f established, at single-cell resolution, that a gene-set score in
tumour tissue is dominated by whichever members are cell-type markers. If that
is right it makes a sharp, falsifiable prediction about bulk data, which is
where these scores are actually used clinically:

    a dormancy score computed on a bulk tumour should track that tumour's
    cell-type composition rather than any property of its cancer cells.

Signatures with high cell-type-marker content (TRADITIOM, EMT) should therefore
correlate strongly with stromal and endothelial abundance. E2F targets -- the
WP2e negative control, AUROC 0.498 -- should not, because cell-cycle genes are
expressed by any dividing cell and carry no cell-type identity.

Compartment abundance is estimated from the Wu atlas itself rather than an
external purity caller: marker genes are derived per compartment directly from
the single-cell reference used throughout, so the bulk analysis rests on the
same annotation as everything upstream and introduces no new dependency.

ER status enters as a stratifier, not as the contrast. ER- tumours are less
pure and more immune-infiltrated, so a composition-driven score will differ
between subtypes for reasons that have nothing to do with dormancy. Showing
that is a concrete illustration of how the artefact misleads in practice.

Cohorts: METABRIC (microarray) as primary, SCAN-B (RNA-seq) as cross-platform
replication. A result that holds on both is not a platform artefact.
"""

from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config
from ..io import (DataNotFound, find_atlas_h5ad, load_metabric, load_scanb,
                  load_tcga, tidy_metabric_expression)
from ..logging_utils import banner, get_logger

CANCER = "Cancer Epithelial"
STROMAL = ["CAFs", "PVL", "Endothelial"]
IMMUNE = ["T-cells", "Myeloid", "B-cells", "Plasmablasts"]


def _compartment_markers(cfg: Config, log, n_markers: int = 50) -> dict[str, list]:
    """Top marker genes per compartment, derived from the atlas and cached.

    Specificity score = mean expression in the compartment minus the highest
    mean in any other compartment. That selects genes that are high in one
    compartment AND low elsewhere, which is what a composition estimate needs;
    ranking on within-compartment mean alone would return housekeeping genes.
    """
    cache = cfg.paths.results / "wp3_compartment_markers.json"
    if cache.exists():
        log.info(f"   using cached {cache.name}")
        with open(cache, encoding="utf-8") as fh:
            return json.load(fh)

    means_path = cfg.paths.results / "wp2f_gene_compartment_means.csv.gz"
    if means_path.exists():
        log.info(f"   deriving markers from {means_path.name}")
        means = pd.read_csv(means_path, index_col=0)
    else:
        log.info("   computing compartment means from the atlas (first run)")
        import anndata as ad
        import scanpy as sc

        adata = ad.read_h5ad(find_atlas_h5ad(cfg.paths.atlas_dir))
        if adata.raw is not None:
            adata = ad.AnnData(X=adata.raw.X.copy(), obs=adata.obs.copy(),
                               var=adata.raw.var.copy()
                               if adata.raw.var is not None else adata.var.copy())
        if "feature_name" in adata.var.columns:
            adata.var_names = pd.Index(adata.var["feature_name"].astype(str))
            adata.var_names_make_unique()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        ct = adata.obs["celltype_major"].astype(str).values
        means = pd.DataFrame(
            {c: np.asarray(adata.X[ct == c].mean(axis=0)).ravel()
             for c in np.unique(ct)}, index=adata.var_names)

    markers = {}
    for comp in means.columns:
        other = means.drop(columns=[comp]).max(axis=1)
        spec = (means[comp] - other).sort_values(ascending=False)
        markers[comp] = spec.head(n_markers).index.astype(str).tolist()
        log.info(f"      {comp:20s} e.g. {markers[comp][:5]}")

    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(markers, fh, indent=2)
    return markers


def _zscore_score(expr: pd.DataFrame, genes: list[str]) -> pd.Series:
    """Mean z-score across a gene set. Genes x samples in, per-sample out."""
    present = [g for g in genes if g in expr.index]
    if len(present) < 3:
        return pd.Series(np.nan, index=expr.columns)
    sub = expr.loc[present]
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1) + 1e-9, axis=0)
    return z.mean(axis=0)


def _analyse_cohort(name: str, expr: pd.DataFrame, panel: pd.DataFrame,
                    markers: dict, er: pd.Series | None, log,
                    n_boot: int = 200, rng=None, n_null: int = 100) -> dict:
    rng = rng if rng is not None else np.random.default_rng(0)
    # float32 halves peak memory and costs nothing at correlation precision
    if expr.dtypes.iloc[0] != np.float32:
        expr = expr.astype(np.float32)
        gc.collect()
    banner(log, f"{name}  —  {expr.shape[0]} genes x {expr.shape[1]} samples")

    comp_scores = pd.DataFrame(
        {c: _zscore_score(expr, g) for c, g in markers.items()})
    usable = comp_scores.columns[comp_scores.notna().any()].tolist()
    log.info(f"   compartment scores computed for: {usable}")

    sig_scores = {}
    for (sig, direction), grp in panel.groupby(["signature", "direction"]):
        genes = grp["gene"].astype(str).tolist()
        s = _zscore_score(expr, genes)
        if s.notna().any():
            sig_scores[f"{sig}|{direction}"] = s
    sig_df = pd.DataFrame(sig_scores)
    log.info(f"   signature scores computed: {sig_df.shape[1]}")
    log.info("")

    # ---- the core test --------------------------------------------------
    stroma = [c for c in STROMAL if c in comp_scores.columns]
    stroma_score = comp_scores[stroma].mean(axis=1) if stroma else None
    cancer_score = comp_scores[CANCER] if CANCER in comp_scores.columns else None

    rows = []
    log.info(f"{'signature':30s} {'r(stroma)':>10s} {'r(cancer)':>10s} "
             f"{'r(endo)':>9s} {'r(CAF)':>8s}")
    for col in sorted(sig_df.columns):
        v = sig_df[col]

        def r(x):
            if x is None:
                return np.nan
            ok = v.notna() & x.notna()
            if ok.sum() < 20:
                return np.nan
            return float(stats.spearmanr(v[ok], x[ok])[0])

        row = {
            "cohort": name, "score": col,
            "r_stroma": r(stroma_score), "r_cancer": r(cancer_score),
            "r_endothelial": r(comp_scores.get("Endothelial")),
            "r_CAFs": r(comp_scores.get("CAFs")),
            "r_immune": r(comp_scores[[c for c in IMMUNE
                                       if c in comp_scores.columns]].mean(axis=1))
            if any(c in comp_scores.columns for c in IMMUNE) else np.nan,
        }
        rows.append(row)
        log.info(f"{col:30s} {row['r_stroma']:+10.3f} {row['r_cancer']:+10.3f} "
                 f"{row['r_endothelial']:+9.3f} {row['r_CAFs']:+8.3f}")
    log.info("")

    res = pd.DataFrame(rows)

    # ---- ER stratification ----------------------------------------------
    er_rows = []
    if er is not None:
        shared = [s for s in sig_df.index if s in er.index]
        if len(shared) > 50:
            e = er.loc[shared]
            pos, neg = e.eq("Positive"), e.eq("Negative")
            log.info(f"   ER+ n={int(pos.sum())}   ER- n={int(neg.sum())}")
            log.info("")
            log.info(f"{'signature':30s} {'ER+ mean':>9s} {'ER- mean':>9s} "
                     f"{'diff':>7s} {'diff|purity-adj':>16s}")
            purity = (cancer_score.loc[shared]
                      if cancer_score is not None else None)
            for col in sorted(sig_df.columns):
                v = sig_df[col].loc[shared]
                a, b = float(v[pos].mean()), float(v[neg].mean())
                diff = a - b
                adj = np.nan
                if purity is not None:
                    ok = v.notna() & purity.notna()
                    if ok.sum() > 50:
                        # residualise on cancer-cell content, then recompute
                        beta, alpha = np.polyfit(purity[ok], v[ok], 1)
                        resid = v - (alpha + beta * purity)
                        adj = float(resid[pos].mean() - resid[neg].mean())
                er_rows.append({"cohort": name, "score": col, "er_pos_mean": a,
                                "er_neg_mean": b, "diff": diff,
                                "diff_purity_adjusted": adj})
                log.info(f"{col:30s} {a:+9.3f} {b:+9.3f} {diff:+7.3f} "
                         f"{adj:+16.3f}")
            log.info("")
        else:
            log.info("   ER status could not be matched to expression samples")
    # ---- sensitivity -----------------------------------------------------
    # Bulk is where resampling is properly powered: hundreds to thousands of
    # patients, against 26 donors in the atlas. Two questions, same as WP4 --
    # does one gene carry the correlation, and how precise is it across
    # patients?
    sens_rows = []
    if stroma_score is not None and n_boot > 0:
        banner(log, f"{name} — SENSITIVITY")
        log.info(f"{'signature':30s} {'r':>7s} {'LOO range':>16s} "
                 f"{'patient 95% CI':>18s}")
        for (sig, direction), grp in panel.groupby(["signature", "direction"]):
            genes = [g for g in grp["gene"].astype(str) if g in expr.index]
            if len(genes) < 20:
                continue
            sub = expr.loc[genes]
            z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1) + 1e-9,
                                                      axis=0)
            tot = z.sum(axis=0)
            n = len(genes)
            obs = float(stats.spearmanr(tot / n, stroma_score,
                                        nan_policy="omit")[0])

            # leave one gene out: (total - gene) / (n - 1)
            loo = np.array([
                float(stats.spearmanr((tot - z.loc[g]) / (n - 1), stroma_score,
                                      nan_policy="omit")[0])
                for g in genes])

            # patient bootstrap
            score = (tot / n).values
            sv = stroma_score.reindex(tot.index).values
            ok = np.isfinite(score) & np.isfinite(sv)
            idx_ok = np.where(ok)[0]
            boots = []
            for _ in range(n_boot):
                pick = rng.choice(idx_ok, size=len(idx_ok), replace=True)
                boots.append(stats.spearmanr(score[pick], sv[pick])[0])
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])

            worst = int(np.argmax(np.abs(loo - obs)))
            sens_rows.append({
                "cohort": name, "signature": sig, "direction": direction,
                "n_genes": n, "r_observed": obs,
                "loo_min": float(loo.min()), "loo_max": float(loo.max()),
                "max_gene_delta": float(np.abs(loo - obs).max()),
                "top_gene": genes[worst],
                "boot_ci_low": float(lo), "boot_ci_high": float(hi),
            })
            log.info(f"{sig+'|'+direction:30s} {obs:+7.3f} "
                     f"[{loo.min():+.3f},{loo.max():+.3f}] "
                     f"  [{lo:+.3f},{hi:+.3f}]")
        log.info("")

    # ---- matched null ----------------------------------------------------
    # Compartment scores and signature scores come from the SAME matrix, so any
    # two gene sets share technical structure (library size, batch, mean
    # expression) and will correlate to some degree with no biology involved.
    # Without a reference point "median |r| = 0.5" is uninterpretable. Random
    # sets matched on size and expression decile give that reference.
    null_rows = []
    if stroma_score is not None and n_null > 0:
        banner(log, f"{name} — MATCHED NULL")
        log.info("Random gene sets matched on size and expression decile.")
        log.info("")
        mean_expr = expr.mean(axis=1)
        try:
            bins = pd.qcut(mean_expr, 10, labels=False, duplicates="drop")
        except ValueError:
            bins = pd.Series(0, index=expr.index)
        bin_members = {b: expr.index[bins == b].tolist()
                       for b in pd.unique(bins.dropna())}

        log.info(f"{'signature':30s} {'r':>7s} {'null mean':>10s} "
                 f"{'null sd':>8s} {'z':>7s} {'p_emp':>7s}")
        for (sig, direction), grp in panel.groupby(["signature", "direction"]):
            genes = [g for g in grp["gene"].astype(str) if g in expr.index]
            if len(genes) < 20:
                continue
            obs = float(res.loc[res["score"] == f"{sig}|{direction}",
                                "r_stroma"].iloc[0])
            counts = pd.Series(bins.loc[genes].dropna()).value_counts().to_dict()

            null_r = []
            for _ in range(n_null):
                pick = []
                for b, k in counts.items():
                    pool = bin_members.get(b, [])
                    if pool:
                        pick.extend(rng.choice(pool, size=min(k, len(pool)),
                                               replace=False))
                if len(pick) < 5:
                    continue
                null_r.append(float(stats.spearmanr(
                    _zscore_score(expr, list(pick)), stroma_score,
                    nan_policy="omit")[0]))
            if not null_r:
                continue
            null_r = np.array(null_r)
            nm, nsd = float(np.nanmean(null_r)), float(np.nanstd(null_r))
            z = (obs - nm) / (nsd + 1e-12)
            n_ge = int((null_r >= obs).sum())
            n_le = int((null_r <= obs).sum())
            p_emp = float(min(1.0, 2.0 * (min(n_ge, n_le) + 1) / (len(null_r) + 1)))
            null_rows.append({
                "cohort": name, "signature": sig, "direction": direction,
                "n_genes": len(genes), "r_observed": obs,
                "null_mean": nm, "null_sd": nsd, "z": float(z),
                "p_emp": p_emp})
            log.info(f"{sig+'|'+direction:30s} {obs:+7.3f} {nm:+10.3f} "
                     f"{nsd:8.3f} {z:+7.2f} {p_emp:7.3f}")
        log.info("")
        if null_rows:
            nn = pd.DataFrame(null_rows)
            log.info(f"   null |r| averages {float(nn['null_mean'].abs().mean()):.3f}"
                     f" (sd {float(nn['null_sd'].mean()):.3f}) — this is the"
                     " baseline")
            log.info(f"   signatures exceeding |z| > 2: "
                     f"{int((nn['z'].abs() > 2).sum())}/{len(nn)}")
            log.info("")

    # Deliberately NOT returning comp_scores / sig_scores: holding them for
    # every cohort keeps hundreds of MB resident and is what makes a three-
    # cohort run die partway through on a workstation.
    del comp_scores, sig_df
    gc.collect()
    return {"correlations": res, "er": pd.DataFrame(er_rows),
            "sensitivity": pd.DataFrame(sens_rows),
            "null": pd.DataFrame(null_rows)}


def _common_universe(cfg: Config, log) -> set | None:
    """Genes shared by every cohort analysed so far, cached to results/.

    Null baselines are not comparable across cohorts when the gene universes
    differ: TCGA's 59,427-gene matrix includes many barely-detected genes whose
    random sets track sequencing depth, which itself tracks stromal content, so
    its null sits near +0.32 while METABRIC's sits near -0.30. Restricting all
    cohorts to a shared universe removes most of that difference.
    """
    path = cfg.paths.results / "wp3_common_genes.txt"
    if path.exists():
        genes = set(path.read_text(encoding="utf-8").split())
        log.info(f"   common gene universe: {len(genes)} genes "
                 f"(from {path.name})")
        return genes
    return None


def run(cfg: Config, n_markers: int = 50, n_boot: int = 200,
        n_null: int = 100, only: str | None = None,
        common_genes: bool = False) -> dict:
    """`only` restricts to one cohort (METABRIC / SCAN-B / TCGA), which is the
    reliable route on a memory-constrained machine: results are appended to the
    existing CSVs, so three separate runs give the same combined output as one."""
    log = get_logger("wp3", cfg.paths.logs)
    cfg.paths.ensure_outputs()

    report: dict = {"step": "wp3",
                    "generated": datetime.now().isoformat(timespec="seconds")}

    banner(log, "WP3 — DOES THE COMPARTMENT ARTEFACT REACH BULK COHORTS?")

    panel = pd.read_csv(cfg.paths.results / "signature_panel.csv")
    log.info(f"Signature panel: {panel['signature'].nunique()} signatures")
    log.info("")

    universe = _common_universe(cfg, log) if common_genes else None
    if common_genes and universe is None:
        log.info("   [!] --common-genes requested but wp3_common_genes.txt is")
        log.info("       absent. Run `dormbench wp3 --build-universe` first.")
    log.info("")

    banner(log, "COMPARTMENT MARKERS FROM THE ATLAS")
    markers = _compartment_markers(cfg, log, n_markers)
    log.info("")

    rng = np.random.default_rng(cfg.seed)
    outputs, all_corr, all_er, all_sens, all_null = {}, [], [], [], []

    # ---- METABRIC --------------------------------------------------------
    try:
        if only and only.upper() not in "METABRIC".upper():
            raise DataNotFound(f"skipped ({only} requested)")
        mb = load_metabric(cfg.paths.metabric_dir)
        expr = tidy_metabric_expression(mb["expr"])
        er = None
        smp = mb["sample"]
        er_col = next((c for c in smp.columns if "ER_STATUS" in c.upper()), None)
        id_col = next((c for c in ("SAMPLE_ID", "PATIENT_ID") if c in smp.columns),
                      None)
        if er_col and id_col:
            er = smp.set_index(id_col)[er_col].astype(str).str.strip().str.title()
        if universe:
            expr = expr.loc[expr.index.intersection(universe)]
            log.info(f"   restricted to common universe: {expr.shape[0]} genes")
        out = _analyse_cohort("METABRIC", expr, panel, markers, er, log,
                              n_boot, rng, n_null)
        outputs["METABRIC"] = out
        all_corr.append(out["correlations"])
        all_er.append(out["er"])
        all_sens.append(out["sensitivity"])
        all_null.append(out["null"])
        del expr, out
        gc.collect()
    except MemoryError:
        log.info("[X] METABRIC ran out of memory. Retry with fewer "
                 "permutations, e.g. --n-null 30 --n-boot 50")
    except (DataNotFound, FileNotFoundError, KeyError) as e:
        log.info(f"[!] METABRIC unavailable: {type(e).__name__}: {e}")

    # ---- SCAN-B ----------------------------------------------------------
    try:
        if only and only.upper() not in "SCAN-B".upper():
            raise DataNotFound(f"skipped ({only} requested)")
        sb = load_scanb(cfg.paths.scanb_dir)
        log.info("")
        log.info(f"Reading SCAN-B expression ({sb['expr_path'].name}) ...")
        sexpr = pd.read_csv(sb["expr_path"], index_col=0, low_memory=False)
        sexpr = sexpr.apply(pd.to_numeric, errors="coerce",
                            downcast="float")
        er = None
        cl = sb.get("clinical")
        if cl is not None:
            ercol = next((c for c in cl.columns if "er status" in c.lower()), None)
            tcol = next((c for c in ("title", "geo_accession") if c in cl.columns),
                        None)
            if ercol and tcol:
                er = (cl.set_index(tcol)[ercol].astype(str)
                      .map({"1": "Positive", "0": "Negative"}))
        if universe:
            sexpr = sexpr.loc[sexpr.index.intersection(universe)]
            log.info(f"   restricted to common universe: {sexpr.shape[0]} genes")
        out = _analyse_cohort("SCAN-B", sexpr, panel, markers, er, log,
                              n_boot, rng, n_null)
        outputs["SCAN-B"] = out
        all_corr.append(out["correlations"])
        all_er.append(out["er"])
        all_sens.append(out["sensitivity"])
        all_null.append(out["null"])
        del sexpr, out
        gc.collect()
    except MemoryError:
        log.info("[X] SCAN-B ran out of memory. Retry with --n-null 30 "
                 "--n-boot 50, or run one cohort at a time with --only")
    except (DataNotFound, FileNotFoundError, KeyError, ValueError) as e:
        log.info(f"[!] SCAN-B unavailable: {type(e).__name__}: {e}")

    # ---- TCGA-BRCA -------------------------------------------------------
    try:
        if only and only.upper() not in "TCGA-BRCA".upper():
            raise DataNotFound(f"skipped ({only} requested)")
        tc = load_tcga(cfg.paths.tcga_dir, log=log)
        log.info("")
        log.info(f"Reading TCGA ({tc['expr_path'].name}) ...")
        texpr = tc["expr"]
        # ORDER MATTERS: sample types are keyed on sample barcodes
        # (TCGA-3C-AAAU-01A) while clinical data is keyed on patient barcodes
        # (TCGA-3C-AAAU). Renaming to patient level first destroys the ability
        # to identify normals, which is how 113 of them silently survived an
        # earlier version of this step.
        stypes = tc.get("sample_types")
        tnorm = None
        if stypes is not None:
            st = stypes.reindex(texpr.columns)
            matched = int(st.notna().sum())
            log.info(f"   sample types matched for {matched}/{texpr.shape[1]} "
                     "columns")
            is_norm = st.astype(str).str.contains("normal", case=False, na=False)
            if is_norm.any():
                tnorm = texpr.loc[:, is_norm.values]
                texpr = texpr.loc[:, ~is_norm.values]
                log.info(f"   excluded {int(is_norm.sum())} non-tumour samples; "
                         f"{texpr.shape[1]} tumours retained")
            elif matched == 0:
                log.info("   [!] no sample types matched. Column identifiers do")
                log.info("       not look like sample barcodes; normals cannot")
                log.info("       be removed.")
        else:
            log.info("   [!] no tcga_sample_types.csv; run `dormbench tcgamap`")

        er = None
        # Review point 17. If `dormbench tcgaer` has written real
        # immunohistochemical ER calls, use them. PAM50 luminal/basal is a
        # subtype proxy, not ER status, and an analysis built on it cannot be
        # described as ER-stratified.
        er_ihc_path = cfg.paths.tcga_dir / "tcga_er_ihc.csv"
        if er_ihc_path.exists():
            _ihc = pd.read_csv(er_ihc_path)
            er = pd.Series(_ihc["er_status"].astype(str).values,
                           index=_ihc["patient"].astype(str))
            er = er[er.isin(["Positive", "Negative"])]
            log.info(f"   ER source: clinical IHC ({er_ihc_path.name})")
            log.info(f"   ER labels: {er.value_counts().to_dict()}")
            texpr = texpr.rename(
                columns=lambda c: "-".join(str(c).split("-")[:3]))
            texpr = texpr.loc[:, ~texpr.columns.duplicated()]

        cl = tc.get("clinical")
        if er is None and cl is not None:
            # Match ER columns on tokens, not substrings: "CANCER_STATUS"
            # contains "ER_STATUS" and was silently selected in an earlier
            # version, giving a table of NaNs.
            def _norm(c):
                return str(c).upper().replace(" ", "_").strip("_")

            def _is_er(c):
                n = _norm(c)
                toks = n.split("_")
                return (("ER" in toks and "STATUS" in toks)
                        or n.startswith("ER_STATUS")
                        or "ESTROGEN" in n
                        or n == "SUBTYPE")

            ercol = next((c for c in cl.columns if _is_er(c)), None)
            idcol = next((c for c in cl.columns
                          if _norm(c) in ("PATIENT_ID", "SAMPLE_ID",
                                          "BCR_PATIENT_BARCODE")), None)
            if ercol and idcol:
                log.info(f"   clinical: id='{idcol}', ER='{ercol}'")
                # pandas 3 uses Arrow-backed strings, where .astype(str) can
                # leave NA as a float. Go via object and fill before mapping.
                vals = (cl[ercol].astype("object").where(cl[ercol].notna(), "")
                        .astype(str).str.strip())
                if _norm(ercol) == "SUBTYPE":
                    # PAM50 proxy: Luminal = ER+, Basal = ER-
                    def _pam50(v):
                        if not isinstance(v, str) or not v:
                            return None
                        if "Lum" in v:
                            return "Positive"
                        if "Basal" in v:
                            return "Negative"
                        return None      # Her2, Normal-like, unclassified
                    vals = vals.map(_pam50)
                    log.info("   [!] using PAM50 SUBTYPE as an ER proxy "
                             "(Luminal=ER+, Basal=ER-; Her2 and Normal-like "
                             "excluded)")
                    log.info("   [!] This is NOT immunohistochemical ER status.")
                    log.info("       Do not describe the result as an")
                    log.info("       ER-stratified TCGA analysis. Run")
                    log.info("       `dormbench tcgaer` to obtain the real IHC")
                    log.info("       calls, or rename the analysis 'PAM50")
                    log.info("       luminal vs basal' throughout.")
                er = pd.Series(vals.values,
                               index=cl[idcol].astype("object").astype(str))
                er = er[er.notna()]
                log.info(f"   ER labels: {er.value_counts().to_dict()}")
                # now safe to move to patient level for the clinical join
                texpr = texpr.rename(
                    columns=lambda c: "-".join(str(c).split("-")[:3]))
                texpr = texpr.loc[:, ~texpr.columns.duplicated()]
            else:
                log.info("   [!] no usable ER column in the clinical file")
                log.info(f"       columns: {list(cl.columns)[:12]}")

        if universe:
            texpr = texpr.loc[texpr.index.intersection(universe)]
            log.info(f"   restricted to common universe: {texpr.shape[0]} genes")

        # TCGA-BRCA ships ~114 normal-adjacent samples alongside the tumours.
        # Normal breast is overwhelmingly stroma and adipose, so leaving them in
        # pushes almost every gene into positive correlation with the stromal
        # score and shifts the entire null. They are analysed separately.
        stypes = tc.get("sample_types")
        tnorm = None
        if stypes is not None:
            st = stypes.reindex(texpr.columns)
            # "Solid Tissue Normal" (old GDC) or "Normal" (current)
            is_norm = st.astype(str).str.contains("normal", case=False, na=False)
            if is_norm.any():
                log.info(f"   sample types: "
                         f"{st.value_counts().to_dict()}")
                tnorm = texpr.loc[:, is_norm.values]
                texpr = texpr.loc[:, ~is_norm.values]
                log.info(f"   excluded {int(is_norm.sum())} non-tumour samples; "
                         f"{texpr.shape[1]} tumours retained")
        else:
            log.info("   [!] no tcga_sample_types.csv; normals cannot be")
            log.info("       identified. Delete tcga_expression_matrix.parquet")
            log.info("       and re-run with gdc_sample_sheet.tsv present.")
        out = _analyse_cohort("TCGA-BRCA", texpr, panel, markers, er, log,
                              n_boot, rng, n_null)
        outputs["TCGA-BRCA"] = out
        all_corr.append(out["correlations"])
        all_er.append(out["er"])
        all_sens.append(out["sensitivity"])
        all_null.append(out["null"])
        # what does a dormancy score read in tissue with no tumour in it?
        if tnorm is not None and tnorm.shape[1] >= 20:
            banner(log, f"TCGA — NORMAL-ADJACENT TISSUE (n={tnorm.shape[1]})")
            log.info("A dormancy score is meant to report a cancer-cell state.")
            log.info("Normal breast contains no cancer cells at all, so what it")
            log.info("scores there is a direct check on what the score measures.")
            log.info("")
            rows_n = []
            for (sig, direction), grp in panel.groupby(["signature", "direction"]):
                genes = [g for g in grp["gene"].astype(str)]
                a = _zscore_score(texpr, genes)
                b = _zscore_score(tnorm, genes)
                if a.isna().all() or b.isna().all():
                    continue
                # z-scores are per-matrix, so recompute on the pooled data
                both = pd.concat([texpr, tnorm], axis=1)
                sc = _zscore_score(both, genes)
                mt = float(sc[texpr.columns].mean())
                mn = float(sc[tnorm.columns].mean())
                rows_n.append({"signature": sig, "direction": direction,
                               "tumour_mean": mt, "normal_mean": mn,
                               "diff": mn - mt})
            nd = pd.DataFrame(rows_n).sort_values("diff", ascending=False)
            log.info(f"{'signature':30s} {'tumour':>8s} {'normal':>8s} "
                     f"{'normal-tumour':>14s}")
            for _, r in nd.iterrows():
                log.info(f"{r['signature']+'|'+r['direction']:30s} "
                         f"{r['tumour_mean']:+8.3f} {r['normal_mean']:+8.3f} "
                         f"{r['diff']:+14.3f}")
            nd.to_csv(cfg.paths.results / "wp3_tcga_normal.csv", index=False)
            hi = nd[nd["diff"] > 0.3]
            log.info("")
            log.info(f"Signatures scoring HIGHER in normal breast than in "
                     f"tumour: {len(hi)}/{len(nd)}")
            if len(hi):
                log.info(f"   {', '.join(hi['signature'] + '|' + hi['direction'])}")
                log.info("   A dormancy score that reads higher in tissue with no")
                log.info("   cancer cells is not measuring a cancer-cell state.")
            report["tcga_normal"] = nd.to_dict("records")
            del tnorm
        del texpr, out
        gc.collect()
    except MemoryError:
        log.info("[X] TCGA ran out of memory. Retry with --n-null 30 "
                 "--n-boot 50, or run one cohort at a time with --only")
    except (DataNotFound, FileNotFoundError, KeyError, ValueError) as e:
        log.info(f"[!] TCGA unavailable: {type(e).__name__}: {e}")

    if not all_corr:
        raise FileNotFoundError("No bulk cohort could be loaded.")

    corr = pd.concat(all_corr, ignore_index=True)
    prev = cfg.paths.results / "wp3_bulk_correlations.csv"
    if only and prev.exists():
        old = pd.read_csv(prev)
        old = old[~old["cohort"].isin(corr["cohort"].unique())]
        corr = pd.concat([old, corr], ignore_index=True)
        log.info(f"   merged with existing results for "
                 f"{sorted(set(old['cohort']))}")
    corr.to_csv(prev, index=False)
    _merge_out_er = all_er
    def _merge_out(frames, name):
        """Write results, preserving cohorts not covered by this run.

        With --only, a run holds one cohort. Without merging, each run
        overwrites the previous and the other cohorts' numbers are lost from
        the results files even though they appear in the logs.
        """
        if not any(len(d) for d in frames):
            return None
        new = pd.concat(frames, ignore_index=True)
        path = cfg.paths.results / name
        if only and path.exists() and "cohort" in new.columns:
            old = pd.read_csv(path)
            if "cohort" in old.columns:
                old = old[~old["cohort"].isin(new["cohort"].unique())]
                new = pd.concat([old, new], ignore_index=True)
                log.info(f"   {name}: merged with "
                         f"{sorted(set(old['cohort']))}")
        new.to_csv(path, index=False)
        return new

    _merge_out(_merge_out_er, "wp3_er_stratified.csv")
    sens = _merge_out(all_sens, "wp3_sensitivity.csv")
    if sens is not None:
        banner(log, "SENSITIVITY ACROSS COHORTS")
        log.info(f"max shift from removing one gene: "
                 f"{float(sens['max_gene_delta'].max()):.4f}")
        wide = sens.pivot_table(index=["signature", "direction"],
                                columns="cohort", values="r_observed")
        if wide.shape[1] > 1:
            cons = wide.corr(method="spearman")
            log.info("")
            log.info("Cross-cohort consistency of the correlations (Spearman):")
            log.info(cons.round(3).to_string())
            report["cross_cohort_consistency"] = cons.round(4).to_dict()
        log.info("")

    nulls = _merge_out(all_null, "wp3_matched_null.csv")
    if nulls is not None:
        banner(log, "MATCHED NULL — ACROSS COHORTS")
        a = nulls["null_mean"].abs()
        base = float(a.median())
        log.info(f"Null |r| per signature: median {base:.3f}, "
                 f"range {float(a.min()):.3f}-{float(a.max()):.3f}")
        log.info("Each signature has its OWN null, matched to its size and")
        log.info("expression profile, so these differ. A high null means that")
        log.info("signature occupies expression deciles which are themselves")
        log.info("composition-linked -- which makes the test conservative, not")
        log.info("broken. Read the per-signature z, not the pooled median.")
        log.info("")
        # Group by signature AND direction. The up and down halves of a
        # signature routinely have z of opposite sign and similar magnitude,
        # so averaging them cancels a real effect to near zero.
        agg = (nulls.groupby(["signature", "direction"])
               .agg(r=("r_observed", "mean"), z=("z", "mean"),
                    z_min=("z", "min"), z_max=("z", "max"),
                    n_cohorts=("cohort", "nunique"))
               .sort_values("z", ascending=False))
        log.info(f"{'signature':24s} {'dir':5s} {'mean r':>8s} {'mean z':>8s} "
                 f"{'z range':>16s} {'n':>3s}")
        for (sig, direction), r in agg.iterrows():
            consistent = (r["z_min"] * r["z_max"] > 0) and abs(r["z"]) > 2
            flag = ("  <== beyond null, all cohorts" if consistent
                    else ("  <== beyond null" if abs(r["z"]) > 2 else ""))
            log.info(f"{sig:24s} {direction:5s} {r['r']:+8.3f} {r['z']:+8.2f} "
                     f"[{r['z_min']:+6.1f},{r['z_max']:+6.1f}] "
                     f"{int(r['n_cohorts']):3d}{flag}")
        agg = agg.reset_index().set_index(
            agg.reset_index()["signature"] + "|"
            + agg.reset_index()["direction"])
        report["null_baseline_abs_r"] = base
        report["n_beyond_null"] = int((agg["z"].abs() > 2).sum())
        report["n_signatures_tested"] = int(len(agg))
        report["null_z_by_signature"] = {k: round(float(v), 2)
                                         for k, v in agg["z"].items()}
        log.info("")
        log.info(f"NOTE: empirical p floors at {2/(n_null+1):.3f} with "
                 f"n_null={n_null}; report z, not p.")
        log.info("")

        # do the proliferation controls separate from the rest?
        log.info("")
        log.info("CAUTION: null means and SDs differ substantially between")
        log.info("cohorts (gene universe size and platform noise), so z is")
        log.info("comparable WITHIN a cohort but not directly BETWEEN them.")
        log.info("Use the sign and the consistency of direction across cohorts,")
        log.info("not the magnitude.")
        log.info("")
        ctrl = agg[agg.index.str.contains("E2F|G2M", case=False)]
        rest = agg[~agg.index.str.contains("E2F|G2M", case=False)]
        if len(ctrl) and len(rest):
            c_in = int((ctrl["z"].abs() <= 2).sum())
            r_out = int((rest["z"].abs() > 2).sum())
            log.info(f"Proliferation controls within null : {c_in}/{len(ctrl)}")
            log.info(f"Other signatures beyond null       : {r_out}/{len(rest)}")
            report["controls_within_null"] = f"{c_in}/{len(ctrl)}"
            report["others_beyond_null"] = f"{r_out}/{len(rest)}"
            if c_in == len(ctrl) and r_out >= 0.7 * len(rest):
                report["null_separates_controls"] = True
                log.info("")
                log.info("The cell-cycle controls are indistinguishable from")
                log.info("matched random gene sets, while the marker-containing")
                log.info("signatures are far outside theirs. Against a proper")
                log.info("null the two classes separate cleanly -- which the raw")
                log.info("correlations obscured, because the baseline is not")
                log.info("zero but roughly -0.2 to -0.4.")
            else:
                report["null_separates_controls"] = False
        log.info("")

    # ---- verdict ---------------------------------------------------------
    banner(log, "VERDICT")
    up = corr[corr["score"].str.endswith("|up")].copy()

    log.info("CAVEAT on r(cancer): the Cancer Epithelial marker set is derived")
    log.info("by specificity and is dominated by luminal differentiation genes")
    log.info("(SCGB2A2, AGR2, KRT19). It is a differentiation score, NOT a")
    log.info("purity estimate. Interpret r(cancer) accordingly; the stromal")
    log.info("marker sets (collagens, VWF/PLVAP) are cell-type-specific and")
    log.info("are the trustworthy half of this table.")
    log.info("")

    up["abs_stroma"] = up["r_stroma"].abs()
    log.info(f"{'signature':24s} {'r(stroma)':>10s} {'|r|':>7s}")
    for sig, g in up.groupby(up["score"].str.split("|").str[0]):
        log.info(f"{sig:24s} {float(g['r_stroma'].mean()):+10.3f} "
                 f"{float(g['abs_stroma'].mean()):7.3f}")
    log.info("")

    med_abs = float(up["abs_stroma"].median())
    n_strong = int((up["abs_stroma"] > 0.3).sum())
    log.info(f"median |r| with stromal content across {len(up)} scores: "
             f"{med_abs:.3f}")
    log.info(f"scores with |r| > 0.3: {n_strong}/{len(up)}")
    log.info("")

    def mean_r(tag):
        sub = up[up["score"].str.startswith(tag)]
        return float(sub["r_stroma"].mean()) if len(sub) else np.nan

    e2f_r, trad_r = mean_r("E2F_control"), mean_r("TRADITIOM_dormancy")
    report["e2f_r_stroma"] = None if np.isnan(e2f_r) else e2f_r
    report["traditiom_r_stroma"] = None if np.isnan(trad_r) else trad_r
    report["median_abs_r_stroma"] = med_abs
    report["n_scores_strong"] = n_strong

    if report.get("null_separates_controls"):
        verdict = "ARTEFACT_PROPAGATES"
        log.info("Against size- and expression-matched nulls, signatures")
        log.info("containing cell-type markers track composition far beyond")
        log.info("chance, while the cell-cycle controls do not. The single-cell")
        log.info("finding therefore has a direct, controlled consequence for")
        log.info("bulk application of these signatures, which is where they are")
        log.info("used clinically.")
        log.info("")
        log.info("Report z against the matched null, NOT the raw correlation:")
        log.info("the raw baseline is roughly -0.2 to -0.4, so an uncontrolled")
        log.info("correlation of -0.45 for a cell-cycle set looks meaningful")
        log.info("and is not.")
    elif med_abs > 0.3 and n_strong >= 0.7 * len(up):
        verdict = "COMPOSITION_DOMINATED"
        log.info("Nearly every signature score in these bulk cohorts is")
        log.info("strongly associated with cell-type composition. Note this")
        log.info("includes the cell-cycle controls, with the opposite sign:")
        log.info("proliferative tumours are more pure, so E2F is ANTI-stromal.")
        log.info("The honest claim is therefore stronger and simpler than the")
        log.info("one this step set out to test -- bulk gene-set scores are")
        log.info("largely composition measurements, whatever the gene set. Do")
        log.info("NOT write that the cell-cycle control is composition-free.")
    elif np.isfinite(trad_r) and np.isfinite(e2f_r) and trad_r > e2f_r + 0.25:
        verdict = "ARTEFACT_PROPAGATES"
        log.info("The dormancy score tracks stromal content much more strongly")
        log.info("than the cell-cycle control, so the single-cell finding has a")
        log.info("direct consequence for bulk application of these signatures.")
    elif np.isfinite(trad_r) and abs(trad_r) < 0.2:
        verdict = "NOT_IN_BULK"
        log.info("The dormancy score does NOT track stromal content in bulk.")
        log.info("Confine the claim to single-cell applications.")
    else:
        verdict = "PARTIAL"
        log.info("An association exists but is not cleanly separated from the")
        log.info("control. Report the table; do not overstate.")

    report["verdict"] = verdict
    # Report every cohort in the merged file, not only this run's, or a
    # --only run makes the summary look like a single-cohort study.
    try:
        report["cohorts"] = sorted(pd.read_csv(prev)["cohort"].unique().tolist())
        report["cohorts_this_run"] = list(outputs)
    except Exception:  # noqa: BLE001
        report["cohorts"] = list(outputs)
    log.info("")
    log.info(f"VERDICT: {verdict}")

    out_json = cfg.paths.results / "wp3_bulk.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {out_json}")
    log.info(f"Written: {cfg.paths.results / 'wp3_bulk_correlations.csv'}")

    try:
        from ..viz.bulk import plot_bulk
        fig = plot_bulk(corr, cfg.paths.figures / "fig7_bulk_composition.svg")
        log.info(f"Written: {fig}")
        report["figure"] = str(fig)
    except Exception as e:  # noqa: BLE001
        log.info(f"[!] Figure skipped: {type(e).__name__}: {e}")

    return report
