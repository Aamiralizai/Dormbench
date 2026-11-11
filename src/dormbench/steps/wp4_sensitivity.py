"""WP4 — sensitivity: is any result driven by one gene or one patient?

Three questions a reviewer will ask about the WP2d/WP2e attribution, each
answered by removing something and recomputing:

  A. Leave-one-gene-out. Drop each member gene in turn and recompute the
     AUROC. If one gene moves it substantially, the signature is not measuring
     a programme -- it is reporting that gene. This is cheap because the score
     is a mean: the leave-one-out score is (total - gene) / (n - 1), so the
     whole sweep costs one column extraction per gene.

  B. Gene bootstrap. Resample the gene set with replacement and recompute,
     giving a confidence interval that reflects gene-set composition rather
     than cell count. With ~75,000 cells any difference is "significant"; the
     honest uncertainty comes from which genes are in the set.

  C. Leave-one-donor-out. THE IMPORTANT ONE. The atlas has 100,064 cells from
     26 patients. Cell-level statistics therefore overstate precision by
     roughly the ratio of those numbers. Dropping each donor in turn shows
     whether the effect is a population property or one patient's tumour.

Scores here are mean log-normalised expression rather than AUCell, so that
leave-one-out becomes arithmetic. Absolute AUROCs will differ slightly from
WP2d; what matters is the spread under resampling, computed consistently.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..io import find_atlas_h5ad
from ..logging_utils import banner, get_logger

CANCER = "Cancer Epithelial"
STROMA = ("CAFs", "PVL", "Endothelial")
CYCLING = ("Cancer Cycling", "Cycling T-cells", "Cycling")


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run(cfg: Config, n_boot: int = 200, min_genes: int = 20,
        null_margin: float = 0.05) -> dict:
    log = get_logger("wp4", cfg.paths.logs)
    cfg.paths.ensure_outputs()

    import anndata as ad
    import scanpy as sc
    from scipy import sparse

    rng = np.random.default_rng(cfg.seed)
    report: dict = {"step": "wp4",
                    "generated": datetime.now().isoformat(timespec="seconds"),
                    "params": {"n_boot": n_boot, "min_genes": min_genes,
                               "null_margin": null_margin}}

    banner(log, "WP4 — SENSITIVITY: ONE GENE, OR ONE PATIENT?")

    panel = pd.read_csv(cfg.paths.results / "signature_panel.csv")

    log.info("Loading atlas ...")
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
    det = np.asarray((adata.X > 0).mean(axis=0)).ravel()
    adata = adata[:, det > 0.01].copy()

    ct = adata.obs["celltype_major"].astype(str).values
    minor = (adata.obs["celltype_minor"].astype(str)
             if "celltype_minor" in adata.obs
             else pd.Series(["?"] * adata.n_obs, index=adata.obs_names))
    donor = (adata.obs["donor_id"].astype(str).values
             if "donor_id" in adata.obs else np.array(["?"] * adata.n_obs))
    # Match the WP2c/d/e filter exactly: drop cycling-labelled cells AND the
    # top G2M quartile. Without the second term the cancer compartment retains
    # ~4,000 proliferating cells and AUROCs are not comparable across steps.
    keep = ~minor.str.contains("|".join(CYCLING), case=False, na=False).values
    prev = cfg.paths.results / "wp2b_cell_scores.csv.gz"
    if prev.exists():
        sc_prev = pd.read_csv(prev)
        g2m_col = next((c for c in sc_prev.columns
                        if "G2M" in c and c.startswith("AUCell")), None)
        if g2m_col and len(sc_prev) == adata.n_obs:
            g2m = sc_prev[g2m_col].values
            keep &= g2m <= np.nanquantile(g2m, 0.75)
            log.info("   cell filter matched to WP2c/d/e (cycling + top-quartile"
                     " G2M removed)")
        else:
            log.info("   [!] G2M scores unusable; filter is cycling-labels only,"
                     " so AUROCs will NOT match WP2d/WP2e")
    else:
        log.info("   [!] wp2b_cell_scores.csv.gz absent; filter is cycling-labels"
                 " only, so AUROCs will NOT match WP2d/WP2e")

    best = max(STROMA, key=lambda s: int((keep & (ct == s)).sum()))
    pos = keep & (ct == CANCER)
    neg = keep & (ct == best)
    log.info(f"   {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    log.info(f"   comparison: {CANCER} (n={int(pos.sum()):,}) vs "
             f"{best} (n={int(neg.sum()):,})")
    log.info(f"   donors: {len(np.unique(donor))}")
    log.info("")

    X = adata.X.tocsc() if sparse.issparse(adata.X) else sparse.csc_matrix(adata.X)
    gindex = {g: i for i, g in enumerate(adata.var_names)}

    loo_rows, boot_rows, donor_rows, summary = [], [], [], []

    for (sig, direction), grp in panel.groupby(["signature", "direction"]):
        if "G2M" in sig:
            continue
        idx = [gindex[g] for g in grp["gene"].astype(str) if g in gindex]
        genes = [g for g in grp["gene"].astype(str) if g in gindex]
        if len(idx) < min_genes:
            log.info(f"[skip] {sig}|{direction}: {len(idx)} genes "
                     f"(< {min_genes}); resampling is meaningless at this size")
            continue

        banner(log, f"{sig} | {direction}   ({len(idx)} genes)")
        sub = X[:, idx]
        total = np.asarray(sub.sum(axis=1)).ravel()
        n = len(idx)
        observed = _auroc((total / n)[pos], (total / n)[neg])
        log.info(f"   observed AUROC (mean-expression score): {observed:.3f}")

        # ---- A. leave one gene out --------------------------------------
        loo = np.empty(n)
        for j in range(n):
            col = np.asarray(X[:, idx[j]].todense()).ravel()
            s = (total - col) / (n - 1)
            loo[j] = _auroc(s[pos], s[neg])
        delta = loo - observed
        worst = int(np.argmax(np.abs(delta)))
        log.info(f"   LOO range: {loo.min():.3f} - {loo.max():.3f}")
        log.info(f"   largest single-gene effect: {genes[worst]} "
                 f"({delta[worst]:+.4f})")
        for j in np.argsort(-np.abs(delta))[:5]:
            loo_rows.append({"signature": sig, "direction": direction,
                             "gene": genes[j], "auroc_without": float(loo[j]),
                             "delta": float(delta[j])})

        # ---- B. gene bootstrap ------------------------------------------
        cols_r, rows_r, vals_r = [], [], []
        for b in range(n_boot):
            pick = rng.choice(n, size=n, replace=True)
            for k in pick:
                rows_r.append(idx[k]); cols_r.append(b); vals_r.append(1.0 / n)
        B = sparse.csr_matrix((vals_r, (rows_r, cols_r)),
                              shape=(adata.n_vars, n_boot), dtype=np.float32)
        BS = np.asarray((X @ B).todense())
        boot = np.array([_auroc(BS[pos, b], BS[neg, b]) for b in range(n_boot)])
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        log.info(f"   bootstrap 95% CI (gene resampling): {lo:.3f} - {hi:.3f}")
        if not (lo <= observed <= hi):
            # AUROC is a nonlinear function of the score, so the bootstrap
            # distribution need not be centred on the point estimate. This is
            # expected behaviour, not a failure.
            log.info("   note: the CI does not contain the observed value. "
                     "AUROC is")
            log.info("   nonlinear in the score, so resampling genes shifts the")
            log.info("   distribution; read the CI as a width, not as a test.")
        boot_rows.append({"signature": sig, "direction": direction,
                          "observed": observed, "ci_low": float(lo),
                          "ci_high": float(hi), "boot_sd": float(np.nanstd(boot))})

        # ---- C. leave one donor out -------------------------------------
        score = total / n
        d_res = []
        for d in np.unique(donor):
            m = donor != d
            a = _auroc(score[pos & m], score[neg & m])
            d_res.append({"donor": d, "auroc": a,
                          "n_cancer": int((pos & m).sum())})
        d_df = pd.DataFrame(d_res)
        d_delta = d_df["auroc"] - observed
        wd = int(np.argmax(np.abs(d_delta)))
        log.info(f"   donor-LOO range: {d_df['auroc'].min():.3f} - "
                 f"{d_df['auroc'].max():.3f}")
        log.info(f"   largest single-donor effect: {d_df['donor'].iat[wd]} "
                 f"({d_delta.iat[wd]:+.4f})")
        for _, r in d_df.iterrows():
            donor_rows.append({"signature": sig, "direction": direction,
                               "donor": r["donor"], "auroc_without": r["auroc"],
                               "delta": float(r["auroc"] - observed)})

        # A signature sitting at the null crosses 0.5 under any perturbation;
        # that is what a null control does, not a fragility. Only count a flip
        # when the point estimate was meaningfully away from 0.5 to begin with.
        near_null = abs(observed - 0.5) < null_margin
        crosses = (False if near_null
                   else bool(((observed - 0.5) * (loo - 0.5) < 0).any()))
        d_crosses = (False if near_null
                     else bool(((observed - 0.5) * (d_df["auroc"] - 0.5) < 0).any()))
        if near_null:
            log.info(f"   near the null (|AUROC-0.5| < {null_margin}); "
                     "sign-flip test not applicable")
        summary.append({
            "signature": sig, "direction": direction, "n_genes": n,
            "observed": observed,
            "loo_min": float(loo.min()), "loo_max": float(loo.max()),
            "max_gene_delta": float(np.abs(delta).max()),
            "top_gene": genes[worst],
            "boot_ci_low": float(lo), "boot_ci_high": float(hi),
            "donor_min": float(d_df["auroc"].min()),
            "donor_max": float(d_df["auroc"].max()),
            "max_donor_delta": float(np.abs(d_delta).max()),
            "top_donor": str(d_df["donor"].iat[wd]),
            "any_gene_flips_sign": crosses,
            "any_donor_flips_sign": d_crosses,
            "near_null": bool(near_null),
        })
        log.info("")

    if not summary:
        raise ValueError("No signature had enough genes for sensitivity testing.")

    summ = pd.DataFrame(summary)
    summ.to_csv(cfg.paths.results / "wp4_sensitivity.csv", index=False)
    pd.DataFrame(loo_rows).to_csv(
        cfg.paths.results / "wp4_leave_one_gene.csv", index=False)
    pd.DataFrame(donor_rows).to_csv(
        cfg.paths.results / "wp4_leave_one_donor.csv", index=False)

    banner(log, "SUMMARY")
    log.info(f"{'signature':24s} {'dir':5s} {'obs':>6s} {'gene Δ':>7s} "
             f"{'donor Δ':>8s} {'boot CI':>16s}")
    for _, r in summ.iterrows():
        log.info(f"{r['signature']:24s} {r['direction']:5s} "
                 f"{r['observed']:6.3f} {r['max_gene_delta']:7.4f} "
                 f"{r['max_donor_delta']:8.4f} "
                 f"  [{r['boot_ci_low']:.3f},{r['boot_ci_high']:.3f}]")
    log.info("")

    # ---- verdict ---------------------------------------------------------
    banner(log, "VERDICT")
    gene_fragile = summ[summ["any_gene_flips_sign"]]
    donor_fragile = summ[summ["any_donor_flips_sign"]]
    max_gene = float(summ["max_gene_delta"].max())
    max_donor = float(summ["max_donor_delta"].max())

    log.info(f"Largest AUROC shift from removing ONE GENE  : {max_gene:.4f}")
    log.info(f"Largest AUROC shift from removing ONE DONOR : {max_donor:.4f}")
    log.info("")
    n_null = int(summ["near_null"].sum())
    if n_null:
        log.info(f"Excluded from the sign-flip test ({n_null}) — already at the "
                 "null, so crossing 0.5 is expected:")
        for _, r in summ[summ["near_null"]].iterrows():
            log.info(f"   {r['signature']}|{r['direction']} "
                     f"(AUROC {r['observed']:.3f})")
        log.info("")
    report["n_near_null"] = n_null

    log.info(f"Signatures where some single gene flips the direction : "
             f"{len(gene_fragile)}/{len(summ) - n_null} testable")
    log.info(f"Signatures where some single donor flips the direction: "
             f"{len(donor_fragile)}/{len(summ) - n_null} testable")
    if len(donor_fragile):
        for _, r in donor_fragile.iterrows():
            log.info(f"   [!] {r['signature']}|{r['direction']} — donor "
                     f"{r['top_donor']} ({r['max_donor_delta']:+.3f})")
    log.info("")

    # which donors are repeatedly the most influential?
    top_donors = summ["top_donor"].value_counts()
    if len(top_donors) and top_donors.iat[0] >= max(3, 0.5 * len(summ)):
        log.info(f"[!] Donor {top_donors.index[0]} is the most influential for "
                 f"{top_donors.iat[0]}/{len(summ)} signatures.")
        log.info("    One patient carrying consistent leverage across unrelated")
        log.info("    gene sets points to unusual cell-type composition in that")
        log.info("    sample rather than to a gene-set property. Check its")
        log.info("    compartment proportions and say so in the text.")
        d0 = top_donors.index[0]
        m = donor == d0
        if m.sum():
            comp = pd.Series(ct[m]).value_counts(normalize=True)
            allc = pd.Series(ct).value_counts(normalize=True)
            log.info("")
            log.info(f"    {d0} composition vs cohort:")
            for c in comp.index[:6]:
                log.info(f"      {c:20s} {comp[c]*100:5.1f}%  "
                         f"(cohort {allc.get(c, 0)*100:5.1f}%)")
            report["influential_donor_composition"] = {
                k: float(v) for k, v in comp.items()}
        log.info("")
    report["most_influential_donor"] = (str(top_donors.index[0])
                                        if len(top_donors) else None)
    report["most_influential_donor_count"] = (int(top_donors.iat[0])
                                              if len(top_donors) else 0)

    flips = len(gene_fragile) + len(donor_fragile)
    if flips:
        verdict = ("DONOR_SENSITIVE" if len(donor_fragile) else "GENE_SENSITIVE")
        who = "patient" if len(donor_fragile) else "gene"
        log.info(f"At least one signature's direction depends on a single {who}.")
        if len(donor_fragile):
            log.info("With 26 donors that is a real limitation, not a")
            log.info("technicality: the effective sample size is donors, not")
            log.info("cells. Report the donor-LOO range, not a point estimate.")
        else:
            log.info("Name that gene in the text -- a signature that is really")
            log.info("one gene should be described as such.")
    elif max_donor > 2 * max_gene:
        verdict = "ROBUST_DONOR_LIMITED"
        log.info("No single gene and no single patient reverses any conclusion,")
        log.info("so the attribution is robust in direction.")
        log.info("")
        log.info(f"But removing one patient moves the AUROC up to "
                 f"{max_donor:.3f}, against {max_gene:.3f} for removing one")
        log.info("gene -- patient variation dominates gene variation by roughly")
        log.info(f"{max_donor/max(max_gene,1e-9):.1f}x. That is the honest")
        log.info("precision statement: with 26 donors, report donor-LOO ranges")
        log.info("rather than cell-level confidence intervals, which imply a")
        log.info("precision the design does not have.")
    else:
        verdict = "ROBUST"
        log.info("No single gene and no single patient changes any conclusion,")
        log.info("and neither source of variation dominates. Report the")
        log.info("leave-one-out ranges in supplementary.")

    report["verdict"] = verdict
    report["max_gene_delta"] = max_gene
    report["max_donor_delta"] = max_donor
    report["n_gene_fragile"] = int(len(gene_fragile))
    report["n_donor_fragile"] = int(len(donor_fragile))
    report["n_tested"] = int(len(summ))
    log.info("")
    log.info(f"VERDICT: {verdict}")

    out = cfg.paths.results / "wp4_sensitivity.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {out}")

    try:
        from ..viz.sensitivity import plot_sensitivity
        fig = plot_sensitivity(summ, pd.DataFrame(donor_rows),
                               cfg.paths.figures / "fig8_sensitivity.svg")
        log.info(f"Written: {fig}")
        report["figure"] = str(fig)
    except Exception as e:  # noqa: BLE001
        log.info(f"[!] Figure skipped: {type(e).__name__}: {e}")

    return report
