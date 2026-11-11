"""WP2e — size- and expression-matched null control.

WP2d reports AUROCs for cancer epithelium versus stroma that differ sharply
between signatures (0.147 to 0.562). Before any of that can be interpreted, one
objection has to be closed: gene sets differ in size and in the expression level
of their members, and both affect a score's behaviour independently of biology.
A 7-gene set of low-expressed genes and a 739-gene set of housekeeping genes
will not behave alike even if neither carries dormancy information.

For every real signature this step draws random gene sets matched on
  (a) number of genes, and
  (b) mean expression, by binning all genes into expression deciles and drawing
      the same number from each bin as the real set occupies,
then computes the identical statistic. The real value is reported as an
empirical percentile and z-score against that null.

The same control is applied to the bimodality coefficient, because the WP2d
bimodality ranking tracked gene-set size closely enough to be suspect: at ~97%
sparsity a 7-gene score is mostly zeros, and a spike at zero inflates Sarle's
coefficient without any subpopulation being present.

Scores here use mean log-normalised expression rather than AUCell, so that
thousands of permutations reduce to a single sparse matrix product. The
statistic differs from WP2d's but the null and the observation are computed
identically, which is what the comparison requires.
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


def _auroc_matrix(scores: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """AUROC for each column of `scores`, pos vs neg boolean masks."""
    from scipy.stats import rankdata

    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n0 == 0:
        return np.full(scores.shape[1], np.nan)
    sub = scores[pos | neg]
    lbl = pos[pos | neg]
    out = np.empty(scores.shape[1])
    for j in range(scores.shape[1]):
        r = rankdata(sub[:, j])
        out[j] = (r[lbl].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def _bimodality(v: np.ndarray) -> float:
    from scipy.stats import kurtosis, skew

    v = v[np.isfinite(v)]
    if len(v) < 50:
        return np.nan
    n = len(v)
    g, k = skew(v), kurtosis(v, fisher=True)
    denom = k + 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    return float((g**2 + 1) / denom) if denom else np.nan


def _bh(p):
    """Benjamini-Hochberg adjusted p values, NaN-safe."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    q = p[ok]
    if q.size == 0:
        return out
    order = np.argsort(q)
    ranked = q[order] * q.size / (np.arange(q.size) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty_like(ranked)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def run(cfg: Config, n_perm: int = 1000, n_bins: int = 10) -> dict:
    log = get_logger("wp2e", cfg.paths.logs)
    cfg.paths.ensure_outputs()

    import anndata as ad
    import scanpy as sc
    from scipy import sparse

    rng = np.random.default_rng(cfg.seed)
    report: dict = {"step": "wp2e",
                    "generated": datetime.now().isoformat(timespec="seconds"),
                    "params": {"n_perm": n_perm, "n_bins": n_bins}}

    banner(log, "WP2e — SIZE- AND EXPRESSION-MATCHED NULL CONTROL")

    panel = pd.read_csv(cfg.paths.results / "signature_panel.csv")
    scores_prev = pd.read_csv(cfg.paths.results / "wp2b_cell_scores.csv.gz")

    log.info(f"Loading atlas ...")
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

    # restrict to reasonably detected genes; undetected genes cannot be in a
    # meaningful null and inflate the low-expression bins
    det = np.asarray((adata.X > 0).mean(axis=0)).ravel()
    keep_genes = det > 0.01
    adata = adata[:, keep_genes].copy()
    log.info(f"   {adata.n_obs:,} cells x {adata.n_vars:,} genes "
             f"(detected in >1% of cells)")

    ct = adata.obs["celltype_major"].astype(str).values
    minor = (adata.obs["celltype_minor"].astype(str)
             if "celltype_minor" in adata.obs else pd.Series(["?"] * adata.n_obs))

    g2m_col = next((c for c in scores_prev.columns
                    if "G2M" in c and c.startswith("AUCell")), None)
    g2m = scores_prev[g2m_col].values if g2m_col else np.zeros(adata.n_obs)
    is_cyc = minor.str.contains("|".join(CYCLING), case=False, na=False).values
    keep = ~(is_cyc | (g2m > np.nanquantile(g2m, 0.75)))
    log.info(f"   non-cycling subset: {int(keep.sum()):,} cells")
    log.info("")

    # expression bins
    mean_expr = np.asarray(adata.X.mean(axis=0)).ravel()
    bins = pd.qcut(pd.Series(mean_expr), n_bins, labels=False, duplicates="drop")
    bins = np.asarray(bins)
    bin_members = {b: np.where(bins == b)[0] for b in np.unique(bins)}
    gene_index = {g: i for i, g in enumerate(adata.var_names)}

    X = adata.X.tocsc() if sparse.issparse(adata.X) else sparse.csc_matrix(adata.X)

    pos = keep & (ct == CANCER)
    best_stroma, best_n = None, -1
    for s in STROMA:
        n = int((keep & (ct == s)).sum())
        if n > best_n:
            best_stroma, best_n = s, n
    neg = keep & (ct == best_stroma)
    log.info(f"Comparison: {CANCER} (n={int(pos.sum()):,}) vs "
             f"{best_stroma} (n={int(neg.sum()):,})")
    log.info("")

    rows = []
    for (sig, direction), grp in panel.groupby(["signature", "direction"]):
        if "G2M" in sig:
            continue
        idx = [gene_index[g] for g in grp["gene"].astype(str) if g in gene_index]
        if len(idx) < 5:
            log.info(f"   {sig}|{direction}: only {len(idx)} genes present, skipped")
            continue

        # real score = mean log-normalised expression over the set
        real = np.asarray(X[:, idx].mean(axis=1)).ravel()

        # build a matched null: same bin composition, same size
        counts = pd.Series(bins[idx]).value_counts().to_dict()
        # The signature's own genes are removed from the pool it is compared
        # against. Leaving them in lets a null replicate redraw part of the
        # real set, which biases the null toward the observed value and
        # shrinks the apparent effect.
        own = set(int(i) for i in idx)
        pools = {b: np.array([g for g in m if int(g) not in own], dtype=int)
                 for b, m in bin_members.items()}
        cols = []
        for _ in range(n_perm):
            picks = []
            for b, k in counts.items():
                pool = pools.get(b, np.array([], dtype=int))
                if len(pool) == 0:
                    continue
                picks.append(rng.choice(pool, size=min(k, len(pool)),
                                        replace=False))
            cols.append(np.concatenate(picks) if picks else np.array([], dtype=int))

        # one sparse product for all permutations
        rowi, coli, vals = [], [], []
        for j, pick in enumerate(cols):
            if len(pick) == 0:
                continue
            rowi.extend(pick.tolist())
            coli.extend([j] * len(pick))
            vals.extend([1.0 / len(pick)] * len(pick))
        B = sparse.csr_matrix((vals, (rowi, coli)),
                              shape=(adata.n_vars, n_perm), dtype=np.float32)
        null = np.asarray((X @ B).todense()) if sparse.issparse(X @ B) else np.asarray(X @ B)

        auroc_real = float(_auroc_matrix(real[:, None], pos, neg)[0])
        auroc_null = _auroc_matrix(null, pos, neg)
        pct = float((auroc_null < auroc_real).mean())
        z = float((auroc_real - np.nanmean(auroc_null)) / (np.nanstd(auroc_null) + 1e-12))
        # two-sided empirical p with the standard +1 correction, capped at 1
        n_ge = int((auroc_null >= auroc_real).sum())
        n_le = int((auroc_null <= auroc_real).sum())
        p_emp = float(min(1.0, 2.0 * (min(n_ge, n_le) + 1) / (len(auroc_null) + 1)))

        bim_real = _bimodality(real[pos])
        bim_null = np.array([_bimodality(null[pos, j]) for j in range(null.shape[1])])
        bim_pct = float(np.nanmean(bim_null < bim_real))

        rows.append({
            "signature": sig, "direction": direction, "n_genes": len(idx),
            "auroc_real": auroc_real,
            "auroc_null_mean": float(np.nanmean(auroc_null)),
            "auroc_null_sd": float(np.nanstd(auroc_null)),
            "auroc_percentile": pct, "auroc_z": z, "auroc_p_emp": p_emp,
            "bimodality_real": bim_real,
            "bimodality_null_mean": float(np.nanmean(bim_null)),
            "bimodality_percentile": bim_pct,
        })
        log.info(f"{sig}|{direction}  ({len(idx)} genes)")
        log.info(f"   AUROC real {auroc_real:.3f}   null "
                 f"{np.nanmean(auroc_null):.3f} +/- {np.nanstd(auroc_null):.3f}"
                 f"   z={z:+.2f}  p={p_emp:.3f}")
        log.info(f"   bimodality real {bim_real:.3f}   null "
                 f"{np.nanmean(bim_null):.3f}   percentile {bim_pct:.2f}")
        log.info("")

    res = pd.DataFrame(rows)
    # Multiplicity across the panel. Every signature is tested against its own
    # matched null, so the family is the panel and BH is applied across it.
    if "auroc_p_emp" in res.columns:
        res["auroc_q_bh"] = np.nan
        for d, idx in res.groupby("direction").groups.items():
            res.loc[idx, "auroc_q_bh"] = _bh(res.loc[idx, "auroc_p_emp"].values)
        n_q = int((res["auroc_q_bh"] < 0.05).sum())
        log.info("")
        log.info(f"   Benjamini-Hochberg across the panel: {n_q} of "
                 f"{len(res)} scores significant at q < 0.05")
        report["n_significant_bh"] = n_q
    res.to_csv(cfg.paths.results / "wp2e_null_control.csv", index=False)
    report["results"] = res.to_dict("records")

    # ---- verdict ---------------------------------------------------------
    banner(log, "VERDICT")
    up = res[res["direction"] == "up"]
    sig_dev = up[np.abs(up["auroc_z"]) > 2]
    log.info(f"Up-direction signatures deviating from matched null "
             f"(|z| > 2): {len(sig_dev)}/{len(up)}")
    for _, r in up.sort_values("auroc_z").iterrows():
        tag = "  <== beyond null" if abs(r["auroc_z"]) > 2 else "  (within null)"
        log.info(f"   {r['signature']:22s} AUROC {r['auroc_real']:.3f}  "
                 f"z={r['auroc_z']:+6.2f}{tag}")
    log.info("")

    bim_flag = up[up["bimodality_percentile"] < 0.95]
    log.info("Bimodality within cancer epithelium:")
    for _, r in up.iterrows():
        verdict_b = ("exceeds null" if r["bimodality_percentile"] >= 0.95
                     else "explained by set size / sparsity")
        log.info(f"   {r['signature']:22s} real {r['bimodality_real']:.3f}  "
                 f"null {r['bimodality_null_mean']:.3f}  -> {verdict_b}")
    log.info("")

    # A wide matched null reduces sensitivity to MODERATE effects, but does not
    # cap the achievable z: a signature far enough from 0.5 still clears the
    # threshold (TGF-beta, null SD 0.120, reaches z = -3.3). So this is
    # reported descriptively, not as a categorical "cannot be significant".
    wide_null = up[up["auroc_null_sd"] > 0.10]
    if len(wide_null):
        log.info("[!] WIDE MATCHED NULL")
        log.info("    These signatures have a null SD > 0.10, which reduces")
        log.info("    sensitivity to moderate compartment effects. Large")
        log.info("    effects remain detectable, so a non-significant result")
        log.info("    is uninformative for a moderate effect but is not a")
        log.info("    ceiling on significance:")
        for _, r in wide_null.iterrows():
            lo = r["auroc_null_mean"] - 2 * r["auroc_null_sd"]
            hi = r["auroc_null_mean"] + 2 * r["auroc_null_sd"]
            sig_now = "significant" if abs(r["auroc_z"]) > 2 else "n.s."
            log.info(f"      {r['signature']:22s} {int(r['n_genes']):4d} genes, "
                     f"null 95% [{lo:.2f}, {hi:.2f}], observed AUROC "
                     f"{r['auroc_real']:.3f} ({sig_now})")
        log.info("")
    # kept as 'underpowered' key for backward compatibility of downstream code,
    # but the values are now "wide-null" signatures, some of which ARE significant
    report["wide_null_signatures"] = wide_null["signature"].tolist()
    report["underpowered"] = wide_null["signature"].tolist()

    if len(sig_dev) == 0:
        verdict = "WITHIN_NULL"
        log.info("No signature departs from its matched null. The AUROC")
        log.info("differences in WP2d are attributable to gene-set size and")
        log.info("expression level rather than dormancy content. This is a")
        log.info("strong negative and must be reported as such.")
    elif len(sig_dev) == len(up):
        verdict = "ALL_BEYOND_NULL"
        log.info("Every signature departs from its matched null. The")
        log.info("compartment attribution is carrying real information beyond")
        log.info("set size and expression level.")
    else:
        verdict = "PARTIAL"
        log.info("Some signatures depart from the null and some do not. Only")
        log.info("the departing ones support a biological interpretation; the")
        log.info("rest should be reported as indistinguishable from random")
        log.info("gene sets of matched size and expression.")

    report["verdict"] = verdict
    report["n_beyond_null"] = int(len(sig_dev))
    log.info("")
    log.info(f"VERDICT: {verdict}")

    out = cfg.paths.results / "wp2e_null.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {out}")
    log.info(f"Written: {cfg.paths.results / 'wp2e_null_control.csv'}")

    try:
        from ..viz.nullctl import plot_null_control
        fig = plot_null_control(res, cfg.paths.figures / "fig4_null_control.svg")
        log.info(f"Written: {fig}")
        report["figure"] = str(fig)
    except Exception as e:  # noqa: BLE001
        log.info(f"[!] Figure skipped: {type(e).__name__}: {e}")

    return report
