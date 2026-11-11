"""WP2b — score dormancy signatures per cell across tumour compartments.

This is the project's go/no-go gate. It answers claim C1: are published
dormancy signatures actually cancer-cell-specific, or do they score highest in
stroma? Given that NR2F1 -- the canonical dormancy marker -- turned out to be
predominantly a CAF gene, the answer is not obvious.

Design choices that matter for review
-------------------------------------
Two independent scoring methods. AUCell (rank-based, decoupler) and scanpy's
score_genes (mean-centred against an expression-matched background). Reviewers
do not accept single-method signature scoring, and the two disagree in
informative ways when a signature is driven by a few high-expressors.

Up and down scored separately. A signed signature scored as one set conflates
"dormancy programme on" with "proliferation programme off". Each direction is
scored alone and the difference reported as the combined score.

Cell-cycle deconfounding is explicit. The atlas contains a 'Cancer Cycling'
population, giving an internal positive control: if a dormancy score merely
inverts proliferation, it will separate Cancer Cycling from other cancer cells
as strongly as it separates compartments. Both comparisons are reported.

Effect sizes, not p-values. With 100k cells every comparison is significant.
AUROC and Cohen's d are reported instead.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..genes import assert_no_dates, resolve_against
from ..io import find_atlas_h5ad
from ..logging_utils import banner, get_logger

CANCER_LABEL = "Cancer Epithelial"
STROMAL_LABELS = ["CAFs", "PVL", "Endothelial"]
IMMUNE_LABELS = ["T-cells", "Myeloid", "B-cells", "Plasmablasts"]


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC; 0.5 = no separation."""
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    from scipy.stats import rankdata

    allv = np.concatenate([pos, neg])
    r = rankdata(allv)
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def _map_to_symbols(adata, log) -> None:
    """Replace Ensembl var_names with gene symbols, collapsing duplicates."""
    if "feature_name" not in adata.var.columns:
        log.info("   [!] no feature_name column; leaving identifiers as-is")
        return
    sym = adata.var["feature_name"].astype(str)
    n_dup = int(sym.duplicated().sum())
    adata.var["ensembl_id"] = adata.var_names
    adata.var_names = pd.Index(sym)
    adata.var_names_make_unique()
    log.info(f"   mapped Ensembl -> symbols ({n_dup} duplicate symbols made unique)")


def run(cfg: Config, subsample: int | None = None, er_only: bool = False,
        method: str = "both") -> dict:
    log = get_logger("wp2b", cfg.paths.logs)
    cfg.paths.ensure_outputs()

    import anndata as ad
    import scanpy as sc

    report: dict = {
        "step": "wp2b",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "params": {"subsample": subsample, "er_only": er_only, "method": method},
    }

    banner(log, "WP2b — SIGNATURE SCORING ACROSS COMPARTMENTS")

    # ---- signature panel ------------------------------------------------
    panel_path = cfg.paths.results / "signature_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} not found. Run `dormbench wp2a` first."
        )
    panel = pd.read_csv(panel_path)
    assert_no_dates(panel)
    log.info(f"Signature panel: {panel_path}")
    for (sig, d), n in panel.groupby(["signature", "direction"]).size().items():
        log.info(f"   {sig:22s} {d:5s} {n:5d} genes")
    log.info("")

    # ---- atlas ----------------------------------------------------------
    path = find_atlas_h5ad(cfg.paths.atlas_dir)
    log.info(f"Loading {path.name} ...")
    adata = ad.read_h5ad(path)
    log.info(f"   {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # Use raw counts if present; CELLxGENE keeps them in .raw
    if adata.raw is not None:
        log.info("   using .raw for counts, re-normalising")
        adata = ad.AnnData(
            X=adata.raw.X.copy(), obs=adata.obs.copy(),
            var=adata.raw.var.copy() if adata.raw.var is not None else adata.var.copy(),
        )
    _map_to_symbols(adata, log)

    if er_only and "subtype" in adata.obs:
        # Anchored: "HER2+" contains "ER" and would otherwise be kept.
        keep = adata.obs["subtype"].astype(str).str.match(
            r"^\s*ER[\s+-]*(pos|positive)?\s*$", case=False)
        adata = adata[keep].copy()
        log.info(f"   ER+ subset: {adata.n_obs:,} cells")

    if subsample and subsample < adata.n_obs:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(adata.n_obs, subsample, replace=False)
        adata = adata[idx].copy()
        log.info(f"   subsampled to {adata.n_obs:,} cells (testing only)")

    log.info("   normalising (CPM + log1p) ...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    log.info("")

    present = set(adata.var_names)
    sets: dict[str, list[str]] = {}
    for (sig, direction), grp in panel.groupby(["signature", "direction"]):
        genes = resolve_against(grp["gene"].astype(str).tolist(), present)
        key = f"{sig}|{direction}"
        sets[key] = genes
        log.info(f"   {key:30s} {len(genes):5d}/{len(grp):5d} genes found "
                 f"({100*len(genes)/max(len(grp),1):.0f}%)")
    report["gene_recovery"] = {k: len(v) for k, v in sets.items()}
    log.info("")

    # ---- scoring --------------------------------------------------------
    banner(log, "1. SCORING")
    scores = pd.DataFrame(index=adata.obs_names)

    if method in ("both", "aucell"):
        try:
            import decoupler as dc

            net = pd.concat([
                pd.DataFrame({"source": k, "target": v})
                for k, v in sets.items() if len(v) >= 5
            ], ignore_index=True)
            log.info(f"   AUCell on {net['source'].nunique()} sets ...")
            aucell = getattr(getattr(dc, "mt", dc), "aucell")
            aucell(adata, net=net, verbose=False)
            key = next((k for k in adata.obsm if "aucell" in k.lower()), None)
            if key:
                res = adata.obsm[key]
                for c in res.columns:
                    scores[f"AUCell|{c}"] = res[c].values
                log.info(f"   done ({len(res.columns)} sets)")
            else:
                log.info("   [!] AUCell produced no obsm entry")
        except Exception as e:  # noqa: BLE001
            log.info(f"   [!] AUCell failed: {type(e).__name__}: {e}")

    if method in ("both", "scanpy"):
        log.info("   scanpy score_genes ...")
        for k, genes in sets.items():
            if len(genes) < 5:
                continue
            name = f"sg_{k.replace('|', '_')}"
            sc.tl.score_genes(adata, gene_list=genes, score_name=name,
                              random_state=cfg.seed)
            scores[f"scanpy|{k}"] = adata.obs[name].values
        log.info("   done")
    log.info("")

    # combined signed scores: up minus down
    for prefix in ("AUCell", "scanpy"):
        for sig in panel["signature"].unique():
            up, dn = f"{prefix}|{sig}|up", f"{prefix}|{sig}|down"
            if up in scores and dn in scores:
                scores[f"{prefix}|{sig}|combined"] = scores[up] - scores[dn]

    meta_cols = [c for c in ("celltype_major", "celltype_minor", "celltype_subset",
                             "subtype", "donor_id", "nFeature_RNA", "nCount_RNA",
                             "percent_mito")
                 if c in adata.obs.columns]
    out = pd.concat([adata.obs[meta_cols].reset_index(drop=True),
                     scores.reset_index(drop=True)], axis=1)
    score_path = cfg.paths.results / "wp2b_cell_scores.csv.gz"
    out.to_csv(score_path, index=False, compression="gzip")
    log.info(f"   Written: {score_path}")
    log.info("")

    # ---- compartment attribution ---------------------------------------
    banner(log, "2. COMPARTMENT ATTRIBUTION  (claim C1)")
    ct = adata.obs["celltype_major"].astype(str).values
    score_cols = [c for c in scores.columns if c.endswith("|combined")] or \
                 [c for c in scores.columns if "|up" in c]

    rows = []
    for col in score_cols:
        v = scores[col].values
        means = {lab: float(np.nanmean(v[ct == lab])) for lab in np.unique(ct)}
        top = max(means, key=means.get)
        cancer = v[ct == CANCER_LABEL]
        best_stroma_lab = max(
            (l for l in STROMAL_LABELS if l in means), key=lambda l: means[l],
            default=None)
        stroma = v[ct == best_stroma_lab] if best_stroma_lab else np.array([])
        rows.append({
            "score": col,
            "top_compartment": top,
            "mean_cancer": means.get(CANCER_LABEL, np.nan),
            "top_stromal": best_stroma_lab,
            "mean_top_stromal": means.get(best_stroma_lab, np.nan),
            "auroc_cancer_vs_stroma": _auroc(cancer, stroma),
            "cohens_d_cancer_vs_stroma": _cohens_d(cancer, stroma),
            "cancer_is_top": top == CANCER_LABEL,
        })
        log.info(f"{col}")
        log.info(f"   top compartment : {top}")
        log.info(f"   cancer mean     : {means.get(CANCER_LABEL, float('nan')):+.4f}")
        log.info(f"   {str(best_stroma_lab):15s} : {means.get(best_stroma_lab, float('nan')):+.4f}")
        log.info(f"   AUROC cancer>stroma : {rows[-1]['auroc_cancer_vs_stroma']:.3f}"
                 f"   Cohen's d {rows[-1]['cohens_d_cancer_vs_stroma']:+.2f}")
        log.info("")

    attrib = pd.DataFrame(rows)
    attrib.to_csv(cfg.paths.results / "wp2b_compartment_attribution.csv", index=False)
    report["attribution"] = attrib.to_dict("records")

    # per-compartment mean table
    tbl = (pd.DataFrame({**{"celltype_major": ct},
                         **{c: scores[c].values for c in score_cols}})
           .groupby("celltype_major").mean().round(4))
    log.info("Mean score by compartment:")
    log.info(tbl.to_string())
    tbl.to_csv(cfg.paths.results / "wp2b_scores_by_compartment.csv")
    log.info("")

    # ---- cell-cycle deconfounding --------------------------------------
    banner(log, "3. IS IT JUST PROLIFERATION?")
    cc = {}
    g2m_cols = [c for c in scores.columns if "G2M" in c]
    dorm_cols = [c for c in score_cols if "G2M" not in c]
    for d in dorm_cols:
        for g in g2m_cols:
            r = float(np.corrcoef(scores[d].values, scores[g].values)[0, 1])
            cc[f"{d} vs {g}"] = r
            log.info(f"   corr({d}, {g}) = {r:+.3f}")

    if "celltype_minor" in adata.obs.columns:
        minor = adata.obs["celltype_minor"].astype(str).values
        cyc = minor == "Cancer Cycling"
        noncyc = (ct == CANCER_LABEL) & ~cyc
        log.info("")
        log.info(f"   Cancer Cycling n={cyc.sum():,}   other cancer n={noncyc.sum():,}")
        for d in dorm_cols:
            a = _auroc(scores[d].values[noncyc], scores[d].values[cyc])
            cc[f"{d}|noncycling_vs_cycling_auroc"] = a
            log.info(f"   AUROC non-cycling > cycling  {d}: {a:.3f}")
    report["cellcycle"] = cc
    log.info("")

    # ---- verdict --------------------------------------------------------
    banner(log, "VERDICT  (week-3 gate)")
    dorm_rows = attrib[~attrib["score"].str.contains("G2M")]
    n_cancer_top = int(dorm_rows["cancer_is_top"].sum())
    log.info(f"Dormancy scores with cancer epithelial as top compartment: "
             f"{n_cancer_top}/{len(dorm_rows)}")
    for _, r in dorm_rows.iterrows():
        log.info(f"   {r['score']:40s} top={r['top_compartment']}")
    log.info("")

    if n_cancer_top == 0:
        verdict = "STROMAL"
        log.info("All dormancy signatures score highest OUTSIDE cancer epithelium.")
        log.info("This generalises the NR2F1 result to the whole signature class")
        log.info("and is a strong, publishable negative. Proceed to WP3.")
    elif n_cancer_top == len(dorm_rows):
        verdict = "CANCER_INTRINSIC"
        log.info("All dormancy signatures are cancer-cell-dominant. The")
        log.info("compartment confound does not apply; the paper pivots to")
        log.info("whether the score predicts late relapse (WP4).")
    else:
        verdict = "MIXED"
        log.info("Signatures disagree on compartment. This is the most")
        log.info("interesting outcome: it means 'dormancy signature' is not one")
        log.info("thing, and compartment attribution is a necessary step before")
        log.info("any of them is used. Proceed to WP3.")

    report["verdict"] = verdict
    report["n_cancer_top"] = n_cancer_top
    report["n_dormancy_scores"] = int(len(dorm_rows))
    log.info("")
    log.info(f"VERDICT: {verdict}")

    # ---- figure ---------------------------------------------------------
    try:
        from ..viz.compartment import plot_compartment_scores
        fig = plot_compartment_scores(
            pd.DataFrame({**{"celltype_major": ct},
                          **{c: scores[c].values for c in score_cols}}),
            cfg.paths.figures / "fig1_compartment_attribution.svg")
        log.info(f"Written: {fig}")
        report["figure"] = str(fig)
    except Exception as e:  # noqa: BLE001
        log.info(f"[!] Figure skipped: {type(e).__name__}: {e}")

    out_json = cfg.paths.results / "wp2b_compartment.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {out_json}")

    return report
