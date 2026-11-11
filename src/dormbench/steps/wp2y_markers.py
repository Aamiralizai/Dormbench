"""WP2y — is compartment attribution explained by marker-gene content?

WP2e established that compartment attribution varies widely between gene sets
(AUROC 0.107 to 0.680) without an evident structure: the effect is not specific
to dormancy, is not explained by in vitro derivation, and is not consistent
within any origin. This step tests the remaining obvious explanation, which is
also the one a reviewer asks for first:

    is a signature's compartment preference simply a function of how many
    cell-type marker genes it contains?

That makes a falsifiable prediction. If marker content is the explanation, the
fraction of a signature made up of stromal markers should predict its AUROC
across the whole panel, and removing those genes should move the AUROC toward
0.5. If a signature retains its preference on the disjoint gene set, marker
content is not sufficient and something else is operating.

Two marker sources are used, and the analysis is repeated with each:

  atlas       the specificity-ranked markers derived in WP3 from the same
              single-cell reference. Internally consistent but circular: the
              markers and the tested scores come from one dataset.

  literature  canonical markers taken from the primary literature and marker
              databases, independent of this dataset. Smaller and less
              comprehensive, but not circular.

Agreement between the two is itself informative. Divergence would indicate that
the atlas-derived result depends on how the markers were selected.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config
from ..genes import resolve_against
from ..io import find_atlas_h5ad
from ..logging_utils import banner, get_logger

CANCER = "Cancer Epithelial"
STROMAL = ["CAFs", "PVL", "Endothelial"]
CYCLING = ("Cancer Cycling", "Cycling T-cells", "Cycling")

# Canonical markers from the primary literature and marker databases.
# VERIFY against PanglaoDB / CellMarker before publication; these are a
# deliberately conservative core rather than a comprehensive panel.
LITERATURE_MARKERS = {
    "CAFs": ["DCN", "LUM", "COL1A1", "COL1A2", "COL3A1", "FAP", "PDGFRB",
             "PDGFRA", "THY1", "POSTN", "FBLN1", "MMP2", "SPARC", "VIM",
             "ACTA2", "TAGLN", "COL5A1", "COL6A3", "FN1", "THBS2"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "PLVAP", "ACKR1",
                    "EGFL7", "RAMP2", "AQP1", "MMRN2", "ESAM", "TIE1",
                    "KDR", "NOTCH4", "SOX17", "ERG", "FLI1", "ENG",
                    "PODXL", "NRP1"],
    "PVL": ["RGS5", "ACTA2", "MYH11", "PDGFRB", "NOTCH3", "CSPG4", "DES",
            "MCAM", "KCNJ8", "HIGD1B"],
    "Immune": ["PTPRC", "CD3D", "CD3E", "CD2", "CD8A", "CD68", "CD14",
               "LYZ", "AIF1", "ITGAM", "MS4A1", "CD79A", "IGKC", "JCHAIN"],
    "Cancer Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "ESR1", "GATA3",
                          "FOXA1", "AGR2", "SCGB2A2", "TFF1", "CDH1", "MUC1"],
}


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run(cfg: Config, min_disjoint: int = 20) -> dict:
    log = get_logger("wp2y", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    report: dict = {"step": "wp2y",
                    "generated": datetime.now().isoformat(timespec="seconds")}

    banner(log, "WP2y — DOES MARKER CONTENT EXPLAIN COMPARTMENT ATTRIBUTION?")

    import anndata as ad
    import scanpy as sc
    from scipy import sparse

    panel = pd.read_csv(cfg.paths.results / "signature_panel.csv")

    # ---- marker sources -------------------------------------------------
    sources = {"literature": LITERATURE_MARKERS}
    mk = cfg.paths.results / "wp3_compartment_markers.json"
    if mk.exists():
        with open(mk, encoding="utf-8") as fh:
            sources["atlas"] = json.load(fh)
        log.info("   marker sources: literature (curated) and atlas (WP3)")
    else:
        log.info("   [!] wp3_compartment_markers.json absent; literature only")
    log.info("")

    # ---- atlas ----------------------------------------------------------
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
    keep = ~minor.str.contains("|".join(CYCLING), case=False, na=False).values
    best = max(STROMAL, key=lambda s: int((keep & (ct == s)).sum()))
    pos, neg = keep & (ct == CANCER), keep & (ct == best)
    present = set(adata.var_names)
    universe = len(present)
    log.info(f"   {adata.n_obs:,} cells, {universe:,} genes; "
             f"{CANCER} vs {best}")
    log.info("")

    X = adata.X.tocsc() if sparse.issparse(adata.X) else sparse.csc_matrix(adata.X)
    gidx = {g: i for i, g in enumerate(adata.var_names)}

    def score(genes):
        idx = [gidx[g] for g in genes if g in gidx]
        if len(idx) < 3:
            return None
        return np.asarray(X[:, idx].mean(axis=1)).ravel()

    rows = []
    for src, markers in sources.items():
        stromal_markers = set()
        for c in STROMAL:
            stromal_markers |= set(markers.get(c, []))
        stromal_markers = {g for g in stromal_markers if g in present}
        log.info(f"[{src}] stromal marker pool: {len(stromal_markers)} genes")
        if len(stromal_markers) < 10:
            log.info("   [!] very few markers matched the reference dataset;")
            log.info("       check gene symbol conventions before interpreting")

        for (sig, direction), grp in panel.groupby(["signature", "direction"]):
            genes = resolve_against(grp["gene"].astype(str).tolist(), present)
            if len(genes) < min_disjoint:
                continue
            gset = set(genes)
            overlap = gset & stromal_markers
            disjoint = sorted(gset - stromal_markers)

            # hypergeometric enrichment of stromal markers in the signature
            p_over = float(stats.hypergeom.sf(
                len(overlap) - 1, universe, len(stromal_markers), len(gset))) \
                if len(overlap) else 1.0

            v_all = score(genes)
            v_dis = score(disjoint) if len(disjoint) >= min_disjoint else None
            a_all = _auroc(v_all[pos], v_all[neg]) if v_all is not None else np.nan
            a_dis = _auroc(v_dis[pos], v_dis[neg]) if v_dis is not None else np.nan

            rows.append({
                "source": src, "signature": sig, "direction": direction,
                "n_genes": len(gset), "n_marker_overlap": len(overlap),
                "overlap_pct": round(100 * len(overlap) / len(gset), 2),
                "hypergeom_p": p_over,
                "auroc_all": a_all, "auroc_disjoint": a_dis,
                # Positive = the disjoint set sits CLOSER to 0.5 than the full
                # set, i.e. removing markers moved the signature toward chance.
                # The operands were previously reversed, which inverted the
                # sign and reported zero signatures moving toward the null.
                "shift_toward_null": (abs(a_all - 0.5) - abs(a_dis - 0.5))
                if np.isfinite(a_dis) and np.isfinite(a_all) else np.nan,
                "n_disjoint": len(disjoint),
            })
        log.info("")

    res = pd.DataFrame(rows)
    res.to_csv(cfg.paths.results / "wp2y_marker_overlap.csv", index=False)

    # ---- does overlap predict AUROC? ------------------------------------
    for src in res["source"].unique():
        d = res[(res["source"] == src) & (res["direction"] == "up")]
        banner(log, f"{src.upper()} MARKERS — up-direction signatures")
        log.info(f"{'signature':24s} {'n':>5s} {'ovl':>5s} {'ovl%':>6s} "
                 f"{'AUROC':>7s} {'disjoint':>9s} {'shift':>7s}")
        for _, r in d.sort_values("auroc_all").iterrows():
            log.info(f"{r['signature']:24s} {int(r['n_genes']):5d} "
                     f"{int(r['n_marker_overlap']):5d} {r['overlap_pct']:6.1f} "
                     f"{r['auroc_all']:7.3f} {r['auroc_disjoint']:9.3f} "
                     f"{r['shift_toward_null']:+7.3f}")
        ok = d["auroc_all"].notna() & d["overlap_pct"].notna()
        if ok.sum() > 4:
            rho, p = stats.spearmanr(d.loc[ok, "overlap_pct"],
                                     d.loc[ok, "auroc_all"])
            log.info("")
            log.info(f"   overlap% vs AUROC:  Spearman rho = {rho:+.3f} "
                     f"(p = {p:.3g}, n = {int(ok.sum())})")
            report[f"rho_overlap_auroc_{src}"] = float(rho)
            report[f"p_overlap_auroc_{src}"] = float(p)
            moved = d[(d["shift_toward_null"] > 0.05)]
            log.info(f"   signatures moving >0.05 toward 0.5 after marker "
                     f"removal: {len(moved)}/{len(d)}")
            report[f"n_moved_toward_null_{src}"] = int(len(moved))
            report[f"n_tested_{src}"] = int(len(d))
            report[f"median_overlap_pct_{src}"] = float(d["overlap_pct"].median())
            report[f"median_shift_{src}"] = float(
                d["shift_toward_null"].median())
        log.info("")

    # ---- verdict ---------------------------------------------------------
    banner(log, "VERDICT")
    src0 = "literature" if "rho_overlap_auroc_literature" in report else "atlas"
    n_moved = int(report.get(f"n_moved_toward_null_{src0}", 0))
    n_tested = int(report.get(f"n_tested_{src0}", 0)) or 1
    rho = report.get("rho_overlap_auroc_literature",
                     report.get("rho_overlap_auroc_atlas", np.nan))
    p = report.get("p_overlap_auroc_literature",
                   report.get("p_overlap_auroc_atlas", np.nan))
    log.info(f"Correlation between stromal marker content and AUROC: "
             f"rho = {rho:+.3f} (p = {p:.3g})")
    log.info("")

    if not np.isfinite(rho):
        verdict = "NOT_COMPUTED"
        log.info("The correlation could not be computed -- too few signatures")
        log.info("with both an AUROC and a marker overlap. Check that the")
        log.info("marker gene symbols match the reference dataset.")
    elif rho < -0.5 and np.isfinite(p) and p < 0.05 and n_moved >= 0.5 * n_tested:
        verdict = "MARKER_CONTENT_EXPLAINS"
        log.info("Marker content predicts compartment attribution AND removing")
        log.info("marker genes moves signatures toward chance. The effect is")
        log.info("largely definitional: a signature scores in the compartment")
        log.info("whose markers it contains.")
    elif rho < -0.5 and np.isfinite(p) and p < 0.05:
        verdict = "CORRELATED_NOT_CAUSAL"
        log.info(f"Marker content correlates with attribution (rho = {rho:+.2f})")
        log.info(f"but removing marker genes moved only {n_moved}/{n_tested}")
        log.info("signatures materially toward chance. Marker content is")
        log.info("therefore a correlate, not the cause.")
        log.info("")
        log.info("The overlaps are also small in absolute terms, so a handful")
        log.info("of canonical markers cannot produce the observed effect. This")
        log.info("is consistent with a small, consistent shift distributed")
        log.info("across many genes rather than contamination by a few. Report")
        log.info("the correlation and the disjoint-set result together, and do")
        log.info("NOT describe the effect as definitional.")
    elif np.isfinite(rho) and abs(rho) < 0.3:
        verdict = "MARKER_CONTENT_INSUFFICIENT"
        log.info("Marker content does not predict compartment attribution.")
        log.info("The effect therefore does not reduce to signatures")
        log.info("containing marker genes, and no mechanism is currently")
        log.info("supported. Report the observation without an explanation")
        log.info("rather than proposing one the data does not carry.")
    else:
        verdict = "PARTIAL"
        log.info("Marker content is associated with attribution but does not")
        log.info("account for it. Report both the correlation and the")
        log.info("disjoint-set results, and avoid a single-cause claim.")

    report["verdict"] = verdict
    log.info("")
    log.info(f"VERDICT: {verdict}")
    with open(cfg.paths.results / "wp2y_marker_overlap.json", "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {cfg.paths.results / 'wp2y_marker_overlap.csv'}")
    return report
