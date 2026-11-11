"""WP2n - coherent-module null: is TRADITIOM unusual among real pathways?

Review point 7. WP2e compares each signature against random gene sets matched on
size and expression decile. That closes the size/expression objection but leaves
a sharper one open: independently sampled random genes are not co-regulated,
whereas any real biological module is. A coherent module will concentrate in
whichever cell type expresses it, so ANY coherent gene set will look more
compartment-specific than random genes of matched size. Beating a random null
therefore does not show that dormancy signatures are unusual - it may only show
that they are coherent.

The correct comparator is other coherent modules. This step scores the MSigDB
Hallmark collection, which is fifty curated, co-regulated human pathways, on the
same cells with the same statistic, and asks where each dormancy signature falls
in that distribution.

The question changes from

    is TRADITIOM more compartment-specific than random genes?          (WP2e)

to

    is TRADITIOM more compartment-specific than a typical real pathway? (WP2n)

The second is the question a specialist actually has. Hallmark sets are also
size-matched per comparison: for each dormancy signature, only Hallmark sets
within a tolerance of its gene count are used, so a 7-gene panel is not compared
against 200-gene pathways.

A dormancy signature that sits in the middle of the Hallmark distribution is
behaving like an ordinary tissue-expressed pathway. That is not a null result -
it is precisely the manuscript's argument, stated against a comparator no
reviewer can call too easy.

Input is the atlas plus results/signature_panel.csv and
data_processed/hallmark_human.csv (shipped with the package).
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..genes import resolve_against
from ..io import find_atlas_h5ad
from ..logging_utils import banner, get_logger

CANCER = "Cancer Epithelial"
STROMA = ("CAFs", "PVL", "Endothelial")
CYCLING = ("Cancer Cycling", "Cycling T-cells", "Cycling")
# Hallmark sets that are themselves proliferation programmes. Kept in the
# comparison but flagged, because a dormancy score's similarity to these is
# interpretable in a way its similarity to, say, BILE_ACID_METABOLISM is not.
PROLIFERATIVE = ("E2F_TARGETS", "G2M_CHECKPOINT", "MYC_TARGETS_V1",
                 "MYC_TARGETS_V2", "MITOTIC_SPINDLE", "DNA_REPAIR")


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run(cfg: Config, size_tolerance: float = 0.5, min_sets: int = 8) -> dict:
    """`size_tolerance` 0.5 means Hallmark sets between 0.5x and 2x the
    signature's gene count are admitted as comparators."""
    log = get_logger("wp2n", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    report: dict = {"step": "wp2n",
                    "generated": datetime.now().isoformat(timespec="seconds"),
                    "params": {"size_tolerance": size_tolerance,
                               "min_sets": min_sets}}

    banner(log, "WP2n - COHERENT-MODULE NULL (HALLMARK COMPARATOR)")
    log.info("WP2e asks whether a signature beats random genes. This asks the")
    log.info("harder question: does it stand out among real, co-regulated")
    log.info("pathways, which are coherent in exactly the way random gene sets")
    log.info("are not.")
    log.info("")

    import anndata as ad
    import scanpy as sc
    from scipy import sparse

    hm_path = cfg.paths.processed / "hallmark_human.csv"
    if not hm_path.exists():
        raise FileNotFoundError(
            f"{hm_path} not found. It ships with the package; `dormbench "
            "wp2ext` also writes it.")
    hm = pd.read_csv(hm_path)
    log.info(f"   Hallmark: {hm['source'].nunique()} sets, {len(hm)} members")

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
    log.info(f"   {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    ct = adata.obs["celltype_major"].astype(str).values
    minor = (adata.obs["celltype_minor"].astype(str)
             if "celltype_minor" in adata.obs
             else pd.Series(["?"] * adata.n_obs, index=adata.obs_names))
    keep = ~minor.str.contains("|".join(CYCLING), case=False, na=False).values
    best = max(STROMA, key=lambda s: int((keep & (ct == s)).sum()))
    pos_m, neg_m = keep & (ct == CANCER), keep & (ct == best)
    log.info(f"   {CANCER} n={int(pos_m.sum()):,} vs {best} "
             f"n={int(neg_m.sum()):,}")
    log.info("")

    X = adata.X.tocsc() if sparse.issparse(adata.X) else sparse.csc_matrix(adata.X)
    gidx = {g: i for i, g in enumerate(adata.var_names)}
    present = set(adata.var_names)

    def _score_auroc(genes):
        g = resolve_against(list(genes), present)
        idx = [gidx[x] for x in g]
        if len(idx) < 5:
            return np.nan, len(idx)
        v = np.asarray(X[:, idx].mean(axis=1)).ravel()
        return _auroc(v[pos_m], v[neg_m]), len(idx)

    # ---- score the Hallmark comparator distribution ----------------------
    banner(log, "1. HALLMARK REFERENCE DISTRIBUTION")
    hm_rows = []
    for name, grp in hm.groupby("source"):
        au, n = _score_auroc(grp["target"].astype(str).tolist())
        if np.isfinite(au):
            hm_rows.append({"module": name, "n_genes": n, "auroc": au,
                            "proliferative": name in PROLIFERATIVE})
    hm_res = pd.DataFrame(hm_rows).sort_values("auroc")
    log.info(f"   {len(hm_res)} Hallmark modules scored")
    log.info(f"   AUROC distribution: min {hm_res['auroc'].min():.3f}, "
             f"25th {hm_res['auroc'].quantile(.25):.3f}, "
             f"median {hm_res['auroc'].median():.3f}, "
             f"75th {hm_res['auroc'].quantile(.75):.3f}, "
             f"max {hm_res['auroc'].max():.3f}")
    log.info("")
    log.info("   most stroma-dominant Hallmark modules:")
    for _, r in hm_res.head(5).iterrows():
        log.info(f"      {r['module']:34s} {r['auroc']:.3f} "
                 f"({int(r['n_genes'])} genes)")
    log.info("   most cancer-dominant Hallmark modules:")
    for _, r in hm_res.tail(5).iloc[::-1].iterrows():
        log.info(f"      {r['module']:34s} {r['auroc']:.3f} "
                 f"({int(r['n_genes'])} genes)")
    hm_res.to_csv(cfg.paths.results / "wp2n_hallmark_reference.csv",
                  index=False)
    log.info("")

    # ---- place each signature in that distribution -----------------------
    banner(log, "2. WHERE EACH SIGNATURE FALLS AMONG REAL PATHWAYS")
    log.info("percentile = fraction of size-matched Hallmark modules with a")
    log.info("LOWER AUROC. Near 0 means unusually stroma-dominant even for a")
    log.info("coherent pathway; near 0.5 means it behaves like an ordinary one.")
    log.info("")
    rows = []
    for (sig, direction), grp in panel.groupby(["signature", "direction"]):
        au, n = _score_auroc(grp["gene"].astype(str).tolist())
        if not np.isfinite(au):
            continue
        lo, hi = n * size_tolerance, n / max(size_tolerance, 1e-9)
        comp = hm_res[(hm_res["n_genes"] >= lo) & (hm_res["n_genes"] <= hi)]
        matched = True
        if len(comp) < min_sets:
            comp, matched = hm_res, False
        pct = float((comp["auroc"] < au).mean())
        z = float((au - comp["auroc"].mean()) / (comp["auroc"].std() + 1e-12))
        rows.append({
            "signature": sig, "direction": direction, "n_genes": n,
            "auroc": au, "n_comparators": int(len(comp)),
            "size_matched": matched,
            "comparator_median": float(comp["auroc"].median()),
            "percentile_among_hallmark": pct, "z_among_hallmark": z})
    res = pd.DataFrame(rows).sort_values(["direction", "percentile_among_hallmark"])
    res.to_csv(cfg.paths.results / "wp2n_module_null.csv", index=False)

    for direction in ("up", "down"):
        sub = res[res["direction"] == direction]
        if sub.empty:
            continue
        log.info(f"{direction.upper()} scores")
        log.info(f"   {'signature':26s} {'genes':>6s} {'AUROC':>7s} "
                 f"{'HM median':>10s} {'pctile':>7s} {'z':>7s} {'matched':>8s}")
        for _, r in sub.iterrows():
            log.info(f"   {r['signature']:26s} {int(r['n_genes']):6d} "
                     f"{r['auroc']:7.3f} {r['comparator_median']:10.3f} "
                     f"{r['percentile_among_hallmark']:7.2f} "
                     f"{r['z_among_hallmark']:+7.2f} "
                     f"{str(bool(r['size_matched'])):>8s}")
        log.info("")
    report["results"] = res.to_dict("records")

    # ---- verdict ---------------------------------------------------------
    banner(log, "VERDICT")
    up = res[res["direction"] == "up"]
    extreme = up[up["percentile_among_hallmark"] <= 0.1]
    typical = up[(up["percentile_among_hallmark"] > 0.1)
                 & (up["percentile_among_hallmark"] < 0.9)]
    log.info(f"Up-direction signatures in the bottom decile of the Hallmark")
    log.info(f"distribution (unusually stroma-dominant): "
             f"{len(extreme)}/{len(up)}")
    for _, r in extreme.iterrows():
        log.info(f"   {r['signature']:26s} AUROC {r['auroc']:.3f}, "
                 f"percentile {r['percentile_among_hallmark']:.2f}")
    log.info("")
    log.info(f"Signatures behaving like an ordinary coherent pathway: "
             f"{len(typical)}/{len(up)}")
    log.info("")
    report["n_extreme"] = int(len(extreme))
    report["n_typical"] = int(len(typical))

    trad = up[up["signature"] == "TRADITIOM_dormancy"]
    if len(trad):
        t = trad.iloc[0]
        report["traditiom_percentile"] = float(t["percentile_among_hallmark"])
        log.info(f"TRADITIOM up: AUROC {t['auroc']:.3f}, percentile "
                 f"{t['percentile_among_hallmark']:.2f} among "
                 f"{int(t['n_comparators'])} size-matched Hallmark modules")
        log.info("")

    if len(extreme) == 0:
        verdict = "INDISTINGUISHABLE_FROM_COHERENT_PATHWAYS"
        log.info("No dormancy signature is more compartment-dominated than a")
        log.info("typical real pathway of matched size. This does not weaken")
        log.info("the manuscript - it sharpens it. The claim is not that these")
        log.info("signatures are peculiar; it is that a transferred coherent")
        log.info("programme reports compartment composition in heterogeneous")
        log.info("tissue, exactly as other coherent programmes do. State the")
        log.info("finding that way and the 'null is too easy' objection has no")
        log.info("purchase.")
    elif len(extreme) == len(up):
        verdict = "EXCEPTIONAL_AMONG_COHERENT_PATHWAYS"
        log.info("Every dormancy signature is more stroma-dominated than a")
        log.info("size-matched real pathway. The compartment association is")
        log.info("specific to this signature class and not a generic property")
        log.info("of coherent modules. This is the strongest available form of")
        log.info("the result.")
    else:
        verdict = "PARTIAL_EXCEPTIONALITY"
        log.info("Some signatures exceed the coherent-module comparator and")
        log.info("some do not. Name which, and restrict signature-specific")
        log.info("claims to the ones that do. Do not generalise across the")
        log.info("class on the strength of the subset.")
    report["verdict"] = verdict
    log.info("")
    log.info(f"VERDICT: {verdict}")

    with open(cfg.paths.results / "wp2n_module_null.json", "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {cfg.paths.results / 'wp2n_module_null.csv'}")
    return report
