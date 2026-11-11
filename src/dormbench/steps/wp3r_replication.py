"""WP3r - independent donor-level replication in a second breast scRNA-seq atlas.

Review point 4, and the single analysis most likely to change a reviewer's
verdict. The central biological conclusion currently rests on 8-9 evaluable
ER-positive donors from one atlas. Direction consistency across those donors is
real, but n=9 is n=9 however many cells sit underneath it, and a percentile
bootstrap at that size reports intervals more precise than the evidence.

This runs the WP8 donor-level test unchanged on a SECOND, independent atlas:
same signature panel, same compartments, same within-donor paired Wilcoxon, same
multiplicity control. Nothing is refitted or retuned for the new data, which is
what makes it a replication rather than a second exploratory analysis.

    Pal et al. 2021 (EMBO J), GEO GSE161529, is the default target: a large
    human breast atlas with a substantial ER-positive tumour cohort, annotated
    independently of Wu et al.

Column names differ between atlases, so `celltype_col`, `donor_col` and
`subtype_col` are arguments. The step prints what it found before doing any
work and stops with a clear message rather than guessing, because a silently
mismapped cell-type column would produce a confident wrong answer - the failure
mode this project has already hit twice with GEO annotation.

Compartment labels are mapped onto the canonical four via a substring table.
Anything unmatched is reported and excluded rather than forced into a bucket.

Read the outcome honestly. Replication means the DIRECTION and CONSISTENCY of
the within-donor difference reproduce. Effect sizes will differ - different
protocol, different annotation, different patients - and that is expected, not
a failure.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config
from ..genes import resolve_against
from ..io import find_atlas_h5ad
from ..logging_utils import banner, get_logger
from .wp8_donor_level import CANCER, STROMAL, _bh, _boot_ci

MIN_CELLS = 10
MIN_DONORS = 4

# Substring -> canonical compartment. Checked in order, case-insensitively.
COMPARTMENT_MAP = [
    ("cancer epithelial", CANCER), ("malignant", CANCER), ("tumor", CANCER),
    ("tumour", CANCER), ("epithelial", CANCER),
    # "smooth muscle" must be tested BEFORE "vascular": a Cell Ontology label
    # such as "vascular associated smooth muscle cell" contains "vascular" and
    # would otherwise be counted as endothelium, although vessel-wall smooth
    # muscle is perivascular.
    ("smooth muscle", "PVL"),
    ("endothelial", "Endothelial"), ("vascular", "Endothelial"),
    ("caf", "CAFs"), ("fibroblast", "CAFs"), ("stromal", "CAFs"),
    ("pvl", "PVL"), ("perivascular", "PVL"), ("pericyte", "PVL"),
    ("myoepithelial", None), ("immune", None), ("t-cell", None),
    ("t cell", None), ("b-cell", None), ("b cell", None),
    ("myeloid", None), ("plasma", None), ("mast", None),
]


def _canonical(label: str) -> str | None:
    s = str(label).strip().lower()
    for key, target in COMPARTMENT_MAP:
        if key in s:
            return target
    return None


def _paired(scores: pd.DataFrame, sig: str, comp: str):
    sub = scores[scores["signature"] == sig]
    wide = sub.pivot_table(index="donor_id", columns="compartment",
                           values="score")
    if CANCER not in wide.columns or comp not in wide.columns:
        return None
    pair = wide[[CANCER, comp]].dropna()
    if len(pair) < MIN_DONORS:
        return None
    diff = (pair[CANCER] - pair[comp]).values
    try:
        _, p = stats.wilcoxon(diff)
    except ValueError:
        p = np.nan
    return {"n_donors": int(len(diff)), "mean_diff": float(np.mean(diff)),
            "n_lower_in_cancer": int((diff < 0).sum()),
            "wilcoxon_p": float(p), "diffs": diff}


def run(cfg: Config, atlas_dir: str | None = None, tag: str = "pal",
        celltype_col: str | None = None, donor_col: str | None = None,
        subtype_col: str | None = None, er_only: bool = True,
        n_boot: int = 2000, inspect: bool = False) -> dict:
    log = get_logger("wp3r", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    report: dict = {"step": "wp3r",
                    "generated": datetime.now().isoformat(timespec="seconds"),
                    "params": {"tag": tag, "er_only": er_only,
                               "atlas_dir": str(atlas_dir)}}

    banner(log, f"WP3r - INDEPENDENT DONOR-LEVEL REPLICATION ({tag})")
    log.info("The WP8 test, unchanged, on a second atlas. Nothing is retuned.")
    log.info("")

    import anndata as ad
    import scanpy as sc
    from scipy import sparse

    root = Path(atlas_dir) if atlas_dir else (
        cfg.paths.data_root / "GSE161529")
    if not root.exists():
        log.info(f"[X] atlas directory not found: {root}")
        log.info("    Download a second breast scRNA-seq atlas (GSE161529,")
        log.info("    Pal et al. 2021) and pass --atlas-dir, or set")
        log.info("    paths.replication_dir in config.yaml.")
        report["verdict"] = "BLOCKED"
        return report

    path = find_atlas_h5ad(root)
    log.info(f"Loading {path.name} ...")
    adata = ad.read_h5ad(path)
    log.info(f"   {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    log.info("")

    # ---- inspect obs before trusting anything ---------------------------
    banner(log, "1. CELL METADATA")
    for c in adata.obs.columns:
        vals = adata.obs[c]
        n_uniq = int(vals.nunique(dropna=True))
        if n_uniq <= 30:
            top = vals.astype(str).value_counts().head(6).to_dict()
            log.info(f"   {c:26s} {n_uniq:5d} values  e.g. {top}")
        else:
            log.info(f"   {c:26s} {n_uniq:5d} values (too many to list)")
    log.info("")

    def _guess(explicit, keys):
        if explicit:
            return explicit if explicit in adata.obs.columns else None
        for k in keys:
            for c in adata.obs.columns:
                if k in c.lower():
                    return c
        return None

    ct_col = _guess(celltype_col, ["celltype_major", "celltype", "cell_type",
                                   "annotation", "cluster_label"])
    d_col = _guess(donor_col, ["donor_id", "donor", "patient", "sample_id",
                               "orig.ident"])
    s_col = _guess(subtype_col, ["subtype", "er_status", "group", "condition"])
    log.info(f"   cell-type column : {ct_col}")
    log.info(f"   donor column     : {d_col}")
    log.info(f"   subtype column   : {s_col}")
    log.info("")
    if inspect:
        log.info("--inspect given: stopping before analysis.")
        report["verdict"] = "INSPECTED"
        return report
    if ct_col is None or d_col is None:
        log.info("[X] Cannot identify the cell-type and donor columns.")
        log.info("    Inspect the listing above and pass them explicitly:")
        log.info("    dormbench wp3r --celltype-col <name> --donor-col <name>")
        log.info("    Guessing here would silently produce a wrong answer.")
        report["verdict"] = "BLOCKED"
        return report

    # ---- map compartments ------------------------------------------------
    banner(log, "2. COMPARTMENT MAPPING")
    raw_ct = adata.obs[ct_col].astype(str)
    mapped = raw_ct.map(_canonical)
    tbl = pd.DataFrame({"original": raw_ct, "canonical": mapped})
    summary = (tbl.groupby(["original", "canonical"], dropna=False)
               .size().reset_index(name="n_cells")
               .sort_values("n_cells", ascending=False))
    for _, r in summary.iterrows():
        tgt = r["canonical"] if isinstance(r["canonical"], str) else "(excluded)"
        log.info(f"   {str(r['original'])[:34]:34s} -> {tgt:16s} "
                 f"{int(r['n_cells']):>7,}")
    unmapped = summary[summary["canonical"].isna()]["original"].tolist()
    if unmapped:
        log.info("")
        log.info(f"   {len(unmapped)} label(s) excluded (immune, unmatched or")
        log.info("   not one of the four analysed compartments). If a cancer or")
        log.info("   stromal label was excluded by mistake, pass the correct")
        log.info("   --celltype-col; do not let it pass silently.")
    summary.to_csv(cfg.paths.results / f"wp3r_{tag}_compartment_map.csv",
                   index=False)
    log.info("")

    adata.obs["_comp"] = mapped.values
    adata.obs["_donor"] = adata.obs[d_col].astype(str).values
    adata = adata[adata.obs["_comp"].notna()].copy()
    if adata.n_obs == 0:
        log.info("[X] no cells mapped to the analysed compartments")
        report["verdict"] = "BLOCKED"
        return report

    if er_only and s_col is not None:
        m = adata.obs[s_col].astype(str).str.contains(
            r"ER|luminal|lum", case=False, na=False)
        m &= ~adata.obs[s_col].astype(str).str.contains(
            r"HER2|TNBC|basal|triple", case=False, na=False)
        if m.any():
            adata = adata[m.values].copy()
            log.info(f"   ER+ subset: {adata.n_obs:,} cells, "
                     f"{adata.obs['_donor'].nunique()} donors")
        else:
            log.info("   [!] no ER+ cells matched; analysing all subtypes.")
            log.info("       The Wu primary analysis is ER+ only, so an")
            log.info("       all-subtype replication is a weaker comparison.")
            report["er_restriction_applied"] = False
    log.info("")

    # ---- score -----------------------------------------------------------
    banner(log, "3. SCORING (same panel, same statistic as WP8p)")
    panel = pd.read_csv(cfg.paths.results / "signature_panel.csv")
    if adata.raw is not None:
        adata = ad.AnnData(X=adata.raw.X.copy(), obs=adata.obs.copy(),
                           var=adata.raw.var.copy()
                           if adata.raw.var is not None else adata.var.copy())
    if "feature_name" in adata.var.columns:
        adata.var_names = pd.Index(adata.var["feature_name"].astype(str))
        adata.var_names_make_unique()

    Xc = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    dn = adata.obs["_donor"].astype(str).values
    cp = adata.obs["_comp"].astype(str).values
    libs, meta = [], []
    for d in sorted(set(dn)):
        for c in [CANCER] + STROMAL:
            m = (dn == d) & (cp == c)
            if int(m.sum()) < MIN_CELLS:
                continue
            libs.append(np.asarray(Xc[m].sum(axis=0)).ravel())
            meta.append({"donor_id": d, "compartment": c,
                         "n_cells": int(m.sum())})
    if not libs:
        log.info("[X] no donor x compartment library met the cell minimum")
        report["verdict"] = "BLOCKED"
        return report
    mat = pd.DataFrame(np.vstack(libs).T, index=adata.var_names)
    meta = pd.DataFrame(meta)
    cpm = mat.div(mat.sum(axis=0).replace(0, np.nan), axis=1) * 1e6
    lg = np.log1p(cpm)
    lg = lg.loc[lg.notna().all(axis=1)]
    ranks = lg.rank(axis=0, method="average") / float(lg.shape[0])
    log.info(f"   {mat.shape[1]} libraries, {meta['donor_id'].nunique()} donors")

    present = set(lg.index)
    rows = []
    for (sig, direction), grp in panel.groupby(["signature", "direction"]):
        genes = resolve_against(grp["gene"].astype(str).tolist(), present)
        if len(genes) < 5:
            continue
        s = ranks.loc[genes].mean(axis=0)
        for j, val in enumerate(s.values):
            rows.append({"donor_id": meta.iloc[j]["donor_id"],
                         "compartment": meta.iloc[j]["compartment"],
                         "signature": f"{sig}|{direction}",
                         "n_genes": len(genes), "score": float(val)})
    scores = pd.DataFrame(rows)
    log.info(f"   {scores['signature'].nunique()} signatures scored")
    log.info("")

    # ---- donor-level test ------------------------------------------------
    banner(log, "4. WITHIN-DONOR PAIRED TEST")
    rng = np.random.default_rng(cfg.seed)
    res_rows = []
    for sig in sorted(scores["signature"].unique()):
        for comp in STROMAL:
            r = _paired(scores, sig, comp)
            if r is None:
                continue
            lo, hi = _boot_ci(r["diffs"], n_boot, rng)
            res_rows.append({"signature": sig, "compartment": comp,
                             "n_donors": r["n_donors"],
                             "mean_diff": r["mean_diff"],
                             "ci_low": lo, "ci_high": hi,
                             "n_lower_in_cancer": r["n_lower_in_cancer"],
                             "wilcoxon_p": r["wilcoxon_p"]})
    res = pd.DataFrame(res_rows)
    if res.empty:
        log.info("[X] no donor-level comparison could be formed")
        report["verdict"] = "BLOCKED"
        return report
    res["p_adj"] = np.nan
    for comp, idx in res.groupby("compartment").groups.items():
        res.loc[idx, "p_adj"] = _bh(res.loc[idx, "wilcoxon_p"].values)
    res["consistent"] = res["n_lower_in_cancer"] == res["n_donors"]
    res.to_csv(cfg.paths.results / f"wp3r_{tag}_donor_level.csv", index=False)
    log.info(f"   {res['n_donors'].max()} donors contribute at most")
    report["n_donors"] = int(res["n_donors"].max())
    log.info("")

    # ---- compare with the discovery atlas -------------------------------
    banner(log, "5. REPLICATION AGAINST THE WU ATLAS")
    disc_path = cfg.paths.results / "wp8p_donor_level.csv"
    fallback = cfg.paths.results / "wp8_donor_level.csv"
    disc, src = None, None
    if disc_path.exists():
        disc, src = pd.read_csv(disc_path), "wp8p (pseudobulk)"
    elif fallback.exists():
        d = pd.read_csv(fallback)
        disc, src = d[d["cohort"] == "ER+"], "wp8 (AUCell mean)"
    if disc is None:
        log.info("   no discovery donor-level result on disk; run wp8 or wp8p")
        report["verdict"] = "REPLICATION_COMPUTED_NOT_COMPARED"
        return report

    log.info(f"   discovery source: {src}")
    j = res.merge(disc, on=["signature", "compartment"],
                  suffixes=("_rep", "_disc"))
    if j.empty:
        log.info("   no overlapping rows")
        report["verdict"] = "REPLICATION_COMPUTED_NOT_COMPARED"
        return report
    j["same_direction"] = (np.sign(j["mean_diff_rep"])
                           == np.sign(j["mean_diff_disc"]))
    j.to_csv(cfg.paths.results / f"wp3r_{tag}_vs_discovery.csv", index=False)

    disc_sig = j[j["p_adj_disc"] < 0.05]
    repl = disc_sig[disc_sig["same_direction"] & (disc_sig["p_adj_rep"] < 0.05)]
    dir_only = disc_sig[disc_sig["same_direction"] & (disc_sig["p_adj_rep"] >= 0.05)]
    failed = disc_sig[~disc_sig["same_direction"]]
    rho = float(stats.spearmanr(j["mean_diff_rep"], j["mean_diff_disc"])[0])

    log.info(f"   comparable rows                       : {len(j)}")
    log.info(f"   Spearman of mean differences          : {rho:+.3f}")
    log.info(f"   significant in discovery              : {len(disc_sig)}")
    log.info(f"   replicated (same sign AND q<0.05)     : {len(repl)}")
    log.info(f"   same direction but not significant    : {len(dir_only)}")
    log.info(f"   OPPOSITE direction                    : {len(failed)}")
    log.info("")
    if len(failed):
        log.info("   [!] reversed in the replication atlas:")
        for _, r in failed.iterrows():
            log.info(f"       {r['signature']} vs {r['compartment']}: "
                     f"discovery {r['mean_diff_disc']:+.4f}, "
                     f"replication {r['mean_diff_rep']:+.4f}")
        log.info("")

    trad = j[j["signature"] == "TRADITIOM_dormancy|up"]
    if len(trad):
        log.info("   TRADITIOM up:")
        for _, r in trad.iterrows():
            log.info(f"      vs {r['compartment']:12s} discovery "
                     f"{r['mean_diff_disc']:+.4f} (q={r['p_adj_disc']:.3g}), "
                     f"replication {r['mean_diff_rep']:+.4f} "
                     f"(q={r['p_adj_rep']:.3g})")
        log.info("")

    rate = len(repl) / max(len(disc_sig), 1)
    report["replication"] = {
        "source": src, "n_rows": int(len(j)), "spearman": rho,
        "n_discovery_significant": int(len(disc_sig)),
        "n_replicated": int(len(repl)),
        "n_same_direction_only": int(len(dir_only)),
        "n_opposite": int(len(failed)),
        "replication_rate": float(rate)}

    # ---- verdict ---------------------------------------------------------
    banner(log, "VERDICT")
    if len(failed) == 0 and rate >= 0.6:
        verdict = "REPLICATED"
        log.info("The donor-level compartment result reproduces in an")
        log.info("independent atlas with independent annotation. The n=9")
        log.info("objection is answered: report both cohorts, lead with the")
        log.info("direction consistency, and the manuscript's central claim no")
        log.info("longer rests on a single small donor set.")
    elif len(failed) == 0:
        verdict = "DIRECTION_REPLICATED_POWER_LIMITED"
        log.info("Every comparison points the same way in both atlases, but")
        log.info("not all reach significance in the replication. That is the")
        log.info("expected result for a smaller or noisier second cohort and")
        log.info("is still a genuine replication of DIRECTION. Report the")
        log.info("direction agreement and the per-donor points; do not claim")
        log.info("independent significance where there is none.")
    else:
        verdict = "REPLICATION_FAILED"
        log.info("[!] At least one comparison significant in the discovery")
        log.info("    atlas reverses in the replication. This must be reported.")
        log.info("    Check the compartment mapping above first - a mismapped")
        log.info("    label is the most likely cause and is fixable. If the")
        log.info("    mapping is right, the affected claim is atlas-specific")
        log.info("    and cannot be stated generally.")
    report["verdict"] = verdict
    log.info("")
    log.info(f"VERDICT: {verdict}")

    with open(cfg.paths.results / f"wp3r_{tag}_replication.json", "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {cfg.paths.results / f'wp3r_{tag}_donor_level.csv'}")
    return report
