"""WP8p - TRUE donor x compartment pseudobulk, scored as a biological replicate.

Review point 12. WP8 makes the donor the unit of inference, which closes the
pseudoreplication objection about degrees of freedom. It does not close a
second, narrower objection: WP8 averages per-cell AUCell scores. AUCell is a
nonlinear rank statistic computed within a cell, so the mean of AUCell scores
over a donor's cells is not the AUCell score of that donor's pseudobulk. A
specialist can reasonably ask why the counts were not summed first and the
resulting biological replicate scored directly.

This step does that:

  - sum RAW counts within each donor x compartment, giving one library per
    biological replicate rather than per cell;
  - CPM-normalise and log1p each pseudobulk library;
  - score every signature on those libraries with a rank-based statistic;
  - run the identical within-donor paired Wilcoxon that WP8 runs;
  - compare the two estimates signature by signature.

Agreement is the result that matters. If the donor-level conclusion holds when
the measurement never passes through a cell-level nonlinearity, the objection
is answered. Disagreement would be equally informative and must be reported,
not buried: it would mean the cell-level statistic, not the biology, carries
the WP8 result.

Scoring note. AUCell on 26 x 9 pseudobulk libraries is not comparable to AUCell
on 100,064 cells: the ranking is over a much denser library. The statistic used
here is the mean within-library expression rank of the signature genes, scaled
to [0, 1], which is the pseudobulk analogue of a rank-based enrichment score and
is computed identically for the real signature and for everything it is compared
against. Absolute values therefore differ from WP8 by construction; the
comparison of interest is the SIGN and the CONSISTENCY of the within-donor
difference, not the raw magnitude.

Why more than one statistic. The within-library rank is relative to whatever
dominates the library: a cancer-epithelial library is topped by epithelial genes
and a CAF library by collagens, so a broadly expressed signature takes a
middling rank in both and its difference can flip sign. Section 3b therefore
scores the same libraries with abundance-relative statistics as well
(gene-wise z, log-CPM relative to the library mean, and z over the signature's
most variable members), and the verdict turns on whether the primary signature
keeps its direction across them. Comparing the panel against the EMT and
E2F/G2M controls, whose compartment direction is not in dispute, shows which
statistics are measuring compartment content at all.

Input is the atlas (for counts) and results/signature_panel.csv.
Output is results/wp8p_pseudobulk_scores.csv, results/wp8p_donor_level.csv,
results/wp8p_statistic_sensitivity.csv and results/wp8p_pseudobulk.json.
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
from .wp8_donor_level import CANCER, STROMAL, _bh, _boot_ci

MIN_CELLS = 10          # a pseudobulk library needs cells behind it
MIN_DONORS = 4          # a paired test needs donors
# The manuscript's primary claim. The verdict turns on THIS signature, because
# a panel-wide sign-agreement count mixes dormancy sets with their controls and
# cannot answer 'does the donor-level result survive'.
PRIMARY_SIG = "TRADITIOM_dormancy|up"


def _rank_score(mat: pd.DataFrame, genes: list[str]) -> pd.Series:
    """Mean within-library expression rank of `genes`, scaled to [0, 1].

    Ranks are computed per library (column), so library size and per-library
    scaling cannot move the score - the pseudobulk analogue of AUCell's
    rank-based design. Returns one value per library.
    """
    idx = [g for g in genes if g in mat.index]
    if len(idx) < 5:
        return pd.Series(np.nan, index=mat.columns)
    ranks = mat.rank(axis=0, method="average", na_option="keep")
    n = float(mat.shape[0])
    return (ranks.loc[idx].mean(axis=0) / n).astype(float)


def _score_z(lg: pd.DataFrame, genes: list[str]) -> pd.Series:
    """Gene-wise z across libraries, then mean over the signature.

    This is the scale a bulk signature score normally lives on: 'are these
    genes relatively more expressed in this library than in the others'.
    """
    idx = [g for g in genes if g in lg.index]
    sub = lg.loc[idx]
    z = sub.sub(sub.mean(axis=1), axis=0).div(
        sub.std(axis=1).replace(0, np.nan), axis=0)
    return z.mean(axis=0)


def _score_rel_expr(lg: pd.DataFrame, genes: list[str]) -> pd.Series:
    """Mean log-CPM of the signature minus the library-wide mean."""
    idx = [g for g in genes if g in lg.index]
    return lg.loc[idx].mean(axis=0) - lg.mean(axis=0)


def _score_top_variable(lg: pd.DataFrame, genes: list[str],
                        top: int = 50) -> pd.Series:
    """Gene-wise z restricted to the signature's most variable members.

    A 821-gene set averages away most of its own signal; these signatures are
    effectively used as their leading edge, so the top members are scored.
    """
    idx = [g for g in genes if g in lg.index]
    sub = lg.loc[idx]
    keep = sub.std(axis=1).sort_values(ascending=False).head(
        min(top, sub.shape[0])).index
    sub = sub.loc[keep]
    z = sub.sub(sub.mean(axis=1), axis=0).div(
        sub.std(axis=1).replace(0, np.nan), axis=0)
    return z.mean(axis=0)


# Scoring functions compared in section 3b. The first is the step's original
# statistic; the others are abundance/standardised scores that do not depend on
# what else dominates the library.
SCORERS = {
    "rank": _rank_score,
    "z": _score_z,
    "rel_expr": _score_rel_expr,
    "top_variable": _score_top_variable,
}


def _paired(scores: pd.DataFrame, sig: str, comp: str):
    """Within-donor cancer-minus-compartment difference, across donors."""
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
            "median_diff": float(np.median(diff)),
            "n_lower_in_cancer": int((diff < 0).sum()),
            "wilcoxon_p": float(p), "diffs": diff}


def run(cfg: Config, n_boot: int = 2000, er_only: bool = True,
        atlas_dir=None, tag: str | None = None) -> dict:
    log = get_logger("wp8p", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    stem = f"wp8p_{tag}" if tag else "wp8p"
    report: dict = {"step": "wp8p",
                    "generated": datetime.now().isoformat(timespec="seconds"),
                    "params": {"n_boot": n_boot, "er_only": er_only,
                               "tag": tag}}

    banner(log, "WP8p - TRUE DONOR x COMPARTMENT RAW-COUNT PSEUDOBULK")
    log.info("WP8 averages per-cell AUCell scores. This sums raw counts per")
    log.info("donor x compartment first, then scores the resulting biological")
    log.info("replicate. Agreement between the two answers the objection that")
    log.info("donor inference still rests on a cell-level nonlinear statistic.")
    log.info("")

    import anndata as ad
    import scanpy as sc
    from scipy import sparse

    panel_path = cfg.paths.results / "signature_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"{panel_path} not found. Run `dormbench wp2a`"
                                " and `dormbench wp2ext` first.")
    panel = pd.read_csv(panel_path)

    path = find_atlas_h5ad(atlas_dir or cfg.paths.atlas_dir)
    log.info(f"Loading {path.name} ...")
    adata = ad.read_h5ad(path)
    # RAW counts are required: this step exists to avoid per-cell
    # normalisation, so a pre-normalised matrix would defeat the point.
    if adata.raw is not None:
        log.info("   using .raw for counts")
        adata = ad.AnnData(X=adata.raw.X.copy(), obs=adata.obs.copy(),
                           var=adata.raw.var.copy()
                           if adata.raw.var is not None else adata.var.copy())
    if "feature_name" in adata.var.columns:
        adata.var_names = pd.Index(adata.var["feature_name"].astype(str))
        adata.var_names_make_unique()
    log.info(f"   {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    X = adata.X
    frac = float(np.asarray(
        abs(X[:min(2000, X.shape[0])].data % 1).sum()
        if sparse.issparse(X) else np.abs(X[:2000] % 1).sum()))
    if frac > 0:
        log.info("   [!] matrix is not integer-valued; it appears already")
        log.info("       normalised. Summing it is still a library-level")
        log.info("       aggregate but is no longer a raw-count pseudobulk.")
        report["counts_were_integer"] = False
    else:
        report["counts_were_integer"] = True

    obs = adata.obs
    for col in ("donor_id", "celltype_major"):
        if col not in obs.columns:
            raise ValueError(f"atlas lacks required obs column '{col}'")

    if er_only and "subtype" in obs.columns:
        keep = obs["subtype"].astype(str).str.match(
            r"^\s*ER[\s+-]*(pos|positive)?\s*$", case=False, na=False)
        adata = adata[keep.values].copy()
        obs = adata.obs
        log.info(f"   ER+ subset: {adata.n_obs:,} cells, "
                 f"{obs['donor_id'].nunique()} donors")
    log.info("")

    # ---- aggregate raw counts per donor x compartment --------------------
    banner(log, "1. SUMMING RAW COUNTS PER DONOR x COMPARTMENT")
    comps = [CANCER] + STROMAL
    donors = sorted(obs["donor_id"].astype(str).unique())
    ct = obs["celltype_major"].astype(str).values
    dn = obs["donor_id"].astype(str).values

    Xc = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    libs, meta = [], []
    for d in donors:
        for c in comps:
            m = (dn == d) & (ct == c)
            n = int(m.sum())
            if n < MIN_CELLS:
                continue
            libs.append(np.asarray(Xc[m].sum(axis=0)).ravel())
            meta.append({"donor_id": d, "compartment": c, "n_cells": n})
    if not libs:
        raise RuntimeError("no donor x compartment library met the cell minimum")

    mat = pd.DataFrame(np.vstack(libs).T, index=adata.var_names,
                       columns=[f"{m['donor_id']}|{m['compartment']}"
                                for m in meta])
    meta = pd.DataFrame(meta)
    log.info(f"   {mat.shape[1]} libraries from {meta['donor_id'].nunique()} "
             f"donors x {meta['compartment'].nunique()} compartments")
    log.info(f"   median cells per library: {int(meta['n_cells'].median())}")
    log.info(f"   total counts per library: median "
             f"{float(mat.sum(axis=0).median()):,.0f}")
    log.info("")

    # CPM + log1p per library. Rank scoring is invariant to this, but it keeps
    # the matrix on the same footing as every other bulk step in the package.
    cpm = mat.div(mat.sum(axis=0).replace(0, np.nan), axis=1) * 1e6
    lg = np.log1p(cpm)
    lg = lg.loc[lg.notna().all(axis=1)]
    log.info(f"   {lg.shape[0]:,} genes retained after normalisation")
    log.info("")

    # ---- score -----------------------------------------------------------
    banner(log, "2. SCORING PSEUDOBULK LIBRARIES")
    present = set(lg.index)
    rows = []
    for (sig, direction), grp in panel.groupby(["signature", "direction"]):
        genes = resolve_against(grp["gene"].astype(str).tolist(), present)
        if len(genes) < 5:
            continue
        s = _rank_score(lg, genes)
        name = f"{sig}|{direction}"
        for lib, val in s.items():
            d, c = lib.split("|", 1)
            rows.append({"donor_id": d, "compartment": c, "signature": name,
                         "n_genes": len(genes), "score": float(val)})
    scores = pd.DataFrame(rows)
    if scores.empty:
        raise RuntimeError("no signature could be scored on the pseudobulk")
    scores = scores.merge(meta, on=["donor_id", "compartment"], how="left")
    scores.to_csv(cfg.paths.results / f"{stem}_pseudobulk_scores.csv",
                  index=False)
    log.info(f"   {scores['signature'].nunique()} signatures scored")
    log.info("")

    # ---- paired donor-level test ----------------------------------------
    banner(log, "3. WITHIN-DONOR PAIRED TEST (identical to WP8)")
    rng = np.random.default_rng(cfg.seed)
    res_rows = []
    for sig in sorted(scores["signature"].unique()):
        for comp in STROMAL:
            r = _paired(scores, sig, comp)
            if r is None:
                continue
            lo, hi = _boot_ci(r["diffs"], n_boot, rng)
            res_rows.append({
                "signature": sig, "compartment": comp,
                "n_donors": r["n_donors"], "mean_diff": r["mean_diff"],
                "median_diff": r["median_diff"], "ci_low": lo, "ci_high": hi,
                "n_lower_in_cancer": r["n_lower_in_cancer"],
                "wilcoxon_p": r["wilcoxon_p"]})
    res = pd.DataFrame(res_rows)
    if res.empty:
        raise RuntimeError("no donor-level comparison could be formed")
    res["p_adj"] = np.nan
    for comp, idx in res.groupby("compartment").groups.items():
        res.loc[idx, "p_adj"] = _bh(res.loc[idx, "wilcoxon_p"].values)
    res["consistent"] = res["n_lower_in_cancer"] == res["n_donors"]
    res.to_csv(cfg.paths.results / f"{stem}_donor_level.csv", index=False)

    log.info(f"{'signature':28s} {'comp':12s} {'n':>3s} {'mean diff':>10s} "
             f"{'lower/n':>9s} {'q':>9s}")
    for _, r in res.iterrows():
        log.info(f"{r['signature']:28s} {r['compartment']:12s} "
                 f"{int(r['n_donors']):3d} {r['mean_diff']:+10.4f} "
                 f"{int(r['n_lower_in_cancer']):3d}/{int(r['n_donors']):<5d} "
                 f"{r['p_adj']:9.4g}")
    log.info("")

    # ---- is the disagreement about the biology or about the score? ------
    banner(log, "3b. DOES THE ANSWER DEPEND ON HOW THE LIBRARY IS SCORED?")
    log.info("The rank statistic above is relative to whatever else dominates")
    log.info("the library. A cancer-epithelial library is topped by epithelial")
    log.info("genes and a CAF library by collagens, so a broadly-expressed")
    log.info("signature gets a middling rank in both and the difference can")
    log.info("flip sign. Re-scoring the SAME libraries with abundance-relative")
    log.info("statistics separates that artefact from the biology.")
    log.info("")

    sig_genes = {}
    for (sig, direction), grp in panel.groupby(["signature", "direction"]):
        genes = resolve_against(grp["gene"].astype(str).tolist(), present)
        if len(genes) >= 5:
            sig_genes[f"{sig}|{direction}"] = genes

    sens_rows = []
    for stat_name, fn in SCORERS.items():
        rows = []
        for name, genes in sig_genes.items():
            s = fn(lg, genes)
            for lib, val in s.items():
                d, c = lib.split("|", 1)
                rows.append({"donor_id": d, "compartment": c,
                             "signature": name, "score": float(val)})
        if not rows:
            continue
        sc = pd.DataFrame(rows)
        for sig in sorted(sc["signature"].unique()):
            for comp in STROMAL:
                r = _paired(sc, sig, comp)
                if r is None:
                    continue
                sens_rows.append({
                    "statistic": stat_name, "signature": sig,
                    "compartment": comp, "n_donors": r["n_donors"],
                    "mean_diff": r["mean_diff"],
                    "median_diff": r["median_diff"],
                    "n_lower_in_cancer": r["n_lower_in_cancer"],
                    "wilcoxon_p": r["wilcoxon_p"],
                    "direction": ("stromal" if r["mean_diff"] < 0
                                  else "cancer")})
    sens = pd.DataFrame(sens_rows)
    if sens.empty:
        raise RuntimeError("no statistic could be scored on the pseudobulk")
    sens["p_adj"] = np.nan
    for (stat_name, comp), idx in sens.groupby(["statistic",
                                                "compartment"]).groups.items():
        sens.loc[idx, "p_adj"] = _bh(sens.loc[idx, "wilcoxon_p"].values)
    sens.to_csv(cfg.paths.results / f"{stem}_statistic_sensitivity.csv",
                index=False)

    log.info(f"{'statistic':13s} {'comp':12s} {'n':>3s} {'mean diff':>10s} "
             f"{'lower/n':>9s} {'q':>9s} {'direction':>9s}")
    prim = sens[sens["signature"] == PRIMARY_SIG]
    for _, r in prim.iterrows():
        log.info(f"{r['statistic']:13s} {r['compartment']:12s} "
                 f"{int(r['n_donors']):3d} {r['mean_diff']:+10.4f} "
                 f"{int(r['n_lower_in_cancer']):3d}/{int(r['n_donors']):<5d} "
                 f"{r['p_adj']:9.4g} {r['direction']:>9s}")
    log.info("")

    # Controls are what make the grid readable: if EMT is not stromal and
    # E2F/G2M are not cancer-higher under a statistic, that statistic is not
    # measuring compartment content and must not be used to argue anything.
    log.info("Controls, mean cancer-minus-stroma difference per statistic:")
    for ctrl, expect in (("EMT_hallmark|up", "stromal"),
                         ("E2F_control|up", "cancer"),
                         ("G2M_control|up", "cancer")):
        sub = sens[sens["signature"] == ctrl]
        if sub.empty:
            continue
        got = ", ".join(f"{s}={g['mean_diff'].mean():+.3f}"
                        for s, g in sub.groupby("statistic"))
        bad = sorted({g["statistic"].iloc[0] for _, g in sub.groupby("statistic")
                      if (g["mean_diff"].mean() < 0) != (expect == "stromal")})
        note = "" if not bad else f"   [!] wrong sign under: {', '.join(bad)}"
        log.info(f"   {ctrl:20s} expect {expect:8s} {got}{note}")
    log.info("")

    per_stat = (prim.groupby("statistic")["mean_diff"].mean()
                .sort_values())
    agree_stats = [s for s, v in per_stat.items() if v < 0]
    log.info(f"TRADITIOM-up is stromal under {len(agree_stats)}/"
             f"{len(per_stat)} statistics: "
             f"{', '.join(agree_stats) if agree_stats else 'none'}")
    report["statistic_sensitivity"] = {
        "per_statistic_mean_diff": {k: float(v)
                                    for k, v in per_stat.items()},
        "n_stromal": len(agree_stats),
        "n_statistics": int(len(per_stat)),
        "stromal_statistics": agree_stats,
    }
    log.info("")

    # ---- agreement with WP8 ---------------------------------------------
    banner(log, "4. AGREEMENT WITH THE AUCell-MEAN DONOR ANALYSIS (WP8)")
    wp8_path = cfg.paths.results / "wp8_donor_level.csv"
    agree = None
    if not wp8_path.exists():
        log.info("   wp8_donor_level.csv not found; run `dormbench wp8` to")
        log.info("   enable the comparison. The pseudobulk result above stands")
        log.info("   on its own but cannot be cross-checked.")
    else:
        w8 = pd.read_csv(wp8_path)
        w8 = w8[w8["cohort"] == ("ER+" if er_only else "all")]
        j = res.merge(w8, on=["signature", "compartment"],
                      suffixes=("_pseudobulk", "_aucell"))
        if j.empty:
            log.info("   no overlapping signature x compartment rows")
        else:
            j["same_direction"] = (np.sign(j["mean_diff_pseudobulk"])
                                   == np.sign(j["mean_diff_aucell"]))
            ok = j["same_direction"].mean()
            rho = float(stats.spearmanr(j["mean_diff_pseudobulk"],
                                        j["mean_diff_aucell"])[0])
            log.info(f"   {len(j)} comparable rows")
            log.info(f"   same sign of the within-donor difference: "
                     f"{int(j['same_direction'].sum())}/{len(j)} "
                     f"({100*ok:.0f}%)")
            log.info(f"   Spearman of the two mean differences: {rho:+.3f}")
            log.info("")
            both = j[(j["p_adj_pseudobulk"] < 0.05)
                     & (j["p_adj_aucell"] < 0.05)]
            only_a = j[(j["p_adj_pseudobulk"] >= 0.05)
                       & (j["p_adj_aucell"] < 0.05)]
            only_p = j[(j["p_adj_pseudobulk"] < 0.05)
                       & (j["p_adj_aucell"] >= 0.05)]
            log.info(f"   significant in both (q<0.05)      : {len(both)}")
            log.info(f"   significant only in AUCell-mean   : {len(only_a)}")
            log.info(f"   significant only in pseudobulk    : {len(only_p)}")
            if len(only_a):
                log.info("   [!] rows that do NOT survive true pseudobulk:")
                for _, r in only_a.iterrows():
                    log.info(f"       {r['signature']} vs {r['compartment']}")
            j.to_csv(cfg.paths.results / f"{stem}_vs_wp8.csv", index=False)
            agree = {"n_rows": int(len(j)),
                     "frac_same_direction": float(ok),
                     "spearman_mean_diff": rho,
                     "n_sig_both": int(len(both)),
                     "n_sig_only_aucell": int(len(only_a)),
                     "n_sig_only_pseudobulk": int(len(only_p))}
            report["agreement"] = agree
    log.info("")

    # ---- verdict ---------------------------------------------------------
    banner(log, "VERDICT")
    n_stromal = len(agree_stats)
    n_stats = int(len(per_stat))
    if agree is None:
        verdict = "PSEUDOBULK_COMPUTED_NOT_COMPARED"
        log.info("Pseudobulk donor-level results computed. Run wp8 to compare.")
    elif n_stromal == n_stats:
        verdict = "PSEUDOBULK_CONFIRMS_WP8"
        log.info("TRADITIOM-up scores lower in cancer epithelium than in stroma")
        log.info("under EVERY statistic tried, so the direction does not depend")
        log.info("on how the pseudobulk library is scored. The donor-level")
        log.info("result does not rest on a cell-level nonlinearity, and the")
        log.info("manuscript can say so explicitly.")
    elif n_stromal == 0:
        verdict = "PSEUDOBULK_DISAGREES_WITH_WP8"
        log.info("[!] No pseudobulk statistic reproduces the AUCell-mean donor")
        log.info("    direction. That is a finding about the measurement, not a")
        log.info("    bug to tune away: the donor-level claim would rest on the")
        log.info("    cell-level statistic. Do not report the WP8 numbers as")
        log.info("    donor pseudobulk.")
    else:
        verdict = "PSEUDOBULK_STATISTIC_DEPENDENT"
        log.info(f"TRADITIOM-up is stromal under {n_stromal}/{n_stats} scoring")
        log.info("statistics and cancer-higher under the rest. The disagreement")
        log.info("between this step and WP8 is therefore about the score, not")
        log.info("established biology: the within-library rank is sensitive to")
        log.info("which programme dominates each library.")
        log.info("")
        log.info("What to write:")
        log.info("  - report WP8 as the primary donor-level result and cite the")
        log.info("    abundance-relative pseudobulk scores as an independent")
        log.info("    direction check where they agree;")
        log.info("  - never describe the WP8 numbers as donor pseudobulk;")
        log.info("  - do not claim the CAF/PVL comparisons replicate unless they")
        log.info("    are significant under the abundance-relative statistics.")
        if agree is not None:
            log.info("")
            log.info(f"   panel-wide sign agreement with WP8: "
                     f"{agree['frac_same_direction']:.0%} over "
                     f"{agree['n_rows']} rows (context, not the verdict)")
    report["verdict"] = verdict
    report["n_statistics"] = n_stats
    report["n_statistics_stromal"] = n_stromal
    log.info("")
    log.info(f"VERDICT: {verdict}")

    with open(cfg.paths.results / f"{stem}_pseudobulk.json", "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {cfg.paths.results / f'{stem}_donor_level.csv'}")
    return report
