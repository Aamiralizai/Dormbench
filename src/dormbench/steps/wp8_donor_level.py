"""WP8 - donor-level inference, ER-positive primary.

Every compartment AUROC reported so far is computed over cells. With 100,064
cells from 26 donors, cell-level intervals overstate precision by roughly the
number of cells per donor: the observations are not independent, and a single
stroma-rich biopsy can carry a result (Supplementary Note S2). Leave-one-donor
analysis measures that influence but does not remove it, because the estimator
is still a cell-level statistic.

This step makes the donor the unit of analysis:

  - pseudobulk each signature to one value per donor x compartment;
  - form within-donor paired differences (cancer minus each stromal
    compartment), which removes all between-donor variation by construction;
  - test with a paired Wilcoxon over donors and report the donor-level mean
    difference with a bootstrap confidence interval;
  - fit score ~ compartment * subtype + (1 | donor) where statsmodels is
    available, so the compartment effect is estimated with donor as a random
    intercept and the compartment-by-subtype interaction is testable;
  - repeat everything restricted to ER-positive donors, which is the primary
    analysis, with TNBC and HER2-positive donors as sensitivity analyses.

The atlas is pan-subtype, so a title and abstract that specify ER-positive
disease must rest on the ER-positive donors alone. That restriction is applied
here rather than assumed.

Multiplicity is controlled across signatures with Benjamini-Hochberg.

Input is results/wp2b_cell_scores.csv.gz, so the atlas is not re-loaded.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config
from ..logging_utils import banner, get_logger

CANCER = "Cancer Epithelial"
STROMAL = ["Endothelial", "CAFs", "PVL"]
# Signatures carried in the donor-level tables. Both proliferation controls are
# included: if the donor-level method is sound they must behave differently
# from the dormancy sets.
PRIMARY = ["TRADITIOM_dormancy|up", "PreAdapted|up", "EMT_hallmark|up",
           "Hypoxia_hallmark|up", "TGFbeta_hallmark|up",
           "E2F_control|up", "G2M_control|up"]


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values."""
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


def _boot_ci(x: np.ndarray, n_boot: int, rng, alpha: float = 0.05):
    """Percentile bootstrap CI for the mean of donor-level values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return (np.nan, np.nan)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def _hier_boot(cells: pd.DataFrame, score: str, comp_a: str, comp_b: str,
               n_boot: int, rng) -> tuple[float, float]:
    """Hierarchical bootstrap: resample donors, then cells within donor.

    A flat cell bootstrap treats 100,064 cells as 100,064 independent draws and
    returns an interval far too narrow. Resampling donors first propagates
    between-donor variance, which is the dominant term here.
    """
    donors = cells["donor_id"].unique()
    by_donor = {d: g for d, g in cells.groupby("donor_id", observed=True)}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(donors, size=donors.size, replace=True)
        diffs = []
        for d in pick:
            g = by_donor[d]
            a = g.loc[g["celltype_major"] == comp_a, score].values
            b = g.loc[g["celltype_major"] == comp_b, score].values
            if a.size < 3 or b.size < 3:
                continue
            ra = a[rng.integers(0, a.size, a.size)]
            rb = b[rng.integers(0, b.size, b.size)]
            diffs.append(ra.mean() - rb.mean())
        if diffs:
            out.append(float(np.mean(diffs)))
    if len(out) < 10:
        return (np.nan, np.nan)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def _paired(pseudo: pd.DataFrame, score: str, comp: str):
    """Within-donor cancer-minus-compartment difference across donors."""
    sub = pseudo[pseudo["signature"] == score]
    wide = sub.pivot_table(index="donor_id", columns="celltype_major",
                           values="mean_score")
    if CANCER not in wide.columns or comp not in wide.columns:
        return None
    pair = wide[[CANCER, comp]].dropna()
    if len(pair) < 4:
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


def _mixed(cells: pd.DataFrame, score: str, log):
    """score ~ compartment * subtype + (1 | donor).

    Returns None if statsmodels is unavailable; the paired donor-level test is
    the primary result and does not depend on it.
    """
    try:
        import statsmodels.formula.api as smf
    except Exception:  # noqa: BLE001
        return None
    d = cells[cells["celltype_major"].isin([CANCER] + STROMAL)].copy()
    if d["donor_id"].nunique() < 5:
        return None
    d = d.rename(columns={score: "y"})
    d["compartment"] = pd.Categorical(
        d["celltype_major"],
        categories=[CANCER] + [c for c in STROMAL if c in set(d["celltype_major"])])
    d["subtype"] = d["subtype"].astype(str)
    try:
        m = smf.mixedlm("y ~ C(compartment) * C(subtype)", d,
                        groups=d["donor_id"]).fit(method="lbfgs", maxiter=200)
    except Exception as e:  # noqa: BLE001
        log.info(f"      mixed model did not converge: {type(e).__name__}")
        return None
    terms = {}
    for name in m.params.index:
        if "compartment" in name:
            terms[name] = {"coef": float(m.params[name]),
                           "p": float(m.pvalues[name]),
                           "ci_low": float(m.conf_int().loc[name, 0]),
                           "ci_high": float(m.conf_int().loc[name, 1])}
    inter = [k for k in terms if ":" in k]
    return {"n_obs": int(m.nobs), "n_donors": int(d["donor_id"].nunique()),
            "terms": terms, "n_interaction_terms": len(inter),
            "any_interaction_sig": bool(
                any(terms[k]["p"] < 0.05 for k in inter))}


def run(cfg: Config, n_boot: int = 2000, method: str = "AUCell") -> dict:
    """Donor-level compartment inference, ER-positive primary."""
    log = get_logger("wp8", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    banner(log, "WP8 - DONOR-LEVEL INFERENCE (ER-POSITIVE PRIMARY)")
    rng = np.random.default_rng(cfg.seed)
    report: dict = {"step": "wp8", "generated": datetime.now().isoformat(),
                    "params": {"n_boot": n_boot, "method": method}}

    path = cfg.paths.results / "wp2b_cell_scores.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `dormbench wp2b` first.")
    cells = pd.read_csv(path)
    log.info(f"   {len(cells):,} cells x {cells.shape[1]} columns")

    score_cols = [c for c in cells.columns if c.startswith(f"{method}|")]
    if not score_cols:
        raise ValueError(f"no {method} score columns in {path.name}")
    ren = {c: c.split("|", 1)[1] for c in score_cols}
    cells = cells.rename(columns=ren)
    scores = [ren[c] for c in score_cols]

    cells["subtype"] = cells["subtype"].astype(str)
    log.info(f"   donors: {cells['donor_id'].nunique()}")
    st = cells.groupby("subtype")["donor_id"].nunique().to_dict()
    log.info(f"   donors by subtype: {st}")
    cnt = cells["subtype"].value_counts().to_dict()
    log.info(f"   cells by subtype: {cnt}")
    report["donors_total"] = int(cells["donor_id"].nunique())
    report["donors_by_subtype"] = {k: int(v) for k, v in st.items()}
    report["cells_by_subtype"] = {k: int(v) for k, v in cnt.items()}
    log.info("")

    # "HER2+" contains the substring "ER", so a contains() match silently
    # folds HER2-positive donors into the ER-positive cohort. Anchor the match.
    er_mask = cells["subtype"].str.match(r"^\s*ER[\s+-]*(pos|positive)?\s*$",
                                         case=False, na=False)
    if not er_mask.any():
        log.info("[!] no ER-positive donors found; check the subtype column")
    cohorts = {"ER+": cells[er_mask], "all": cells,
               "non-ER+": cells[~er_mask]}

    pseudo_rows, res_rows = [], []
    for cname, sub in cohorts.items():
        if sub.empty or sub["donor_id"].nunique() < 4:
            log.info(f"   [skip] {cname}: too few donors")
            continue
        # one value per donor x compartment x signature
        for sig in scores:
            g = (sub.groupby(["donor_id", "celltype_major", "subtype"],
                             observed=True)[sig]
                 .agg(["mean", "size"]).reset_index())
            g = g[g["size"] >= 10]          # a compartment needs cells to mean
            g = g.rename(columns={"mean": "mean_score", "size": "n_cells"})
            g["signature"] = sig
            g["cohort"] = cname
            pseudo_rows.append(g)
        pdf = pd.concat([r for r in pseudo_rows if r["cohort"].iloc[0] == cname])

        banner(log, f"{cname}  ({sub['donor_id'].nunique()} donors)")
        log.info(f"{'signature':26s} {'comp':12s} {'n':>3s} "
                 f"{'mean diff':>10s} {'lower/n':>8s} {'p':>9s}")
        for sig in scores:
            for comp in STROMAL:
                r = _paired(pdf, sig, comp)
                if r is None:
                    continue
                lo, hi = _boot_ci(r["diffs"], n_boot, rng)
                row = {"cohort": cname, "signature": sig, "compartment": comp,
                       "n_donors": r["n_donors"], "mean_diff": r["mean_diff"],
                       "median_diff": r["median_diff"],
                       "ci_low": lo, "ci_high": hi,
                       "n_lower_in_cancer": r["n_lower_in_cancer"],
                       "wilcoxon_p": r["wilcoxon_p"]}
                res_rows.append(row)
                if sig in PRIMARY:
                    log.info(f"{sig:26s} {comp:12s} {r['n_donors']:3d} "
                             f"{r['mean_diff']:+10.4f} "
                             f"{r['n_lower_in_cancer']:3d}/{r['n_donors']:<4d} "
                             f"{r['wilcoxon_p']:9.4g}")
        log.info("")

    res = pd.DataFrame(res_rows)
    if res.empty:
        raise RuntimeError("no donor-level comparisons could be formed")
    # BH across signatures within cohort x compartment
    res["p_adj"] = np.nan
    for (c, comp), idx in res.groupby(["cohort", "compartment"]).groups.items():
        res.loc[idx, "p_adj"] = _bh(res.loc[idx, "wilcoxon_p"].values)
    res["consistent"] = res["n_lower_in_cancer"] == res["n_donors"]
    res.to_csv(cfg.paths.results / "wp8_donor_level.csv", index=False)
    pd.concat(pseudo_rows).to_csv(
        cfg.paths.results / "wp8_pseudobulk.csv.gz", index=False)

    # hierarchical bootstrap for the primary signature, ER+ only
    banner(log, "HIERARCHICAL BOOTSTRAP (ER+, cancer vs endothelium)")
    log.info("Donors resampled first, then cells within donor, so between-")
    log.info("donor variance is carried into the interval.")
    log.info("")
    hb = {}
    er = cohorts["ER+"]
    for sig in PRIMARY:
        if sig not in scores or er.empty:
            continue
        lo, hi = _hier_boot(er, sig, CANCER, "Endothelial", min(n_boot, 500), rng)
        hb[sig] = {"ci_low": lo, "ci_high": hi}
        flag = "" if not np.isfinite(lo) else (
            "  excludes 0" if lo * hi > 0 else "  spans 0")
        log.info(f"   {sig:26s} 95% CI [{lo:+.4f}, {hi:+.4f}]{flag}")
    report["hierarchical_bootstrap_ER"] = hb
    log.info("")

    # mixed model with donor random intercept
    banner(log, "MIXED MODEL  score ~ compartment * subtype + (1 | donor)")
    mm = {}
    for sig in PRIMARY:
        if sig not in scores:
            continue
        m = _mixed(cells, sig, log)
        if m is None:
            continue
        mm[sig] = m
        main = {k: v for k, v in m["terms"].items() if ":" not in k}
        log.info(f"   {sig}")
        for k, v in list(main.items())[:3]:
            lab = k.replace("C(compartment)[T.", "").replace("]", "")
            log.info(f"      {lab:22s} {v['coef']:+.4f} "
                     f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]  p={v['p']:.3g}")
        log.info(f"      compartment x subtype interaction significant: "
                 f"{m['any_interaction_sig']}")
    if not mm:
        log.info("   statsmodels unavailable or models did not converge;")
        log.info("   the paired donor-level Wilcoxon above is the primary test.")
    report["mixed_models"] = mm
    log.info("")

    # verdict from the ER+ primary analysis
    banner(log, "VERDICT")
    erres = res[(res["cohort"] == "ER+") & (res["signature"].isin(PRIMARY))]
    dorm = erres[erres["signature"] == "TRADITIOM_dormancy|up"]
    n_consistent = int(dorm["consistent"].sum())
    n_sig = int((dorm["p_adj"] < 0.05).sum())
    report["ER_dormancy_consistent_compartments"] = n_consistent
    report["ER_dormancy_significant_compartments"] = n_sig
    for _, r in dorm.iterrows():
        log.info(f"   TRADITIOM up vs {r['compartment']:12s} "
                 f"{r['n_lower_in_cancer']}/{r['n_donors']} donors lower in "
                 f"cancer, mean {r['mean_diff']:+.4f} "
                 f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}], "
                 f"q={r['p_adj']:.3g}")
    ctrl = erres[erres["signature"].isin(["E2F_control|up", "G2M_control|up"])]
    n_ctrl = int((ctrl["p_adj"] < 0.05).sum())
    log.info("")
    log.info(f"   proliferation controls significant in {n_ctrl}/{len(ctrl)} "
             "comparisons")
    if n_sig >= 2 and n_ctrl < len(ctrl):
        verdict = "DONOR_LEVEL_CONFIRMED_ER"
        log.info("")
        log.info("The compartment preference survives donor-level inference in")
        log.info("ER-positive donors, with the donor as the unit of analysis")
        log.info("and multiplicity controlled. It is not an artefact of")
        log.info("treating cells as independent observations.")
    elif n_sig >= 1:
        verdict = "DONOR_LEVEL_PARTIAL"
        log.info("")
        log.info("The preference holds against some but not all stromal")
        log.info("compartments once donors are the unit of analysis. Report")
        log.info("the donor-level result, not the cell-level AUROC, as primary.")
    else:
        verdict = "DONOR_LEVEL_NOT_SUPPORTED"
        log.info("")
        log.info("[!] The cell-level result does not survive donor-level")
        log.info("    inference. The cell-level AUROC must not be reported as")
        log.info("    the primary statistic.")
    report["verdict"] = verdict
    log.info("")

    with open(cfg.paths.results / "wp8_donor_level.json", "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    # one-point-per-donor figure for the primary signature, ER+ cohort
    try:
        from ..viz.donor import plot_donor_level
        er_pseudo = pd.concat(pseudo_rows)
        er_pseudo = er_pseudo[er_pseudo["cohort"] == "ER+"]
        fig = plot_donor_level(
            er_pseudo, "TRADITIOM_dormancy|up",
            ["Endothelial", "CAFs", "PVL"],
            cfg.paths.figures / "fig12_donor_level.svg")
        if fig:
            log.info(f"Written: {fig}")
            report["figure"] = str(fig)
    except Exception as e:  # noqa: BLE001
        log.info(f"   donor-level figure skipped: {type(e).__name__}: {e}")

    log.info(f"Written: {cfg.paths.results / 'wp8_donor_level.csv'}")
    log.info(f"Written: {cfg.paths.results / 'wp8_pseudobulk.csv.gz'}")
    return report
