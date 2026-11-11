"""WP4s - does a dormancy score add anything to CTS5 for late recurrence?

The project plan's C3/C4 claims were never tested: the manuscript says so in its
own limitations ("the study did not evaluate late-recurrence or survival
endpoints directly"). This step tests them.

Design
    Cohort      ER-positive primary tumours with recurrence-free survival.
                METABRIC primary, TCGA replication (reduced covariates: TCGA
                carries no grade or tumour size in the pan-can atlas tables, so
                CTS5 itself cannot be computed there).
    Landmark    Patients disease-free at 5 years. Time is measured FROM year 5,
                so a recurrence in year 2 cannot leak into a "late recurrence"
                model. This is the population the clinical question is about.
    Incumbent   CTS5, computed exactly as published:
                    CTS5 = 0.438*nodes + 0.988*(0.093*size + 0.001*size^2
                                               + 0.375*grade + 0.017*age)
                with nodes in five ordinal categories (0, 1, 2-3, 4-9, >9),
                size in mm capped at 30, grade 1-3, age in years. Risk groups
                use the published cut-offs 3.13 and 3.86.
    Models      M0 CTS5
                M1 CTS5 + dormancy (TRADITIOM-up, mean gene-wise z)
                M2 CTS5 + compartment-adjusted dormancy (residual on the
                   stromal marker score)
    Inference   Likelihood-ratio test, Harrell C, decile calibration at the
                horizon, IPCW cumulative/dynamic AUC.
    PH          Schoenfeld residuals, plus the plan's 0-5 vs 5-20 split: the
                same score fitted in the early window and in the late window
                (left-truncated at 5 years). If the hazard ratio differs across
                windows, proportional hazards fails and a single HR must not be
                reported.

Reading the verdict
    DORMANCY_ADDS_TO_CTS5            dormancy improves fit over CTS5 alone
    DORMANCY_ADDS_NOTHING            it does not
    DORMANCY_LATE_ONLY               it adds only in the years 5-20 window
    PH_VIOLATED                      the dormancy HR is not constant over time,
                                     so no single HR may be quoted regardless
    INSUFFICIENT_EVENTS              too few late events to decide
    CTS5_NOT_COMPUTABLE              required clinicopathological fields absent

A null result here is a publishable finding and is reported as such: it means
dormancy state at diagnosis, as measured by these signatures, is not prognostic
for late recurrence once clinicopathological risk is accounted for.

Input is the bulk expression matrix and clinical tables. Output is
results/wp4s_*.csv, results/wp4s_survival.json and figures/fig14_survival.svg.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import banner, get_logger

# CTS5 node categories: 0, 1, 2-3, 4-9, >9 positive nodes
def _node_category(n: float) -> float:
    """Map a positive-node count onto the five published ordinal categories.

    Boundaries are strict-less-than so that a fractional count cannot fall
    through to the wrong category: a value of 0.1 is zero involved nodes, not
    two-to-three.
    """
    if not np.isfinite(n):
        return np.nan
    if n < 1:
        return 0.0
    if n < 2:
        return 1.0
    if n < 4:
        return 2.0
    if n < 10:
        return 3.0
    return 4.0


def cts5_score(nodes: float, size_mm: float, grade: float, age: float) -> float:
    """CTS5 as published (Dowsett 2018), size capped at 30 mm.

    The quadratic size term is NEGATIVE. Two secondary sources disagree on this
    sign: one restates the formula as `+ 0.001 x size^2`. The negative sign is
    adopted here because it reproduces the published risk-group means of
    Wimmer et al. 2024 (Table 1): their intermediate group (nodes 0.4,
    size 21 mm, grade 2.1, age 64.6) scores 3.36 with the minus sign, which is
    intermediate as that paper reports, but 4.23 with the plus sign, which
    would fall in their "high" group. Using the plus sign inflates the score
    and, on METABRIC, pushes ~73% of patients into "high" against a published
    ~24%.
    """
    if not all(np.isfinite([nodes, size_mm, grade, age])):
        return np.nan
    s = min(float(size_mm), 30.0)
    return (0.438 * _node_category(nodes)
            + 0.988 * (0.093 * s - 0.001 * s * s + 0.375 * grade
                       + 0.017 * age))


def cts5_risk_group(score: float) -> str:
    """Published cut-offs for 5-10 year distant recurrence risk."""
    if not np.isfinite(score):
        return "unknown"
    if score < 3.13:
        return "low"
    if score <= 3.86:
        return "intermediate"
    return "high"


# --------------------------------------------------------------------------
# survival utilities (no sksurv in this environment)
# --------------------------------------------------------------------------
def _km(times: np.ndarray, events: np.ndarray, t: float) -> float:
    """Kaplan-Meier survival at t."""
    order = np.argsort(times)
    times, events = times[order], events[order]
    surv, n = 1.0, len(times)
    for i, ti in enumerate(times):
        if ti > t:
            break
        at_risk = n - i
        if events[i] and at_risk > 0:
            surv *= (1 - 1.0 / at_risk)
    return surv


def _ipcw_auc(risk: np.ndarray, times: np.ndarray, events: np.ndarray,
              horizon: float) -> float:
    """Cumulative/dynamic AUC at a horizon, IPCW-weighted (Uno-style).

    Cases: event at or before the horizon. Controls: still event-free after it.
    Weights are 1/G(t) with G the Kaplan-Meier curve of the censoring
    distribution, so censoring before the horizon does not bias the estimate.
    """
    risk = np.asarray(risk, float)
    times = np.asarray(times, float)
    events = np.asarray(events, float)
    n = len(risk)
    if n < 10:
        return np.nan
    cens_t = np.where(events == 1, times, times)
    cens_e = (events == 0).astype(float)
    cases = np.where((times <= horizon) & (events == 1))[0]
    controls = np.where(times > horizon)[0]
    if len(cases) < 5 or len(controls) < 5:
        return np.nan
    num = 0.0
    den = 0.0
    for i in cases:
        g = _km(cens_t, cens_e, times[i])
        if g <= 0:
            continue
        w = 1.0 / g
        gt = risk[i] > risk[controls]
        eq = risk[i] == risk[controls]
        num += w * (gt.sum() + 0.5 * eq.sum())
        den += w * len(controls)
    return float(num / den) if den > 0 else np.nan


def _calibration(pred_risk: np.ndarray, times: np.ndarray, events: np.ndarray,
                 horizon: float, n_bins: int = 5) -> pd.DataFrame:
    """Observed (KM) vs predicted risk by quantile of predicted risk."""
    d = pd.DataFrame({"pred": pred_risk, "t": times, "e": events})
    d = d[np.isfinite(d["pred"])]
    if len(d) < n_bins * 5:
        return pd.DataFrame()
    d["bin"] = pd.qcut(d["pred"], n_bins, labels=False, duplicates="drop")
    rows = []
    for b, g in d.groupby("bin"):
        obs = 1.0 - _km(g["t"].values, g["e"].values, horizon)
        rows.append({"bin": int(b), "n": int(len(g)),
                     "predicted": float(g["pred"].mean()),
                     "observed": float(obs)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def _purity_controls(cfg: Config, cohort: str) -> pd.DataFrame | None:
    """Independent composition estimates: ESTIMATE stromal and MCP fibroblast.

    These are the reference implementations (scripts/run_official_purity.R), so
    they are not derived from this study's marker logic. Controlling for them is
    a stronger test than controlling for the internal stromal marker score,
    which shares the expression matrix and the marker-selection rule with the
    dormancy score itself.
    """
    path = cfg.paths.results / f"purity_official_{cohort}.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    if "sample" not in d.columns:
        return None
    d = d.set_index("sample")
    out = pd.DataFrame(index=d.index)
    if "StromalScore" in d.columns:
        out["estimate_stromal"] = pd.to_numeric(d["StromalScore"],
                                                errors="coerce")
    if "Fibroblasts" in d.columns:
        out["mcp_fibroblast"] = pd.to_numeric(d["Fibroblasts"],
                                              errors="coerce")
    return out if out.shape[1] else None


def _min_ph_p(ph: dict, keys: tuple | None = None) -> float:
    """Minimum Schoenfeld p, optionally restricted to named covariates."""
    vals = [v.get("p", 1.0)
            for model in ph.values()
            for k, v in model.items()
            if isinstance(v, dict) and (keys is None or k in keys)]
    return min(vals, default=1.0)


def _log_conditional(log, models: pd.DataFrame) -> None:
    """Print the dormancy hazard ratio from every model that contains it.

    This is the direct answer to 'what happens when composition is controlled
    for rather than removed from the score': each line holds the same exposure
    and a different adjustment set.
    """
    rows = [(r["model"], r.get("hr_dormancy_z"), r.get("p_dormancy_z"))
            for _, r in models.iterrows()
            if np.isfinite(r.get("hr_dormancy_z", np.nan))]
    if not rows:
        return
    log.info("")
    log.info("   Dormancy hazard ratio, by adjustment set:")
    for name, hr, p in rows:
        log.info(f"      {name:36s} HR {hr:5.2f}   p = {p:.3g}")


def _zscore_score(expr: pd.DataFrame, genes: list[str]) -> pd.Series:
    """Mean gene-wise z across samples: the package's bulk scoring rule."""
    present = [g for g in genes if g in expr.index]
    if len(present) < 5:
        return pd.Series(np.nan, index=expr.columns)
    sub = expr.loc[present].astype(float)
    mu = sub.mean(axis=1)
    sd = sub.std(axis=1).replace(0, np.nan)
    return sub.sub(mu, axis=0).div(sd, axis=0).mean(axis=0)


def _cox(df: pd.DataFrame, covs: list[str], duration: str = "T",
         event: str = "E", entry: str | None = None):
    from lifelines import CoxPHFitter

    cols = [duration, event] + ([entry] if entry else []) + covs
    d = df[cols].dropna()
    cph = CoxPHFitter()
    cph.fit(d, duration_col=duration, event_col=event,
            entry_col=entry, show_progress=False)
    return cph, d


def _ph_test(cph, d: pd.DataFrame):
    """Schoenfeld residual test per covariate."""
    from lifelines.statistics import proportional_hazard_test

    try:
        res = proportional_hazard_test(cph, d, time_transform="rank")
        s = res.summary
        return {str(k): {"test_statistic": float(v["test_statistic"]),
                         "p": float(v["p"])}
                for k, v in s.iterrows()}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------
def _metabric_cohort(cfg: Config, log, endocrine_only: bool) -> pd.DataFrame:
    from ..io import load_metabric, tidy_metabric_expression

    log.info("Loading METABRIC ...")
    m = load_metabric(cfg.paths.metabric_dir)
    expr = tidy_metabric_expression(m["expr"])
    log.info(f"   expression: {expr.shape[0]:,} genes x {expr.shape[1]:,} samples")
    pat, smp = m["patient"], m["sample"]

    # join patient-level survival to sample-level pathology
    smp = smp.copy()
    smp["TUMOR_SIZE"] = pd.to_numeric(smp.get("TUMOR_SIZE"), errors="coerce")
    smp["GRADE"] = pd.to_numeric(smp.get("GRADE"), errors="coerce")
    j = smp.merge(pat, on="PATIENT_ID", how="inner", suffixes=("_s", "_p"))
    if "SAMPLE_ID" not in j.columns:
        raise ValueError("METABRIC sample table has no SAMPLE_ID")
    j = j[j["SAMPLE_ID"].isin(expr.columns)]
    log.info(f"   {len(j):,} tumours with expression and clinical data")

    # ER status: sample-level ER_STATUS is cleaner than the patient ER_IHC typo
    er = j.get("ER_STATUS", pd.Series(index=j.index, dtype=object)).astype(str)
    er_pos = er.str.lower().str.startswith("pos")
    if er_pos.sum() == 0:
        er_pos = j["ER_IHC"].astype(str).str.lower().str.startswith("pos")
    j = j[er_pos]
    log.info(f"   ER-positive: {len(j):,}")

    if endocrine_only:
        j = j[j["HORMONE_THERAPY"].astype(str).str.upper().eq("YES")]
        log.info(f"   hormone-therapy subset: {len(j):,}")

    out = pd.DataFrame({
        "sample": j["SAMPLE_ID"].values,
        "age": pd.to_numeric(j["AGE_AT_DIAGNOSIS"], errors="coerce").values,
        "nodes": pd.to_numeric(j["LYMPH_NODES_EXAMINED_POSITIVE"],
                               errors="coerce").values,
        "size_mm": j["TUMOR_SIZE"].values,
        "grade": j["GRADE"].values,
        "rfs_months": pd.to_numeric(j["RFS_MONTHS"], errors="coerce").values,
        "rfs_event": j["RFS_STATUS"].astype(str).str.startswith("1").values,
        "chemotherapy": j["CHEMOTHERAPY"].astype(str).values,
        "hormone": j["HORMONE_THERAPY"].astype(str).values,
    }).set_index("sample")

    out["cts5"] = [cts5_score(n, s, g, a) for n, s, g, a
                   in zip(out["nodes"], out["size_mm"], out["grade"],
                          out["age"])]
    out["cts5_group"] = out["cts5"].map(cts5_risk_group)
    return out, expr


def _scores(cfg: Config, expr: pd.DataFrame, log) -> pd.DataFrame:
    """Dormancy and stromal scores for the expression columns."""
    panel_path = cfg.paths.results / "signature_panel.csv"
    panel = pd.read_csv(panel_path)
    up = panel[(panel["signature"] == "TRADITIOM_dormancy")
               & (panel["direction"] == "up")]["gene"].astype(str).tolist()
    dorm = _zscore_score(expr, up)
    log.info(f"   dormancy score: {len([g for g in up if g in expr.index])} "
             f"of {len(up)} signature genes present")

    mk_path = cfg.paths.results / "wp3_compartment_markers.json"
    if not mk_path.exists():
        raise FileNotFoundError(f"{mk_path} missing; run `dormbench wp3` first")
    markers = json.loads(mk_path.read_text(encoding="utf-8"))
    stromal_genes = []
    for comp in ("CAFs", "PVL", "Endothelial"):
        stromal_genes += list(markers.get(comp, []))
    stromal = _zscore_score(expr, stromal_genes)
    log.info(f"   stromal score: {len(set(stromal_genes))} marker genes")

    # compartment adjustment: residual of dormancy on stromal content
    d = pd.DataFrame({"dormancy": dorm, "stromal": stromal}).dropna()
    beta = np.polyfit(d["stromal"], d["dormancy"], 1) if len(d) > 10 else [0, 0]
    adj = dorm - (beta[0] * stromal + beta[1])
    return pd.DataFrame({"dormancy": dorm, "stromal": stromal,
                         "dormancy_adj": adj})


def _ordinal_stage(value, prefix: str) -> float:
    """'T2' -> 2.0, 'N1A' -> 1.0, 'TX'/'NX'/NaN -> NaN."""
    import re

    if not isinstance(value, str):
        return np.nan
    s = value.strip().upper()
    if s.startswith(prefix):
        s = s[len(prefix):]
    m = re.match(r"([0-4])", s)
    return float(m.group(1)) if m else np.nan


def _tcga_cohort(cfg: Config, log, er_only: bool = True):
    """TCGA-BRCA: expression + pan-can clinical + IHC ER status.

    TCGA carries no histological grade and no tumour size in the pan-can atlas
    tables, so CTS5 cannot be reconstructed. The benchmark here is therefore a
    REDUCED clinicopathological model: age, pathological T stage, nodal stage.
    That is a weaker incumbent than CTS5 and the comparison must say so.
    """
    import glob

    tcga = cfg.paths.tcga_dir
    expr_path = tcga / "tcga_expression_matrix.parquet"
    if not expr_path.exists():
        raise FileNotFoundError(
            f"{expr_path} not found. Run `dormbench wp3 --only TCGA` first so "
            "the GDC per-sample files are assembled and cached.")
    log.info(f"Loading {expr_path.name} ...")
    expr = pd.read_parquet(expr_path)
    log.info(f"   expression: {expr.shape[0]:,} genes x {expr.shape[1]:,} samples")

    clin_paths = glob.glob(str(tcga / "**" / "data_clinical_patient.txt"),
                           recursive=True)
    if not clin_paths:
        raise FileNotFoundError("no data_clinical_patient.txt under tcga_dir")
    clin = pd.read_csv(clin_paths[0], sep="\t", comment="#", low_memory=False)
    log.info(f"   clinical: {len(clin):,} patients from "
             f"{Path(clin_paths[0]).parent.name}")

    er_path = tcga / "tcga_er_ihc.csv"
    if er_only and er_path.exists():
        er = pd.read_csv(er_path)
        clin = clin.merge(er, left_on="PATIENT_ID", right_on="patient",
                          how="left")
        log.info(f"   IHC ER status merged ({clin['er_status'].notna().sum():,} "
                 "patients with a call)")
    elif er_only:
        log.info("   [!] no tcga_er_ihc.csv; falling back to the PAM50 proxy")

    out = pd.DataFrame({
        "sample": expr.columns.astype(str),
    })
    out["patient"] = out["sample"].str[:12]
    out = out.merge(clin, left_on="patient", right_on="PATIENT_ID", how="left")
    out = out.set_index("sample")

    out["age"] = pd.to_numeric(out["AGE"], errors="coerce")
    out["t_stage"] = out["PATH_T_STAGE"].map(lambda v: _ordinal_stage(v, "T"))
    out["n_stage"] = out["PATH_N_STAGE"].map(lambda v: _ordinal_stage(v, "N"))
    out["dfs_months"] = pd.to_numeric(out["DFS_MONTHS"], errors="coerce")
    out["dfs_event"] = out["DFS_STATUS"].astype(str).str.startswith("1")
    out["os_months"] = pd.to_numeric(out["OS_MONTHS"], errors="coerce")
    out["os_event"] = out["OS_STATUS"].astype(str).str.startswith("1")

    if er_only:
        if "er_status" in out.columns and out["er_status"].notna().any():
            pos = out["er_status"].astype(str).str.lower().eq("positive")
        else:
            pos = out["SUBTYPE"].astype(str).str.contains("Lum", na=False)
        log.info(f"   ER-positive: {int(pos.sum()):,} of {len(out):,}")
        out = out[pos]

    log.info(f"   with DFS time: {int(out['dfs_months'].notna().sum()):,}")
    return out, expr


def run_tcga(cfg: Config, landmark_years: float = 5.0,
             horizon_years: float = 10.0, tag: str | None = None,
             endpoint: str = "OS") -> dict:
    """Reduced-model replication of the late-recurrence question in TCGA."""
    from scipy.stats import chi2 as chi2dist

    log = get_logger("wp4s", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    stem = f"wp4s_{tag}" if tag else "wp4s_tcga"
    report: dict = {"step": "wp4s", "cohort": "TCGA-BRCA",
                    "benchmark": "age + pathological T + nodal stage",
                    "endpoint": endpoint.upper(),
                    "generated": datetime.now().isoformat(timespec="seconds"),
                    "params": {"landmark_years": landmark_years,
                               "horizon_years": horizon_years,
                               "endpoint": endpoint.upper()}}

    banner(log, "WP4s (TCGA) - REDUCED-MODEL REPLICATION")
    log.info("TCGA has no grade or tumour size, so CTS5 cannot be built here.")
    log.info("The incumbent is age + T stage + N stage: a weaker benchmark,")
    log.info("which makes any dormancy signal EASIER to detect, not harder.")
    log.info("")

    clin, expr = _tcga_cohort(cfg, log)
    sc = _scores(cfg, expr, log)
    # DFS carries only ~49 events among ER+ patients in this clinical export,
    # which cannot support a 5-year landmark (144 patients, 8 events). OS at the
    # same landmark is the default; the endpoint is recorded everywhere.
    tcol, ecol = (("os_months", "os_event") if endpoint.upper() == "OS"
                  else ("dfs_months", "dfs_event"))
    log.info(f"   endpoint: {endpoint.upper()}")
    df = clin.join(sc, how="inner").dropna(
        subset=["dormancy", "dormancy_adj", tcol, "age",
                "t_stage", "n_stage"])
    for col in ("dormancy", "dormancy_adj", "age", "stromal"):
        sd = float(df[col].std())
        df[col + "_z"] = (df[col] - df[col].mean()) / sd if sd > 0 else np.nan
    pur = _purity_controls(cfg, "TCGA")
    if pur is not None:
        df = df.join(pur, how="left")
        for col in ("estimate_stromal", "mcp_fibroblast"):
            if col in df.columns and df[col].notna().any():
                sd = float(df[col].std())
                df[col + "_z"] = (df[col] - df[col].mean()) / sd if sd > 0 \
                    else np.nan
        log.info("   independent controls: "
                 + ", ".join(c for c in ("estimate_stromal", "mcp_fibroblast")
                             if c + "_z" in df.columns))
    log.info(f"   complete cases: {len(df):,}")
    corr_ds = float(df[["dormancy", "stromal"]].corr().iloc[0, 1])
    log.info(f"   corr(dormancy, stromal) = {corr_ds:+.3f}")

    lm = df[df[tcol] > landmark_years * 12.0].copy()
    lm["T"] = (lm[tcol] - landmark_years * 12.0) / 12.0
    lm["E"] = lm[ecol].astype(int)
    log.info("")
    log.info(f"   event-free at {landmark_years:.0f} y: {len(lm):,} of "
             f"{len(df):,}")
    log.info(f"   late {endpoint.upper()} events: {int(lm['E'].sum()):,}")
    if int(lm["E"].sum()) < 25:
        report["verdict"] = "INSUFFICIENT_EVENTS"
        report["n_landmark"] = int(len(lm))
        report["n_events"] = int(lm["E"].sum())
        log.info("[!] fewer than 25 late events; cannot conclude")
        with open(cfg.paths.results / f"{stem}_survival.json", "w",
                  encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        return report
    lm.to_csv(cfg.paths.results / f"{stem}_cohort.csv")

    base = ["age_z", "t_stage", "n_stage"]
    # The stromal-only model is the decisive control: if stromal content
    # carries the same information, the dormancy association is composition.
    specs = [("R0_clinic", base),
             ("R1_clinic_dormancy", base + ["dormancy_z"]),
             ("R2_clinic_dormancy_adj", base + ["dormancy_adj_z"]),
             ("R3_clinic_stromal", base + ["stromal_z"]),
             ("R4_clinic_dormancy_stromal", base + ["dormancy_z", "stromal_z"])]
    if "estimate_stromal_z" in df.columns:
        specs.append(("R5_clinic_dormancy_estimate",
                      base + ["dormancy_z", "estimate_stromal_z"]))
    if "mcp_fibroblast_z" in df.columns:
        specs.append(("R6_clinic_dormancy_mcp",
                      base + ["dormancy_z", "mcp_fibroblast_z"]))
    rows, fits = [], {}
    for name, covs in specs:
        cph, d = _cox(lm, covs)
        fits[name] = (cph, d)
        row = {"model": name, "n": int(cph._n_examples),
               "events": int(d["E"].sum()),
               "log_likelihood": float(cph.log_likelihood_),
               "c_index": float(cph.concordance_index_)}
        for c in covs:
            row[f"hr_{c}"] = float(np.exp(cph.params_[c]))
            row[f"p_{c}"] = float(cph.summary.loc[c, "p"])
        rows.append(row)
    models = pd.DataFrame(rows)

    if "estimate_stromal_z" in df.columns:
        lm_p = lm.dropna(subset=["estimate_stromal_z"])
        cph_p, d_p = _cox(lm_p, base + ["dormancy_z"])
        models = pd.concat([models, pd.DataFrame([{
            "model": "R1b_dormancy_same_samples",
            "n": int(cph_p._n_examples), "events": int(d_p["E"].sum()),
            "log_likelihood": float(cph_p.log_likelihood_),
            "c_index": float(cph_p.concordance_index_),
            "hr_dormancy_z": float(np.exp(cph_p.params_["dormancy_z"])),
            "p_dormancy_z": float(cph_p.summary.loc["dormancy_z", "p"])}])],
            ignore_index=True)

    ll0 = models.loc[models["model"] == "R0_clinic", "log_likelihood"].iloc[0]
    lr_rows = []
    for name, _covs in specs[1:]:
        ll1 = models.loc[models["model"] == name, "log_likelihood"].iloc[0]
        chi2 = 2 * (ll1 - ll0)
        p = float(chi2dist.sf(chi2, 1))
        lr_rows.append({"model": name, "lr_chi2": float(chi2), "lr_p": p})
        log.info(f"LR test {name} vs clinical: chi2 = {chi2:.2f}, p = {p:.3g}")
    lr = pd.DataFrame(lr_rows)

    _log_conditional(log, models)
    horizon_y = horizon_years - landmark_years
    calib = []
    for name, (cph, d) in fits.items():
        try:
            sf = cph.predict_survival_function(d, times=[horizon_y])
            pred = 1.0 - sf.iloc[0].values
        except Exception:  # noqa: BLE001
            pred = np.full(len(d), np.nan)
        auc = _ipcw_auc(pred, d["T"].values, d["E"].values, horizon_y)
        models.loc[models["model"] == name, "auc_horizon"] = auc
        cal = _calibration(pred, d["T"].values, d["E"].values, horizon_y)
        if len(cal):
            cal["model"] = name
            calib.append(cal)
    if calib:
        pd.concat(calib).to_csv(cfg.paths.results / f"{stem}_calibration.csv",
                                index=False)
    models.to_csv(cfg.paths.results / f"{stem}_models.csv", index=False)
    lr.to_csv(cfg.paths.results / f"{stem}_lr_tests.csv", index=False)

    log.info("")
    log.info(f"{'model':26s} {'n':>5s} {'ev':>4s} {'C-index':>8s} "
             f"{'AUC':>6s} {'HR dorm':>8s} {'p':>8s}")
    for _, r in models.iterrows():
        hrd = r.get("hr_dormancy_z", r.get("hr_dormancy_adj_z", np.nan))
        p = r.get("p_dormancy_z", r.get("p_dormancy_adj_z", np.nan))
        log.info(f"{r['model']:26s} {int(r['n']):5d} {int(r['events']):4d} "
                 f"{r['c_index']:8.3f} {r.get('auc_horizon', np.nan):6.3f} "
                 + (f"{hrd:8.3f} {p:8.3g}" if np.isfinite(hrd)
                    else f"{'':17s}"))

    ph = {name: _ph_test(cph, d) for name, (cph, d) in fits.items()}
    min_p = _min_ph_p(ph, keys=("dormancy_z", "dormancy_adj_z"))
    min_p_all = _min_ph_p(ph)
    log.info("")
    log.info(f"   Schoenfeld min p, exposure = {min_p:.3g}; "
             f"any covariate = {min_p_all:.3g}")

    tv_rows = []
    full = df.copy()
    full["T_full"] = full[tcol] / 12.0
    full["E"] = full[ecol].astype(int)
    early = full.copy()
    early["T"] = np.minimum(early["T_full"], 5.0)
    cph_e, d_e = _cox(early, base + ["dormancy_z"])
    tv_rows.append({"window": "0-5y", "n": int(cph_e._n_examples),
                    "events": int(d_e["E"].sum()),
                    "hr_dormancy": float(np.exp(cph_e.params_["dormancy_z"])),
                    "p": float(cph_e.summary.loc["dormancy_z", "p"])})
    late = full[full["T_full"] > 5].copy()
    late["T"] = late["T_full"] - 5.0
    cph_l, d_l = _cox(late, base + ["dormancy_z"])
    tv_rows.append({"window": "5-20y", "n": int(cph_l._n_examples),
                    "events": int(d_l["E"].sum()),
                    "hr_dormancy": float(np.exp(cph_l.params_["dormancy_z"])),
                    "p": float(cph_l.summary.loc["dormancy_z", "p"])})
    tv = pd.DataFrame(tv_rows)
    tv.to_csv(cfg.paths.results / f"{stem}_timevarying.csv", index=False)
    for _, r in tv.iterrows():
        log.info(f"   {r['window']:8s} n={int(r['n']):5d} "
                 f"events={int(r['events']):4d} HR={r['hr_dormancy']:.3f} "
                 f"(p={r['p']:.3g})")

    m1 = models[models["model"] == "R1_clinic_dormancy"].iloc[0]
    lr1 = lr[lr["model"] == "R1_clinic_dormancy"].iloc[0]
    # dormancy p-values from the models that CONTROL for composition
    adj = models[models["model"].str.startswith(("R4", "R5", "R6"))
                 & np.isfinite(models.get("p_dormancy_z", np.nan))]
    adj_sig = bool(len(adj)) and float(adj["p_dormancy_z"].min()) < 0.05
    banner(log, "VERDICT")
    log.info(f"clinic only: C = {models['c_index'].iloc[0]:.3f}")
    log.info(f"+ dormancy : C = {m1['c_index']:.3f} "
             f"(HR {m1['hr_dormancy_z']:.2f}, p = {lr1['lr_p']:.3g})")
    log.info(f"+ dormancy, controlling for composition: "
             + (", ".join(f"p = {p:.3g}" for p in adj["p_dormancy_z"])
                if len(adj) else "not run"))
    if min_p < 0.05:
        verdict = "PH_VIOLATED"
    elif lr1["lr_p"] < 0.05 and adj_sig:
        verdict = "DORMANCY_ADDS_TO_CLINICAL_MODEL"
    elif lr1["lr_p"] < 0.05:
        verdict = "DORMANCY_ATTENUATED_BY_COMPOSITION"
        log.info("")
        log.info("The dormancy association is significant without a composition")
        log.info("control and not significant with one. Report it as")
        log.info("composition-attributable, not as prognostic value.")
    else:
        verdict = "DORMANCY_ADDS_NOTHING"
    report.update({"verdict": verdict, "n_landmark": int(len(lm)),
                   "n_events": int(lm["E"].sum()),
                   "models": models.to_dict("records"),
                   "lr_tests": lr.to_dict("records"),
                   "ph_tests": ph, "min_ph_p": float(min_p),
                   "min_ph_p_all_covariates": float(min_p_all),
                   "timevarying": tv.to_dict("records")})
    log.info("")
    log.info(f"VERDICT (TCGA reduced model): {verdict}")
    with open(cfg.paths.results / f"{stem}_survival.json", "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {cfg.paths.results / f'{stem}_survival.json'}")
    return report


# --------------------------------------------------------------------------
def run(cfg: Config, endocrine_only: bool = False, landmark_years: float = 5.0,
        horizon_years: float = 10.0, cohort: str = "METABRIC",
        tag: str | None = None, endpoint: str = "OS") -> dict:
    log = get_logger("wp4s", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    # a sensitivity run must not overwrite the primary outputs
    stem = f"wp4s_{tag}" if tag else "wp4s"
    report: dict = {"step": "wp4s",
                    "generated": datetime.now().isoformat(timespec="seconds"),
                    "params": {"endocrine_only": endocrine_only,
                               "landmark_years": landmark_years,
                               "horizon_years": horizon_years,
                               "cohort": cohort}}

    banner(log, "WP4s - DOES A DORMANCY SCORE ADD ANYTHING TO CTS5?")
    log.info("Landmark cohort: ER-positive, disease-free at 5 years, time")
    log.info("measured from year 5. CTS5 is the incumbent benchmark.")
    log.info("")

    if cohort.upper() in ("TCGA", "TCGA-BRCA"):
        return run_tcga(cfg, landmark_years=landmark_years,
                        horizon_years=horizon_years, tag=tag,
                        endpoint=endpoint)
    if cohort.upper() != "METABRIC":
        raise ValueError(f"unknown cohort: {cohort}")

    clinical, expr = _metabric_cohort(cfg, log, endocrine_only)
    sc = _scores(cfg, expr, log)
    df = clinical.join(sc, how="inner").dropna(
        subset=["cts5", "dormancy", "dormancy_adj", "rfs_months"])
    # A mean-of-z signature score has a small variance (weakly correlated
    # members), so a per-unit hazard ratio is uninterpretable. Standardise to
    # unit SD: the HR is then 'per SD of dormancy', the conventional scale, and
    # the confidence interval states what effect size the cohort excludes.
    for col in ("dormancy", "dormancy_adj"):
        sd = float(df[col].std())
        df[col + "_z"] = (df[col] - df[col].mean()) / sd if sd > 0 else np.nan
        log.info(f"   {col}: SD = {sd:.4f} -> standardised for modelling")
    # independent composition controls, joined on sample
    pur = _purity_controls(cfg, "METABRIC")
    if pur is not None:
        df = df.join(pur, how="left")
        for col in ("estimate_stromal", "mcp_fibroblast"):
            if col in df.columns and df[col].notna().any():
                sd = float(df[col].std())
                df[col + "_z"] = (df[col] - df[col].mean()) / sd if sd > 0 \
                    else np.nan
        log.info("   independent controls: "
                 + ", ".join(c for c in ("estimate_stromal", "mcp_fibroblast")
                             if c + "_z" in df.columns))
    else:
        log.info("   [!] no purity_official_METABRIC.csv; independent "
                 "composition control unavailable")
    log.info(f"   complete cases for survival: {len(df):,}")
    if "chemotherapy" in df.columns:
        log.info("   chemotherapy: " + str(dict(df['chemotherapy']
                                               .value_counts().head(3))))
    log.info("")

    # ---- landmark construction ------------------------------------------
    banner(log, "1. LANDMARK AT 5 YEARS")
    keep = df["rfs_months"] > landmark_years * 12.0
    lm = df[keep].copy()
    lm["T"] = (lm["rfs_months"] - landmark_years * 12.0) / 12.0   # years
    lm["E"] = lm["rfs_event"].astype(int)
    log.info(f"   disease-free at {landmark_years:.0f} y : {len(lm):,} of "
             f"{len(df):,} ({100*len(lm)/len(df):.1f}%)")
    log.info(f"   late recurrence events : {int(lm['E'].sum()):,}")
    log.info(f"   median follow-up beyond landmark: {lm['T'].median():.1f} y")
    if int(lm["E"].sum()) < 30:
        report["verdict"] = "INSUFFICIENT_EVENTS"
        log.info("[!] fewer than 30 late events; nothing can be concluded")
        return report
    log.info("   CTS5 risk groups: " + str(dict(lm["cts5_group"]
                                               .value_counts())))
    log.info("")

    lm.to_csv(cfg.paths.results / f"{stem}_cohort.csv")

    # ---- nested models ---------------------------------------------------
    banner(log, "2. NESTED MODELS  (does dormancy add to CTS5?)")
    specs = [("M0_cts5", ["cts5"]),
             ("M1_cts5_dormancy", ["cts5", "dormancy_z"]),
             ("M2_cts5_dormancy_adj", ["cts5", "dormancy_adj_z"])]
    # Does dormancy survive when composition is CONTROLLED rather than removed
    # from the score first? A joint model estimates the dormancy effect
    # conditional on composition, which is the question a reviewer asks.
    if "stromal_z" in lm.columns:
        specs.append(("M3_cts5_dormancy_stromal",
                      ["cts5", "dormancy_z", "stromal_z"]))
    if "estimate_stromal_z" in lm.columns:
        specs.append(("M4_cts5_dormancy_estimate",
                      ["cts5", "dormancy_z", "estimate_stromal_z"]))
    if "mcp_fibroblast_z" in lm.columns:
        specs.append(("M5_cts5_dormancy_mcp",
                      ["cts5", "dormancy_z", "mcp_fibroblast_z"]))
    fits, rows = {}, []
    for name, covs in specs:
        cph, d = _cox(lm, covs)
        fits[name] = (cph, d)
        hr = cph.params_
        ci = cph.confidence_intervals_
        row = {"model": name, "n": int(cph._n_examples),
               "events": int(d["E"].sum()),
               "log_likelihood": float(cph.log_likelihood_),
               "aic": float(cph.AIC_partial_),
               "c_index": float(cph.concordance_index_)}
        for c in covs:
            row[f"hr_{c}"] = float(np.exp(hr[c]))
            lo, hi = ci.loc[c].values
            row[f"hr_{c}_lo"] = float(np.exp(lo))
            row[f"hr_{c}_hi"] = float(np.exp(hi))
            row[f"p_{c}"] = float(cph.summary.loc[c, "p"])
        rows.append(row)
    models = pd.DataFrame(rows)

    log.info(f"{'model':24s} {'n':>5s} {'ev':>4s} {'C-index':>8s} "
             f"{'HR dorm':>9s} {'95% CI':>16s} {'p':>9s}")
    for _, r in models.iterrows():
        hrd = r.get("hr_dormancy_z", r.get("hr_dormancy_adj_z", np.nan))
        lo = r.get("hr_dormancy_z_lo", r.get("hr_dormancy_adj_z_lo", np.nan))
        hi = r.get("hr_dormancy_z_hi", r.get("hr_dormancy_adj_z_hi", np.nan))
        p = r.get("p_dormancy_z", r.get("p_dormancy_adj_z", np.nan))
        log.info(f"{r['model']:24s} {int(r['n']):5d} {int(r['events']):4d} "
                 f"{r['c_index']:8.3f} "
                 + (f"{hrd:9.3f} {lo:7.3f}-{hi:<7.3f} {p:9.3g}"
                    if np.isfinite(hrd) else f"{'':9s} {'':16s} {'':9s}"))
    log.info("")

    # likelihood-ratio tests against M0
    # Same patients as the purity models, so the attenuation cannot be
    # attributed to the smaller sample they are fitted on.
    if "estimate_stromal_z" in lm.columns:
        lm_p = lm.dropna(subset=["estimate_stromal_z"])
        cph_p, d_p = _cox(lm_p, ["cts5", "dormancy_z"])
        models = pd.concat([models, pd.DataFrame([{
            "model": "M1b_dormancy_same_samples",
            "n": int(cph_p._n_examples), "events": int(d_p["E"].sum()),
            "log_likelihood": float(cph_p.log_likelihood_),
            "c_index": float(cph_p.concordance_index_),
            "hr_dormancy_z": float(np.exp(cph_p.params_["dormancy_z"])),
            "p_dormancy_z": float(cph_p.summary.loc["dormancy_z", "p"])}])],
            ignore_index=True)

    ll0 = models.loc[models["model"] == "M0_cts5", "log_likelihood"].iloc[0]
    lr_rows = []
    for name, _covs in specs[1:]:
        ll1 = models.loc[models["model"] == name, "log_likelihood"].iloc[0]
        chi2 = 2 * (ll1 - ll0)
        from scipy.stats import chi2 as chi2dist
        p = float(chi2dist.sf(chi2, 1))
        lr_rows.append({"model": name, "lr_chi2": float(chi2), "lr_p": p,
                        "delta_c": float(models.loc[models["model"] == name,
                                                    "c_index"].iloc[0]
                                         - models.loc[models["model"]
                                                      == "M0_cts5",
                                                      "c_index"].iloc[0])})
        log.info(f"LR test {name} vs CTS5: chi2 = {chi2:.2f}, p = {p:.3g}")
    lr = pd.DataFrame(lr_rows)
    lr.to_csv(cfg.paths.results / f"{stem}_lr_tests.csv", index=False)
    log.info("")

    _log_conditional(log, models)

    # ---- calibration and time-dependent AUC -----------------------------
    banner(log, "3. CALIBRATION AND TIME-DEPENDENT DISCRIMINATION")
    horizon_y = horizon_years - landmark_years      # years beyond the landmark
    calib_rows = []
    for name, (cph, d) in fits.items():
        try:
            sf = cph.predict_survival_function(d, times=[horizon_y])
            pred = 1.0 - sf.iloc[0].values
        except Exception:  # noqa: BLE001
            pred = np.full(len(d), np.nan)
        auc = _ipcw_auc(pred, d["T"].values, d["E"].values, horizon_y)
        cal = _calibration(pred, d["T"].values, d["E"].values, horizon_y)
        if len(cal):
            cal["model"] = name
            calib_rows.append(cal)
        log.info(f"   {name:24s} AUC at {horizon_years:.0f} y = {auc:.3f}")
        models.loc[models["model"] == name, "auc_horizon"] = auc
    if calib_rows:
        pd.concat(calib_rows).to_csv(
            cfg.paths.results / f"{stem}_calibration.csv", index=False)
    # written after the AUC loop so the discrimination column is included
    models.to_csv(cfg.paths.results / f"{stem}_models.csv", index=False)
    log.info("")

    # ---- proportional hazards -------------------------------------------
    banner(log, "4. PROPORTIONAL HAZARDS")
    ph = {}
    for name, (cph, d) in fits.items():
        ph[name] = _ph_test(cph, d)
        txt = ", ".join(f"{k}={v.get('p', float('nan')):.3g}"
                        for k, v in ph[name].items() if isinstance(v, dict))
        log.info(f"   {name:24s} Schoenfeld p: {txt}")
    # Proportional hazards is judged on the EXPOSURE: a nuisance covariate
    # violating PH (ESTIMATE stromal does, p = 0.03) does not license saying
    # that the dormancy hazard ratio itself is time-varying.
    min_p = _min_ph_p(ph, keys=("dormancy_z", "dormancy_adj_z"))
    min_p_all = _min_ph_p(ph)
    log.info("")
    log.info(f"   Schoenfeld min p, exposure = {min_p:.3g}; "
             f"any covariate = {min_p_all:.3g}")

    # the plan's C3 test: same score, early versus late window
    banner(log, "5. TIME-VARYING HAZARD: 0-5 y vs 5-20 y")
    tv_rows = []
    full = df.copy()
    full["T_full"] = full["rfs_months"] / 12.0
    full["E"] = full["rfs_event"].astype(int)
    full = full.dropna(subset=["cts5", "dormancy"])

    early = full.copy()
    # administrative censoring at 5 years: everyone contributes, anyone whose
    # event falls after the cut is censored at it. Restricting to T <= 5 would
    # condition on having failed or been censored early.
    early["T"] = np.minimum(early["T_full"], 5.0)
    cph_e, d_e = _cox(early, ["cts5", "dormancy_z"])
    tv_rows.append({"window": "0-5y", "n": int(cph_e._n_examples),
                    "events": int(d_e["E"].sum()),
                    "hr_dormancy": float(np.exp(cph_e.params_["dormancy_z"])),
                    "ci_low": float(np.exp(cph_e.confidence_intervals_
                                            .loc["dormancy_z"].iloc[0])),
                    "ci_high": float(np.exp(cph_e.confidence_intervals_
                                             .loc["dormancy_z"].iloc[1])),
                    "p": float(cph_e.summary.loc["dormancy_z", "p"])})

    late = full[full["T_full"] > 5].copy()
    late["T"] = late["T_full"] - 5.0
    cph_l, d_l = _cox(late, ["cts5", "dormancy_z"])
    tv_rows.append({"window": "5-20y", "n": int(cph_l._n_examples),
                    "events": int(d_l["E"].sum()),
                    "hr_dormancy": float(np.exp(cph_l.params_["dormancy_z"])),
                    "ci_low": float(np.exp(cph_l.confidence_intervals_
                                            .loc["dormancy_z"].iloc[0])),
                    "ci_high": float(np.exp(cph_l.confidence_intervals_
                                             .loc["dormancy_z"].iloc[1])),
                    "p": float(cph_l.summary.loc["dormancy_z", "p"])})
    tv = pd.DataFrame(tv_rows)
    tv.to_csv(cfg.paths.results / f"{stem}_timevarying.csv", index=False)
    for _, r in tv.iterrows():
        log.info(f"   {r['window']:8s} n={int(r['n']):5d} "
                 f"events={int(r['events']):4d} "
                 f"HR dormancy={r['hr_dormancy']:.3f} (p={r['p']:.3g})")
    ratio = (tv.loc[tv["window"] == "5-20y", "hr_dormancy"].iloc[0]
             / max(tv.loc[tv["window"] == "0-5y", "hr_dormancy"].iloc[0], 1e-9))
    log.info(f"   HR ratio (late/early) = {ratio:.2f}")
    log.info("   A ratio far from 1 means the dormancy hazard is not constant,")
    log.info("   which is the plan's C3 prediction.")
    log.info("")

    # ---- verdict ---------------------------------------------------------
    banner(log, "VERDICT")
    m1 = models[models["model"] == "M1_cts5_dormancy"].iloc[0]
    m2 = models[models["model"] == "M2_cts5_dormancy_adj"].iloc[0]
    lr1 = lr[lr["model"] == "M1_cts5_dormancy"].iloc[0]
    lr2 = lr[lr["model"] == "M2_cts5_dormancy_adj"].iloc[0]
    best_p = min(lr1["lr_p"], lr2["lr_p"])
    ph_bad = min_p < 0.05

    log.info(f"CTS5 alone: C = {models['c_index'].iloc[0]:.3f}")
    log.info(f"+ dormancy : C = {m1['c_index']:.3f} (LR p = {lr1['lr_p']:.3g})")
    log.info(f"+ adjusted : C = {m2['c_index']:.3f} (LR p = {lr2['lr_p']:.3g})")
    log.info(f"PH assumption violated at p<0.05: {ph_bad} (min p = {min_p:.3g})")
    log.info("")

    if ph_bad:
        verdict = "PH_VIOLATED"
        log.info("The dormancy hazard changes over time, so no single hazard")
        log.info("ratio should be quoted. Report the window-specific estimates")
        log.info("from Section 5 instead, and describe the effect as")
        log.info("time-dependent (the plan's C3).")
    elif best_p < 0.05:
        verdict = "DORMANCY_ADDS_TO_CTS5"
        log.info("The dormancy score improves the model over CTS5 alone. This is")
        log.info("the positive outcome the plan anticipated: report C-index gain,")
        log.info("calibration and the adjusted HR, and replicate before claiming.")
    else:
        verdict = "DORMANCY_ADDS_NOTHING"
        log.info("The dormancy score adds nothing to CTS5 for late recurrence.")
        log.info("This is a publishable null: dormancy state at diagnosis, as")
        log.info("measured by this signature, is not prognostic once")
        log.info("clinicopathological risk is accounted for. State it as such,")
        log.info("and note that it does not contradict Section 3.7, where the")
        log.info("signature was genuinely induced in its source experiment.")

    report.update({
        "verdict": verdict,
        "n_landmark": int(len(lm)),
        "n_events": int(lm["E"].sum()),
        "median_followup_years": float(lm["T"].median()),
        "cts5_groups": {k: int(v) for k, v in
                        lm["cts5_group"].value_counts().items()},
        "models": models.to_dict("records"),
        "lr_tests": lr.to_dict("records"),
        "ph_tests": ph,
        "min_ph_p": float(min_p),
        "min_ph_p_all_covariates": float(min_p_all),
        "timevarying": tv.to_dict("records"),
        "hr_ratio_late_early": float(ratio),
    })
    log.info("")
    log.info(f"VERDICT: {verdict}")

    # ---- figure ----------------------------------------------------------
    try:
        from ..viz.survival import plot_survival

        fig = plot_survival(lm=lm, fits=fits, models=models,
                            calibration=cfg.paths.results / f"{stem}_calibration.csv",
                            horizon_years=horizon_years - landmark_years,
                            timevarying=tv,
                            out_path=cfg.paths.figures / f"fig14_survival{('_{}'.format(tag) if tag else '')}.svg")
        log.info(f"Written: {fig}")
        report["figure"] = str(fig)
    except Exception as e:  # noqa: BLE001
        log.info(f"[!] Figure skipped: {type(e).__name__}: {e}")

    with open(cfg.paths.results / f"{stem}_survival.json", "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {cfg.paths.results / 'wp4s_survival.json'}")
    return report
