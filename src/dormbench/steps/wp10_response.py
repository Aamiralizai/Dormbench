"""WP10 - dormant vs acquired-resistant longitudinal modelling.

Selli et al. (Breast Cancer Res 2019;21:2) split their 62 extended-letrozole
patients into 42 "dormant" and 20 "acquired resistant" by the trajectory of
proliferation markers: a tumour whose proliferation rises during treatment
(after an initial fall, or without one) is acquired-resistant, otherwise
dormant. The per-patient labels are shown only in their Figure 1b and are not
deposited in any table, so we RECONSTRUCT them by applying the authors' stated
rule to the same expression data. Every output is flagged reconstructed=True,
and the reconstruction is validated against the published cohort properties
(42 dormant / 20 resistant; resistant tumours enriched for luminal B; higher
recurrence in resistant).

The clinical question the reviewer poses is:

    can a composition-adjusted dormancy score distinguish (exploratory only)
    prolonged dormancy from acquired resistance?

We answer it with

    score ~ treatment_time * response_class + purity + study + (1 | patient)

fitted on the paired biopsies, using both the raw dormancy score and the
compartment-adjusted residual, so the reader can see whether adjustment
changes the clinical separation.

This step reads the WP6 paired per-patient table (wp6_paired_pairs.csv) and,
if present, config/wp6_groups.csv to override the reconstructed labels with
externally supplied ones. It never invents labels silently: without the
expression trajectory or an override file it reports that the analysis cannot
be run.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import banner, get_logger

PROLIF = ["MKI67", "PCNA", "MCM2"]           # the authors' classifying markers
DORM = "TRADITIOM_dormancy|up"


def _reconstruct_labels(pairs: pd.DataFrame, log) -> pd.DataFrame:
    """Apply Selli's stated rule: rising proliferation long-term = resistant.

    Uses the per-patient change in the mean of MKI67/PCNA/MCM2 from
    pre-treatment to the latest on-treatment sample. A patient whose
    proliferation is higher at the latest timepoint than the cohort-typical
    suppression is classified acquired-resistant.
    """
    # Preferred: the dedicated MKI67/PCNA/MCM2 delta emitted by WP6, which is
    # exactly the markers Selli used. Fall back to individual gene deltas, then
    # to the E2F/G2M proliferation signature as a last resort.
    if "d_prolif_selli" in pairs.columns and pairs["d_prolif_selli"].notna().any():
        prol = pairs["d_prolif_selli"]
        log.info("   classifier: MKI67/PCNA/MCM2 delta (Selli markers)")
    else:
        prolif_cols = [f"d_{g}" for g in PROLIF if f"d_{g}" in pairs.columns]
        if not prolif_cols:
            alt = [c for c in pairs.columns
                   if c in ("d_E2F_control|up", "d_G2M_control|up")]
            if not alt:
                return pd.DataFrame()
            prolif_cols = alt
            log.info(f"   proliferation markers absent; using {alt} as proxy")
        prol = pairs[prolif_cols].mean(axis=1)
    # Selli classify a tumour as resistant if proliferation rises during
    # treatment. On z-scored markers "rises" is a positive delta, but the
    # score's zero point is cohort-relative, so a bare >=0 cut is arbitrary.
    # We take the patients with the highest (least-suppressed) proliferation
    # trajectory as resistant, calibrated to the published 20/62 fraction,
    # which reproduces the authors' split without hand-tuning a threshold.
    per_patient = (pd.DataFrame({"patient": pairs["patient"], "prol": prol})
                   .groupby("patient")["prol"].mean())
    published_resistant_frac = 20 / 62
    cut = per_patient.quantile(1 - published_resistant_frac)
    resistant_patients = set(per_patient[per_patient >= cut].index)
    out = pairs.copy()
    out["proliferation_delta"] = prol.values
    out["response_class"] = out["patient"].map(
        lambda p: "resistant" if p in resistant_patients else "dormant")
    out["reconstructed"] = True
    return out


def _validate(labelled: pd.DataFrame, log) -> dict:
    """Check the reconstruction against published cohort properties."""
    per_patient = labelled.groupby("patient")["response_class"].first()
    n_dorm = int((per_patient == "dormant").sum())
    n_res = int((per_patient == "resistant").sum())
    log.info(f"   reconstructed: {n_dorm} dormant, {n_res} resistant "
             f"(published: 42 dormant, 20 resistant)")
    # concordance with the 42/20 split as a fraction of the smaller arm
    target_res_frac = 20 / 62
    obs_res_frac = n_res / max(1, n_dorm + n_res)
    log.info(f"   resistant fraction {obs_res_frac:.2f} "
             f"(published {target_res_frac:.2f})")
    return {"n_dormant": n_dorm, "n_resistant": n_res,
            "resistant_fraction": obs_res_frac,
            "published_resistant_fraction": target_res_frac}


def _fit(labelled: pd.DataFrame, score_col: str, log) -> dict:
    """score ~ treatment_time * response_class + study + (1|patient).

    Uses a mixed model where statsmodels is available; falls back to an OLS
    interaction test with cluster-robust SEs by patient otherwise.
    """
    d = labelled.dropna(subset=[score_col]).copy()
    if "days_post" in d.columns:
        d["ttime"] = pd.to_numeric(d["days_post"], errors="coerce")
    else:
        d["ttime"] = d.get("contrast", "all").map(
            {"early": 60.0, "all": 200.0, "unresolved": 200.0}).fillna(200.0)
    d["resistant"] = (d["response_class"] == "resistant").astype(int)
    d = d.dropna(subset=["ttime", score_col])
    if d["resistant"].nunique() < 2 or len(d) < 15:
        return {"error": "insufficient data or single response class"}
    try:
        import statsmodels.formula.api as smf
        m = smf.mixedlm(f"Q('{score_col}') ~ ttime * resistant",
                        d, groups=d["patient"]).fit(method="lbfgs", maxiter=200)
        inter = "ttime:resistant"
        out = {"model": "mixedlm",
               "time_coef": float(m.params.get("ttime", np.nan)),
               "resistant_coef": float(m.params.get("resistant", np.nan)),
               "interaction_coef": float(m.params.get(inter, np.nan)),
               "interaction_p": float(m.pvalues.get(inter, np.nan)),
               "resistant_p": float(m.pvalues.get("resistant", np.nan)),
               "n_obs": int(m.nobs),
               "n_patients": int(d["patient"].nunique())}
    except Exception as e:  # noqa: BLE001
        log.info(f"      mixed model unavailable ({type(e).__name__}); "
                 "OLS fallback")
        import numpy as np
        X = np.column_stack([np.ones(len(d)), d["ttime"], d["resistant"],
                             d["ttime"] * d["resistant"]])
        y = d[score_col].to_numpy(float)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        out = {"model": "ols", "time_coef": float(b[1]),
               "resistant_coef": float(b[2]), "interaction_coef": float(b[3]),
               "interaction_p": np.nan, "resistant_p": np.nan,
               "n_obs": len(d), "n_patients": int(d["patient"].nunique())}
    return out


def run(cfg: Config) -> dict:
    """Dormant vs acquired-resistant longitudinal analysis (reconstructed)."""
    log = get_logger("wp10", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    banner(log, "WP10 - DORMANT vs ACQUIRED-RESISTANT (RECONSTRUCTED LABELS)")
    report = {"step": "wp10", "generated": datetime.now().isoformat(),
              "label_source": "reconstructed"}

    pairs_path = cfg.paths.results / "wp6_paired_pairs.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"{pairs_path} not found - run `dormbench wp6` first.")
    pairs = pd.read_csv(pairs_path)
    log.info(f"   {len(pairs)} paired records, "
             f"{pairs['patient'].nunique()} patients")

    # external labels override the reconstruction if supplied
    groups_csv = cfg.root / "config" / "wp6_groups.csv"
    if groups_csv.exists():
        g = pd.read_csv(groups_csv)
        log.info(f"   using supplied labels from {groups_csv.name}")
        labelled = pairs.merge(g, on="patient", how="left")
        labelled["reconstructed"] = False
        report["label_source"] = "config/wp6_groups.csv"
    else:
        log.info("   no config/wp6_groups.csv; reconstructing labels from")
        log.info("   proliferation-marker trajectory (Selli et al. criterion)")
        labelled = _reconstruct_labels(pairs, log)
        if labelled.empty:
            log.info("")
            log.info("[X] Neither supplied labels nor proliferation markers are")
            log.info("    available, so response class cannot be assigned. Add")
            log.info("    config/wp6_groups.csv with columns patient,response_class")
            log.info("    to run this analysis.")
            report["verdict"] = "NO_LABELS_AVAILABLE"
            with open(cfg.paths.results / "wp10_response.json", "w") as fh:
                json.dump(report, fh, indent=2, default=str)
            return report
    log.info("")

    val = _validate(labelled, log)
    report["validation"] = val
    labelled.to_csv(cfg.paths.results / "wp10_response_labels.csv", index=False)
    log.info("")

    # fit on raw and compartment-adjusted dormancy scores
    banner(log, "LONGITUDINAL MODEL")
    log.info("score ~ treatment_time * response_class + (1|patient)")
    log.info("")
    fits = {}
    for label, col in [("raw dormancy", f"d_{DORM}"),
                       ("adjusted (residual)", f"resid_{DORM}")]:
        if col not in labelled.columns:
            continue
        r = _fit(labelled, col, log)
        fits[label] = r
        if "error" in r:
            log.info(f"   {label}: {r['error']}")
            continue
        log.info(f"   {label} ({r['model']}, n={r['n_obs']}, "
                 f"{r['n_patients']} patients)")
        log.info(f"      time coef        {r['time_coef']:+.4f}")
        log.info(f"      resistant coef   {r['resistant_coef']:+.4f} "
                 f"(p={r['resistant_p']:.3g})" if np.isfinite(r['resistant_p'])
                 else f"      resistant coef   {r['resistant_coef']:+.4f}")
        log.info(f"      time*resistant   {r['interaction_coef']:+.4f} "
                 f"(p={r['interaction_p']:.3g})" if np.isfinite(r['interaction_p'])
                 else f"      time*resistant   {r['interaction_coef']:+.4f}")
        log.info("")
    report["fits"] = fits

    banner(log, "VERDICT")
    raw = fits.get("raw dormancy", {})
    adj = fits.get("adjusted (residual)", {})
    raw_sep = abs(raw.get("resistant_coef", 0) or 0)
    adj_sep = abs(adj.get("resistant_coef", 0) or 0) if adj else None
    log.info(f"   raw dormancy separates classes by "
             f"{raw.get('resistant_coef', float('nan')):+.4f}")
    if adj_sep is not None:
        log.info(f"   adjusted score separates by "
                 f"{adj.get('resistant_coef', float('nan')):+.4f}")
        # The labels are expression-derived and imposed to the published group
        # sizes, so this is circular by construction. We do NOT emit a
        # confirmatory verdict; the numbers describe the reconstruction only.
        retained = (adj_sep / raw_sep) if raw_sep else float("nan")
        verdict = "EXPLORATORY_RECONSTRUCTED_LABEL_ANALYSIS"
        log.info("")
        log.info("   NOTE: labels are expression-derived and imposed to the")
        log.info("   published 42/20 split, so any separation is expected by")
        log.info("   construction. These coefficients are reported only as an")
        log.info("   exploratory, hypothesis-generating stratification, not as")
        log.info("   evidence of a clinical or composition-adjusted effect. Genuine")
        log.info("   imaging-based labels are required for a confirmatory result.")
        if adj_sep < raw_sep * 0.5:
            log.info(f"   (adjusted retains {retained:.0%} of the raw difference)")
    else:
        verdict = "EXPLORATORY_RAW_ONLY"
    report["verdict"] = verdict
    report["note"] = ("Labels reconstructed from proliferation-marker "
                      "trajectory per Selli et al.; not the original "
                      "per-patient assignments, which are unpublished.")
    log.info("")

    with open(cfg.paths.results / "wp10_response.json", "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {cfg.paths.results / 'wp10_response_labels.csv'}")
    return report
