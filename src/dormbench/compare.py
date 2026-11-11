"""Old-versus-new comparison report for the review-response steps.

The point of running wp8p, wp2m, wp2n, wp7n, wp3r and tcgaer is not to generate
more numbers. It is to find out which sentences in the manuscript have to
change. Reading six step logs and diffing them by eye is how a stale value
survives into a submitted paper - which has already happened once on this
project.

This reads the frozen supplementary tables shipped in manuscript/supplementary/
(the values the manuscript currently reports) alongside the new results/*.json
and *.csv, and writes one table:

    claim  |  manuscript value  |  new value  |  changed?  |  what to do

Anything marked CHANGED needs a manuscript edit. Anything marked UNCHANGED can
be cited as-is and, where relevant, cited as robustness evidence. Anything
marked NOT RUN is still open.

    dormbench compare        -> results/REVIEW_RESPONSE.md
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .logging_utils import banner, get_logger


def _load_json(p: Path):
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def _load_csv(p: Path):
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return None


def run(cfg, log=None) -> dict:
    log = log or get_logger("compare")
    cfg.paths.ensure_outputs()
    R = cfg.paths.results
    SUPP = cfg.root / "manuscript" / "supplementary"

    banner(log, "REVIEW-RESPONSE COMPARISON")
    log.info("Which reported numbers change, and which survive.")
    log.info("")

    rows: list[dict] = []

    def add(point, claim, old, new, status, action):
        rows.append({"point": point, "claim": claim, "manuscript": old,
                     "new": new, "status": status, "action": action})

    # --- point 12: pseudobulk -------------------------------------------
    d = _load_json(R / "wp8p_pseudobulk.json")
    if d is None:
        add("12", "Donor-level result is 'donor pseudobulk'", "-", "-",
            "NOT RUN", "run `dormbench wp8p`")
    else:
        a = d.get("agreement") or {}
        frac = a.get("frac_same_direction")
        add("12", "Donor-level result is 'donor pseudobulk'",
            "mean of per-cell AUCell scores",
            f"raw-count pseudobulk, {d['verdict']}"
            + (f", {100*frac:.0f}% same direction" if frac is not None else ""),
            "UNCHANGED" if str(d["verdict"]).startswith("PSEUDOBULK_CONFIRMS")
            else "CHANGED",
            "cite wp8p as confirmation and keep wp8 as primary"
            if str(d["verdict"]).startswith("PSEUDOBULK_CONFIRMS")
            else "report the disagreement; do not call wp8 pseudobulk")

    # --- point 9: cell-cycle filter --------------------------------------
    d = _load_json(R / "wp2m_multiverse.json")
    if d is None:
        add("9", "Top-G2M-quartile filter; E2F control neutral", "-", "-",
            "NOT RUN", "run `dormbench wp2m`")
    else:
        stable = d.get("traditiom_direction_stable")
        e2f = d.get("e2f_neutral_everywhere")
        add("9", "TRADITIOM result at the 25% G2M filter",
            "single threshold (top 25%)",
            f"{d['verdict']}, direction stable={stable}",
            "UNCHANGED" if stable else "CHANGED",
            "add the multiverse table to Supplementary"
            if stable else "cannot report a single threshold; report the grid")
        add("9", "E2F negative control is neutral",
            "neutral at the reported threshold",
            f"neutral in all schemes={e2f}",
            "UNCHANGED" if e2f else "CHANGED",
            "cite as threshold-independent" if e2f
            else "describe the control as threshold-dependent")

    # --- point 7: null strength ------------------------------------------
    d = _load_json(R / "wp2n_module_null.json")
    if d is None:
        add("7", "Signatures exceed a size/expression-matched null", "-", "-",
            "NOT RUN", "run `dormbench wp2n`")
    else:
        add("7", "Null comparator strength",
            "random genes matched on size and expression decile",
            f"Hallmark coherent modules; {d['verdict']}",
            "STRENGTHENED",
            "add the Hallmark comparator; it forecloses the "
            "'null too easy' objection")

    # --- points 25/26: redesign ------------------------------------------
    d = _load_json(R / "wp7n_nested.json")
    old_cross = _load_csv(SUPP / "TableS22b_wp7_crossfit.csv")
    old_mean = (f"{old_cross['held_out_auroc'].mean():.3f}"
                if old_cross is not None
                and "held_out_auroc" in old_cross.columns else "-")
    if d is None:
        add("25", "Compartment-neutral redesign, held-out AUROC",
            f"cross-fit mean {old_mean} (cap fixed before folds)", "-",
            "NOT RUN", "run `dormbench wp7n`")
    else:
        n = d.get("nested") or {}
        add("25", "Compartment-neutral redesign, held-out AUROC",
            f"cross-fit mean {old_mean} (cap fixed at 0.1 before folds)",
            f"nested mean {n.get('auroc_mean', float('nan')):.3f} "
            f"+/- {n.get('auroc_sd', float('nan')):.3f}; {d['verdict']}",
            "CHANGED",
            "replace the cross-fit number with the nested one and "
            "state that the cap is selected within training donors")
        add("27", "'Compartment-neutral' as a property of the score",
            "mean held-out AUROC near 0.5",
            f"{n.get('n_stroma_biased', 0)} stroma-biased and "
            f"{n.get('n_cancer_biased', 0)} cancer-biased donors "
            f"of {n.get('n_folds', 0)}",
            "CHANGED" if (n.get("n_stroma_biased", 0)
                          + n.get("n_cancer_biased", 0)) else "UNCHANGED",
            "state the per-donor spread; neutral on average is not "
            "neutral per patient")

    old_trade = _load_csv(SUPP / "TableS22_invariance_tradeoff.csv")
    new_perf = _load_csv(R / "wp7n_performance.csv")
    if old_trade is not None and "cv_accuracy" in old_trade.columns:
        sat = bool((old_trade["cv_accuracy"] >= 0.999).all())
        newv = "-"
        if new_perf is not None and "log_loss" in new_perf.columns:
            newv = (f"log loss {new_perf['log_loss'].iloc[0]:.4f} -> "
                    f"{new_perf['log_loss'].iloc[-1]:.4f}")
        add("26", "'No accuracy cost' from gene exclusion",
            f"CV accuracy 1.000 at every cap (saturated={sat})", newv,
            "CHANGED",
            "delete the no-cost claim; report balanced accuracy, "
            "ROC AUC and log loss, and call the cost unresolved")

    # --- point 4: replication --------------------------------------------
    hits = sorted(R.glob("wp3r_*_replication.json"))
    if not hits:
        add("4", "Central claim rests on 8-9 ER+ donors",
            "one atlas (Wu et al.)", "-", "NOT RUN",
            "obtain a second atlas and run `dormbench wp3r`")
    for h in hits:
        d = _load_json(h) or {}
        rep = d.get("replication") or {}
        add("4", f"Donor-level replication ({h.stem})",
            "one atlas, n=8-9 ER+ donors",
            f"{d.get('n_donors', '?')} donors; {d.get('verdict')}; "
            f"{rep.get('n_replicated', '?')}/"
            f"{rep.get('n_discovery_significant', '?')} replicated",
            "STRENGTHENED" if d.get("verdict") in (
                "REPLICATED", "DIRECTION_REPLICATED_POWER_LIMITED")
            else "CHANGED",
            "report both cohorts; lead with direction consistency")

    # --- point 17: TCGA ER ------------------------------------------------
    d = _load_json(R / "tcgaer.json")
    old_er = _load_csv(SUPP / "TableS12_er_stratified.csv")
    had_tcga = (old_er is not None and "cohort" in old_er.columns
                and old_er["cohort"].astype(str).str.contains("TCGA").any())
    if d is None:
        add("17", "TCGA 'ER status'",
            "PAM50 SUBTYPE proxy (Luminal=ER+, Basal=ER-)" if had_tcga
            else "not reported for TCGA",
            "-", "NOT RUN",
            "run `dormbench tcgaer` then `dormbench wp3 --only TCGA`")
    else:
        add("17", "TCGA 'ER status'",
            "PAM50 SUBTYPE proxy (Luminal=ER+, Basal=ER-)",
            f"clinical IHC, {d.get('n_usable')} patients",
            "CHANGED",
            "re-run wp3 for TCGA and update every ER-stratified "
            "TCGA number and its wording")

    res = pd.DataFrame(rows)
    res.to_csv(R / "review_response.csv", index=False)

    # --- write the report -------------------------------------------------
    lines = [
        "# DormBench review-response comparison",
        f"generated : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"package   : {cfg.root}",
        "",
        "Manuscript values are read from manuscript/supplementary/.",
        "New values are read from results/. CHANGED means a manuscript edit",
        "is required before submission.",
        "",
        "| pt | claim | manuscript | new | status | action |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in res.iterrows():
        lines.append(
            f"| {r['point']} | {r['claim']} | {r['manuscript']} | "
            f"{r['new']} | **{r['status']}** | {r['action']} |")

    counts = res["status"].value_counts().to_dict()
    lines += ["", "## summary", ""]
    for k, v in counts.items():
        lines.append(f"- {k}: {v}")
    todo = res[res["status"] == "NOT RUN"]
    if len(todo):
        lines += ["", "## still to run", ""]
        for _, r in todo.iterrows():
            lines.append(f"- point {r['point']}: {r['action']}")
    lines.append("")

    out = R / "REVIEW_RESPONSE.md"
    out.write_text("\n".join(lines), encoding="utf-8")

    log.info(f"{'pt':>3s} {'status':>12s}  claim")
    for _, r in res.iterrows():
        log.info(f"{r['point']:>3s} {r['status']:>12s}  {r['claim']}")
    log.info("")
    log.info(f"   {counts}")
    log.info("")
    log.info(f"FULL REPORT WRITTEN TO: {out}")
    log.info("Work through every CHANGED row before submitting.")
    return {"step": "compare", "rows": res.to_dict("records"),
            "counts": {k: int(v) for k, v in counts.items()}}
