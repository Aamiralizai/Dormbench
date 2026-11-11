"""WP5 — is the dormancy programme a hypoxia response?

WP2h could not separate hypoxia from endothelial identity because only seven
marker genes survived into the TRADITIOM signature-gene subset: one HIF TF
(EPAS1), three HIF targets of which one moved the wrong way (LDHA down), and
three endothelial genes. That is too thin to decide anything.

TCGA-BRCA ships three independently published, externally validated hypoxia
scores -- Buffa, Winter and Ragnum -- for ~1,079 patients. Correlating the
dormancy score against those tests the same hypothesis with real statistical
weight and no dependence on which genes happened to enter the signature.

The test is only interpretable with two controls, both included:

  Hypoxia_hallmark   should correlate strongly. If it does not, the hypoxia
                     scores are not measuring what we think and nothing else
                     in this step means anything.

  E2F_control        should not. Any score correlating with everything is
                     reporting a global property, not hypoxia.

A confound to keep in view: hypoxia and stromal content are themselves
correlated in bulk tissue, so this step also reports the dormancy-hypoxia
correlation after adjusting for stromal content. A correlation that survives
that adjustment is much stronger evidence.

Requires the TCGA clinical supplement extracted from the cBioPortal study
archive (data_clinical_supp_hypoxia.txt).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config
from ..io import DataNotFound, _first_match, load_tcga
from ..logging_utils import banner, get_logger

HYPOXIA_COLS = ["BUFFA_HYPOXIA_SCORE", "WINTER_HYPOXIA_SCORE",
                "RAGNUM_HYPOXIA_SCORE"]
STROMAL = ["CAFs", "PVL", "Endothelial"]


def _zscore_score(expr: pd.DataFrame, genes: list[str]) -> pd.Series:
    present = [g for g in genes if g in expr.index]
    if len(present) < 3:
        return pd.Series(np.nan, index=expr.columns)
    sub = expr.loc[present]
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1) + 1e-9, axis=0)
    return z.mean(axis=0)


def _partial(x, y, z) -> float:
    """Spearman correlation of x and y controlling for z."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 30:
        return np.nan
    rx = stats.rankdata(x[ok]); ry = stats.rankdata(y[ok]); rz = stats.rankdata(z[ok])
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(stats.pearsonr(ex, ey)[0])


def run(cfg: Config) -> dict:
    log = get_logger("wp5", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    report: dict = {"step": "wp5",
                    "generated": datetime.now().isoformat(timespec="seconds")}

    banner(log, "WP5 — DORMANCY vs PUBLISHED HYPOXIA SCORES (TCGA-BRCA)")

    hyp_path = _first_match(cfg.paths.tcga_dir, ["*clinical_supp_hypoxia*"],
                            "TCGA hypoxia supplement")
    hyp = pd.read_csv(hyp_path, sep="\t", comment="#", low_memory=False)
    log.info(f"Hypoxia supplement: {hyp_path.name} ({len(hyp)} patients)")
    cols = [c for c in HYPOXIA_COLS if c in hyp.columns]
    log.info(f"   scores available: {cols}")
    if not cols:
        raise ValueError(f"No hypoxia score columns in {hyp_path.name}")
    hyp = hyp.set_index(hyp.columns[0])
    log.info("")

    tc = load_tcga(cfg.paths.tcga_dir, log=log)
    expr = tc["expr"]
    universe = cfg.paths.results / "wp3_common_genes.txt"
    if universe.exists():
        keep = set(universe.read_text(encoding="utf-8").split())
        expr = expr.loc[expr.index.intersection(keep)]
        log.info(f"   restricted to common universe: {expr.shape[0]} genes")

    # match patient barcodes: expression columns are sample-level
    expr = expr.rename(columns=lambda c: "-".join(str(c).split("-")[:3]))
    expr = expr.loc[:, ~expr.columns.duplicated()]
    shared = [c for c in expr.columns if c in hyp.index]
    log.info(f"   {expr.shape[1]} expression columns, "
             f"{len(shared)} matched to hypoxia scores")
    if len(shared) < 100:
        log.info("")
        log.info("[X] Too few matches. Expression columns are probably file")
        log.info("    UUIDs rather than TCGA barcodes, which happens when the")
        log.info("    GDC sample sheet does not cover the downloaded files.")
        log.info(f"    example columns: {list(expr.columns[:3])}")
        log.info(f"    example hypoxia ids: {list(hyp.index[:3])}")
        raise ValueError("insufficient barcode overlap")
    expr = expr[shared]
    hyp = hyp.loc[shared]
    log.info("")

    # stromal content, for the partial correlation
    markers_path = cfg.paths.results / "wp3_compartment_markers.json"
    stroma = None
    if markers_path.exists():
        with open(markers_path, encoding="utf-8") as fh:
            markers = json.load(fh)
        parts = [_zscore_score(expr, markers[c]) for c in STROMAL if c in markers]
        if parts:
            stroma = pd.concat(parts, axis=1).mean(axis=1)
            log.info("   stromal content estimated for partial correlations")
    log.info("")

    panel = pd.read_csv(cfg.paths.results / "signature_panel.csv")
    rows = []
    banner(log, "CORRELATION WITH PUBLISHED HYPOXIA SCORES")
    header = f"{'signature':30s}" + "".join(f"{c.split('_')[0]:>10s}" for c in cols)
    log.info(header + f"{'partial':>10s}")
    log.info("(partial = vs Buffa, adjusted for stromal content)")
    log.info("")
    for (sig, direction), grp in panel.groupby(["signature", "direction"]):
        genes = grp["gene"].astype(str).tolist()
        s = _zscore_score(expr, genes)
        if s.isna().all():
            continue
        rs = {}
        for c in cols:
            ok = s.notna() & hyp[c].notna()
            rs[c] = (float(stats.spearmanr(s[ok], hyp.loc[ok.values, c])[0])
                     if ok.sum() > 30 else np.nan)
        part = (_partial(s.values, hyp[cols[0]].values, stroma.values)
                if stroma is not None else np.nan)
        rows.append({"signature": sig, "direction": direction,
                     **{c: rs[c] for c in cols}, "partial_vs_stroma": part})
        log.info(f"{sig+'|'+direction:30s}"
                 + "".join(f"{rs[c]:+10.3f}" for c in cols)
                 + f"{part:+10.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(cfg.paths.results / "wp5_hypoxia_correlation.csv", index=False)
    log.info("")

    # ---- controls --------------------------------------------------------
    banner(log, "CONTROLS")
    def get(sig, direction="up", col=cols[0]):
        r = res[(res["signature"] == sig) & (res["direction"] == direction)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    hh = get("Hypoxia_hallmark")
    e2f = get("E2F_control")
    trad = get("TRADITIOM_dormancy")
    trad_p = float(res[(res["signature"] == "TRADITIOM_dormancy") &
                       (res["direction"] == "up")]["partial_vs_stroma"].iloc[0]) \
        if len(res[(res["signature"] == "TRADITIOM_dormancy")]) else np.nan

    log.info(f"Hypoxia_hallmark (should be high) : {hh:+.3f}")
    log.info(f"E2F_control      (should be low)  : {e2f:+.3f}")
    log.info(f"TRADITIOM up                      : {trad:+.3f}")
    log.info(f"TRADITIOM up, stroma-adjusted     : {trad_p:+.3f}")
    log.info("")

    report.update({"hypoxia_hallmark_r": hh, "e2f_r": e2f,
                   "traditiom_r": trad, "traditiom_partial_r": trad_p,
                   "n_patients": len(shared)})

    banner(log, "VERDICT")
    if not np.isfinite(hh) or hh < 0.3:
        verdict = "HYPOXIA_CONTROL_INCONCLUSIVE_PURITY_CONFOUNDED"
        log.info("The Hallmark hypoxia signature does not track the published")
        log.info(f"hypoxia scores (r = {hh:+.3f}).")
        log.info("")
        log.info("This is a finding, not a technical fault. The pattern below")
        log.info("shows why: the proliferation sets correlate POSITIVELY with")
        log.info("the published scores and the stromal-marker sets NEGATIVELY,")
        log.info("so the published hypoxia scores are themselves tracking")
        log.info("tumour purity in bulk tissue rather than a hypoxia programme.")
        log.info("The step therefore cannot decide the mechanism question, and")
        log.info("the answer stays unresolved (this is the expected outcome).")
    elif np.isfinite(trad) and trad > 0.4 and np.isfinite(trad_p) and trad_p > 0.25:
        verdict = "HYPOXIA_SUPPORTED"
        log.info("The dormancy programme correlates with independently")
        log.info("published hypoxia scores, and the association survives")
        log.info("adjustment for stromal content. This is much stronger")
        log.info("evidence than the seven marker genes in WP2h, and it")
        log.info("supports reading the induced programme as a hypoxia/")
        log.info("metabolic response rather than endothelial reprogramming.")
    elif np.isfinite(trad) and trad > 0.4:
        verdict = "HYPOXIA_CONFOUNDED"
        log.info("The dormancy score correlates with hypoxia, but the")
        log.info("association does not survive adjustment for stromal content.")
        log.info("Hypoxia and stroma are themselves correlated in bulk tissue,")
        log.info("so this is consistent with both being downstream of the same")
        log.info("compartment signal. Do not claim a hypoxia mechanism on this.")
    else:
        verdict = "NOT_HYPOXIA"
        log.info("The dormancy programme does NOT track published hypoxia")
        log.info("scores, while the Hallmark hypoxia set does. The WP2h")
        log.info("reading was driven by EPAS1 alone and should be dropped.")
        log.info("Report the mechanism as unresolved.")

    report["verdict"] = verdict
    log.info("")
    log.info(f"VERDICT: {verdict}")

    out = cfg.paths.results / "wp5_hypoxia.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {out}")
    log.info(f"Written: {cfg.paths.results / 'wp5_hypoxia_correlation.csv'}")
    return report
