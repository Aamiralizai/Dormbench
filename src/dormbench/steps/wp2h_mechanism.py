"""WP2h — hypoxia response or endothelial identity?

WP2g showed the tissue leading-edge genes are strongly and specifically induced
in dormant MCF7 cells. WP2f called that programme "endothelial-like", but the
top inducers -- EPAS1 (HIF2A), CD36, CAV1, IGFBP3, IL6 -- are equally
compatible with a hypoxia/metabolic response whose member genes simply happen
to be endothelial-enriched in tissue.

The two hypotheses make opposite predictions and this step tests them directly:

  Endothelial identity  requires the master transcription factors and junctional
                        markers (ERG, FLI1, SOX17, SOX18, TAL1, ETV2, CDH5,
                        PECAM1, VWF, KDR, TEK). Adopting a cell identity means
                        turning on its regulators, not only its metabolic genes.

  Hypoxia response      requires only HIF targets (VEGFA, SLC2A1, CA9, LDHA,
                        PGK1, BNIP3, ADM, NDRG1, P4HA1). These can rise with no
                        change in cell identity at all.

Interpretation is asymmetric and that matters. HIF targets up with endothelial
TFs flat is strong evidence for hypoxia. Endothelial TFs up is suggestive of
identity but not conclusive, since some are themselves hypoxia-responsive.

IMPORTANT LIMITATION: results/traditiom_expression.csv contains only the 1,550
dormancy signature genes, not the transcriptome. Marker genes absent from that
subset cannot be tested here, and absence from the file is NOT evidence they
are unchanged -- it means the question is unanswered and needs the full count
matrix from GEO. Coverage is reported explicitly for that reason.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config
from ..logging_utils import banner, get_logger

PANELS = {
    "Endothelial_TF": ["ERG", "FLI1", "SOX17", "SOX18", "TAL1", "ETV2", "ETS1",
                       "LMO2", "GATA2", "HEY1", "HEY2"],
    "Endothelial_junction": ["CDH5", "PECAM1", "VWF", "KDR", "TEK", "TIE1",
                             "CLDN5", "ESAM", "ROBO4", "MMRN2"],
    "HIF_target": ["VEGFA", "SLC2A1", "CA9", "LDHA", "PGK1", "BNIP3", "ADM",
                   "NDRG1", "P4HA1", "ANKRD37", "EGLN3", "PDK1", "ALDOA",
                   "ENO1", "HK2"],
    "HIF_TF": ["HIF1A", "EPAS1", "ARNT", "HIF3A"],
    "Vasculogenic_mimicry": ["VE-cadherin", "CDH5", "EPHA2", "LAMC2", "MMP2",
                             "MMP9", "TFPI2", "SERPINE1"],
}

GROUPS = {
    "T0": lambda c: c.startswith("POT"),
    "Untreated": lambda c: c.startswith("U") and c[1:2].isdigit(),
    "Latency": lambda c: any(c.startswith(p) for p in ("T7", "T14", "AI7", "AI14")),
    "Dormancy": lambda c: any(c.startswith(p) for p in
                              ("T30", "T60", "AI30", "AI60", "AI90")),
    "Awakening": lambda c: c.startswith(("T.", "AI.")) and "TEP" not in c,
    "TEP": lambda c: "TEP" in c,
}
ORDER = ["T0", "Untreated", "Latency", "Dormancy", "Awakening", "TEP"]


def _assign(col: str) -> str:
    for g in ORDER:
        if GROUPS[g](str(col)):
            return g
    return "Other"


def run(cfg: Config) -> dict:
    log = get_logger("wp2h", cfg.paths.logs)
    cfg.paths.ensure_outputs()

    report: dict = {"step": "wp2h",
                    "generated": datetime.now().isoformat(timespec="seconds")}

    banner(log, "WP2h — HYPOXIA RESPONSE OR ENDOTHELIAL IDENTITY?")

    expr = pd.read_csv(cfg.paths.results / "traditiom_expression.csv")
    gene_col = expr.columns[0]
    expr = expr.set_index(gene_col)
    dir_col = "direction" if "direction" in expr.columns else None
    sample_cols = [c for c in expr.columns if c != dir_col]
    grp = pd.Series({c: _assign(c) for c in sample_cols})

    log.info(f"Matrix: {expr.shape[0]} genes x {len(sample_cols)} samples")
    log.info("NOTE: this file holds only the dormancy signature genes, not the")
    log.info("      transcriptome. Absent markers are UNTESTED, not unchanged.")
    log.info("")

    t0c = grp.index[grp == "T0"]
    dc = grp.index[grp == "Dormancy"]
    uc = grp.index[grp == "Untreated"]

    def gene_stats(g):
        v0 = pd.to_numeric(expr.loc[g, t0c], errors="coerce")
        vd = pd.to_numeric(expr.loc[g, dc], errors="coerce")
        vu = pd.to_numeric(expr.loc[g, uc], errors="coerce")
        return float(v0.mean()), float(vd.mean()), float(vu.mean())

    rows, coverage = [], {}
    for panel, genes in PANELS.items():
        present = [g for g in genes if g in expr.index]
        coverage[panel] = {"present": len(present), "total": len(genes),
                           "found": present,
                           "missing": [g for g in genes if g not in expr.index]}
        banner(log, f"{panel}   ({len(present)}/{len(genes)} genes in matrix)")
        if not present:
            log.info("   none present — UNTESTED with this file")
            log.info("")
            continue
        log.info(f"{'gene':14s} {'T0':>7s} {'dorm':>7s} {'untr':>7s} "
                 f"{'d-T0':>7s} {'d-untr':>8s}")
        for g in present:
            a, b, c = gene_stats(g)
            rows.append({"panel": panel, "gene": g, "T0": a, "dormancy": b,
                         "untreated": c, "dorm_minus_T0": b - a,
                         "dorm_minus_untr": b - c})
            log.info(f"{g:14s} {a:7.2f} {b:7.2f} {c:7.2f} "
                     f"{b-a:+7.2f} {b-c:+8.2f}")
        if coverage[panel]["missing"]:
            log.info(f"   not in matrix (untested): "
                     f"{', '.join(coverage[panel]['missing'])}")
        log.info("")

    df = pd.DataFrame(rows)
    df.to_csv(cfg.paths.results / "wp2h_mechanism_panels.csv", index=False)
    report["coverage"] = coverage

    banner(log, "PANEL SUMMARY")
    summ = []
    log.info(f"{'panel':24s} {'n':>3s} {'median d-T0':>12s} {'median d-untr':>14s}")
    for panel in PANELS:
        sub = df[df["panel"] == panel]
        if sub.empty:
            log.info(f"{panel:24s}   0    UNTESTED")
            summ.append({"panel": panel, "n": 0, "median_dorm_T0": np.nan,
                         "median_dorm_untr": np.nan})
            continue
        m1 = float(sub["dorm_minus_T0"].median())
        m2 = float(sub["dorm_minus_untr"].median())
        summ.append({"panel": panel, "n": int(len(sub)),
                     "median_dorm_T0": m1, "median_dorm_untr": m2})
        log.info(f"{panel:24s} {len(sub):3d} {m1:+12.2f} {m2:+14.2f}")
    report["panels"] = summ
    log.info("")

    # ---- verdict ---------------------------------------------------------
    banner(log, "VERDICT")
    # Pool member genes rather than averaging panel medians: taking the max of
    # panel medians lets a single-gene panel decide the verdict.
    hif_g = df[df["panel"].isin(["HIF_target", "HIF_TF"])]["dorm_minus_untr"]
    endo_g = df[df["panel"].isin(["Endothelial_TF",
                                  "Endothelial_junction"])]["dorm_minus_untr"]
    hif = float(hif_g.median()) if len(hif_g) else np.nan
    endo = float(endo_g.median()) if len(endo_g) else np.nan
    n_hif, n_endo = int(len(hif_g)), int(len(endo_g))

    log.info(f"HIF panels        : {n_hif} genes pooled, median d-untr {hif:+.2f}")
    log.info(f"Endothelial panels: {n_endo} genes pooled, median d-untr {endo:+.2f}")

    # internal consistency: a real hypoxia response moves its targets together
    if len(hif_g) >= 3:
        frac_up = float((hif_g > 0).mean())
        log.info(f"HIF genes moving up: {int((hif_g>0).sum())}/{len(hif_g)} "
                 f"({frac_up*100:.0f}%)")
        if frac_up < 0.75:
            log.info("   [!] The HIF panel is internally inconsistent. A genuine")
            log.info("       hypoxia response moves its targets in one direction;")
            log.info("       a split panel argues against that reading.")
    if len(endo_g) >= 2:
        log.info(f"Endothelial genes moving up: "
                 f"{int((endo_g>0).sum())}/{len(endo_g)}")
    log.info("")

    hif_consistent = (len(hif_g) < 3) or (float((hif_g > 0).mean()) >= 0.75)
    if not hif_consistent:
        verdict = "INCONSISTENT"
        log.info("The HIF targets do not move together, so neither hypothesis")
        log.info("is supported by this panel. Do not report a mechanism from")
        log.info("these genes; get the full count matrix.")
    elif n_hif < 3 or n_endo < 3:
        verdict = "UNDERPOWERED"
        log.info("Too few marker genes are present in this file to discriminate.")
        log.info("The signature-gene subset does not contain enough of either")
        log.info("panel. To answer this you need the full TRADITIOM count matrix")
        log.info("from GEO, not the supplementary expression table.")
    elif hif > endo + 0.5:
        verdict = "HYPOXIA"
        log.info("HIF targets rise substantially while endothelial identity")
        log.info("genes do not. The programme is a hypoxia/metabolic response,")
        log.info("not endothelial reprogramming. The tissue mis-scoring follows")
        log.info("because HIF targets are endothelial-enriched in vivo -- an")
        log.info("expression overlap, not a shared cell state. Frame the paper")
        log.info("that way and drop the mimicry language.")
    elif endo > hif + 0.5:
        verdict = "ENDOTHELIAL_IDENTITY"
        log.info("Endothelial master regulators and junctional markers are")
        log.info("induced beyond the HIF response. That is consistent with")
        log.info("partial endothelial reprogramming. Read the vasculogenic")
        log.info("mimicry literature before claiming novelty -- it is an")
        log.info("established phenomenon in breast cancer.")
    else:
        verdict = "BOTH_OR_AMBIGUOUS"
        log.info("Both panels move comparably. Either hypoxia is driving an")
        log.info("endothelial-like programme (plausible: HIF signalling induces")
        log.info("vascular genes), or the panels are too small to separate.")
        log.info("Check the per-gene table before choosing a framing.")

    report["verdict"] = verdict
    report["hif_median"] = float(hif) if np.isfinite(hif) else None
    report["endo_median"] = float(endo) if np.isfinite(endo) else None
    log.info("")
    log.info(f"VERDICT: {verdict}")

    if verdict == "UNDERPOWERED" or min(n_hif, n_endo) < 6:
        log.info("")
        log.info("NEXT STEP: the definitive test is the full TRADITIOM count")
        log.info("matrix (GEO, from the Rosano data availability statement).")
        log.info("With all ~20k genes both panels become fully testable.")

    out = cfg.paths.results / "wp2h_mechanism.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {out}")
    log.info(f"Written: {cfg.paths.results / 'wp2h_mechanism_panels.csv'}")

    return report
