"""WP2g — are the leading-edge genes real in the source experiment?

WP2f showed TRADITIOM's up programme is led, in tissue, by endothelial identity
genes (MMRN2, NRP1, PODXL, FGD5, CLIC5, LMO2). TRADITIOM was derived in pure
MCF7 culture where no endothelial cell exists, so there are three possible
explanations and they have very different consequences:

  A. Genuinely induced in dormant MCF7 cells, and not in time-matched untreated
     controls. Dormant cancer cells adopt part of an endothelial-like programme.
     That is real biology (compare vasculogenic mimicry) and the tissue
     mis-scoring is a downstream consequence worth reporting.

  B. Induced in dormant cells AND in untreated controls. Then it is long-term
     culture, not dormancy, and the signature's defining contrast is confounded.

  C. Barely changed in either. The genes entered the signature through weak or
     noisy differential expression, and the tissue behaviour is an artefact of
     signature construction.

The untreated arm (U30, U120, U170) is what separates A from B, and it is the
reason this test is worth running: those samples are time-matched proliferating
controls that most dormancy signature papers do not have.

Runs on results/traditiom_expression.csv and results/wp2f_leading_edge.csv.
No atlas, no network.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config
from ..logging_utils import banner, get_logger

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


def run(cfg: Config, signature: str = "TRADITIOM_dormancy",
        top_n: int = 30) -> dict:
    log = get_logger("wp2g", cfg.paths.logs)
    cfg.paths.ensure_outputs()

    report: dict = {"step": "wp2g", "signature": signature,
                    "generated": datetime.now().isoformat(timespec="seconds")}

    banner(log, f"WP2g — ARE THE LEADING-EDGE GENES INDUCED IN VITRO?  ({signature})")

    expr = pd.read_csv(cfg.paths.results / "traditiom_expression.csv")
    lead = pd.read_csv(cfg.paths.results / "wp2f_leading_edge.csv")

    gene_col = expr.columns[0]
    expr = expr.set_index(gene_col)
    dir_col = "direction" if "direction" in expr.columns else None
    sample_cols = [c for c in expr.columns if c != dir_col]
    grp = pd.Series({c: _assign(c) for c in sample_cols})
    log.info("Sample groups in the TRADITIOM matrix:")
    for g in ORDER:
        n = int((grp == g).sum())
        if n:
            log.info(f"   {g:12s} n={n:2d}")
    log.info("")

    le = lead[(lead["signature"] == signature) & (lead["direction"] == "up")]
    le_genes = [g for g in le["gene"].head(top_n) if g in expr.index]
    all_up = [g for g in expr.index
              if (dir_col and expr.loc[g, dir_col] == "up")] if dir_col else []
    rest = [g for g in all_up if g not in set(le_genes)]
    log.info(f"Leading-edge genes present in matrix: {len(le_genes)}/{len(le)}")
    log.info(f"Other 'up' genes for comparison     : {len(rest)}")
    log.info("")

    def stat(genes, group):
        cols = grp.index[grp == group]
        if len(cols) == 0 or len(genes) == 0:
            return np.nan
        return float(expr.loc[genes, cols].apply(
            pd.to_numeric, errors="coerce").mean(axis=1).mean())

    banner(log, "1. GROUP MEANS (log2 normalised counts)")
    log.info(f"{'group':12s} {'leading edge':>14s} {'other up genes':>16s}")
    rows = []
    for g in ORDER:
        a, b = stat(le_genes, g), stat(rest, g)
        rows.append({"group": g, "leading_edge": a, "other_up": b})
        if not np.isnan(a):
            log.info(f"{g:12s} {a:14.3f} {b:16.3f}")
    report["group_means"] = rows
    log.info("")

    banner(log, "2. THE DECISIVE CONTRAST")
    t0 = stat(le_genes, "T0")
    dorm = stat(le_genes, "Dormancy")
    unt = stat(le_genes, "Untreated")
    log.info(f"leading edge, T0        : {t0:.3f}")
    log.info(f"leading edge, Dormancy  : {dorm:.3f}   ({dorm - t0:+.3f} vs T0)")
    log.info(f"leading edge, Untreated : {unt:.3f}   ({unt - t0:+.3f} vs T0)")
    log.info("")

    # per-gene induction, dormancy vs T0 and untreated vs T0
    t0c = grp.index[grp == "T0"]
    dc = grp.index[grp == "Dormancy"]
    uc = grp.index[grp == "Untreated"]
    per = []
    for g in le_genes:
        v0 = pd.to_numeric(expr.loc[g, t0c], errors="coerce")
        vd = pd.to_numeric(expr.loc[g, dc], errors="coerce")
        vu = pd.to_numeric(expr.loc[g, uc], errors="coerce")
        per.append({
            "gene": g,
            "T0": float(v0.mean()),
            "dormancy": float(vd.mean()),
            "untreated": float(vu.mean()),
            "dorm_minus_T0": float(vd.mean() - v0.mean()),
            "untr_minus_T0": float(vu.mean() - v0.mean()),
            "dorm_minus_untr": float(vd.mean() - vu.mean()),
        })
    per_df = pd.DataFrame(per).sort_values("dorm_minus_T0", ascending=False)
    per_df.to_csv(cfg.paths.results / "wp2g_leading_edge_invitro.csv", index=False)

    log.info("Per-gene, sorted by induction in dormancy:")
    log.info(f"{'gene':16s} {'T0':>7s} {'dorm':>7s} {'untr':>7s} "
             f"{'d-T0':>7s} {'u-T0':>7s} {'d-u':>7s}")
    for _, r in per_df.iterrows():
        log.info(f"{r['gene']:16s} {r['T0']:7.2f} {r['dormancy']:7.2f} "
                 f"{r['untreated']:7.2f} {r['dorm_minus_T0']:+7.2f} "
                 f"{r['untr_minus_T0']:+7.2f} {r['dorm_minus_untr']:+7.2f}")
    log.info("")

    # is dormancy induction bigger than untreated drift?
    d_vs_t0 = per_df["dorm_minus_T0"].values
    u_vs_t0 = per_df["untr_minus_T0"].values
    d_vs_u = per_df["dorm_minus_untr"].values
    try:
        w = stats.wilcoxon(d_vs_t0, u_vs_t0)
        p_pair = float(w.pvalue)
    except ValueError:
        p_pair = np.nan

    banner(log, "3. VERDICT")
    log.info(f"median induction vs T0:  dormancy {np.median(d_vs_t0):+.3f}   "
             f"untreated {np.median(u_vs_t0):+.3f}")
    log.info(f"median dormancy vs untreated: {np.median(d_vs_u):+.3f} "
             f"(Wilcoxon p={p_pair:.3g})")
    log.info(f"genes higher in dormancy than untreated: "
             f"{int((d_vs_u > 0).sum())}/{len(d_vs_u)}")
    log.info("")

    induced = np.median(d_vs_t0) > 0.5
    specific = np.median(d_vs_u) > 0.3

    if induced and specific:
        verdict = "DORMANCY_SPECIFIC"
        log.info("The endothelial-like genes ARE induced in dormant cells and")
        log.info("NOT merely in time-matched untreated controls. This is real")
        log.info("in vitro biology: dormant MCF7 cells switch on part of an")
        log.info("endothelial-like programme. The tissue mis-scoring is then a")
        log.info("downstream consequence worth reporting, and the biology is")
        log.info("the more interesting half of the paper. Compare with the")
        log.info("vasculogenic mimicry literature before claiming novelty.")
    elif induced and not specific:
        verdict = "CULTURE_DRIFT"
        log.info("These genes rise in dormancy AND in untreated controls. The")
        log.info("programme tracks time in culture rather than dormancy, so the")
        log.info("signature's defining contrast is confounded. That is a")
        log.info("substantive criticism of how the signature was derived, and")
        log.info("the untreated arm is what makes it demonstrable.")
    else:
        verdict = "WEAK_IN_SOURCE"
        log.info("These genes are barely changed in the source experiment. They")
        log.info("entered the signature on weak differential expression, and")
        log.info("their tissue behaviour is an artefact of signature")
        log.info("construction rather than of dormancy biology.")

    report["verdict"] = verdict
    report["median_dorm_vs_T0"] = float(np.median(d_vs_t0))
    report["median_untr_vs_T0"] = float(np.median(u_vs_t0))
    report["median_dorm_vs_untr"] = float(np.median(d_vs_u))
    report["wilcoxon_p"] = p_pair
    log.info("")
    log.info(f"VERDICT: {verdict}")

    out = cfg.paths.results / "wp2g_invitro.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {out}")
    log.info(f"Written: {cfg.paths.results / 'wp2g_leading_edge_invitro.csv'}")

    try:
        from ..viz.invitro import plot_invitro
        fig = plot_invitro(expr, grp, le_genes, rest, per_df, signature,
                           cfg.paths.figures / "fig6_invitro_trajectory.svg")
        log.info(f"Written: {fig}")
        report["figure"] = str(fig)
    except Exception as e:  # noqa: BLE001
        log.info(f"[!] Figure skipped: {type(e).__name__}: {e}")

    return report
