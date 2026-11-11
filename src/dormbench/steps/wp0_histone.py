"""WP0 — histone PTM erosion analysis (Rosano et al. 2024, Table S8).

Tests the premise underlying the dormancy-timer hypothesis: do repressive
histone marks rise on entry to dormancy and erode toward awakening, and does
the trajectory replicate across cell lines?

This is a self-contained analysis of a published supplementary table. It runs
in seconds and requires no other data, so it is the cheapest possible check
on whether the project's central biological assumption holds.

Findings from the first run (MCF7, L/H ratios):
    H4K20me3   2.15 -> 10.62 (dormancy) -> 4.59 (awakening)   4.9x up, 0.31x down
    H3K27me3   peaks at d30, already eroded by d60 (different timing)
    H3K9me3    flat throughout -- negative control holds
    H4K20me2   flat throughout -- the me3 signal is not a loading artefact
Global cross-line concordance of fold changes was weak (r ~ 0.14), so models
must be built on the specific replicating marks, not the whole profile.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config
from ..io import load_rosano_table
from ..logging_utils import banner, get_logger

REPRESSIVE = ["H4K20me3", "H3K27me3", "H3K9me2", "H3K9me3"]
ACTIVE = ["H3K4me3", "H3K9ac|K14ac", "H3K18acK23ac"]
CONTROL = ["H4K20me2", "H3K4un"]

STAGE_ORDER = ["T0", "Lat7", "Lat14", "Dorm30", "Dorm60", "Awake", "TEP"]


def _stage(col: str) -> str:
    c = str(col).strip()
    if c.startswith("POT"):
        return "T0"
    if "TEP" in c:
        return "TEP"
    if c.startswith(("T 7", "AI 7")):
        return "Lat7"
    if c.startswith(("T 14", "AI 14")):
        return "Lat14"
    if c.startswith(("T 30", "AI 30")):
        return "Dorm30"
    if c.startswith(("T 60", "AI 60")):
        return "Dorm60"
    return "Awake"


def _arm(col: str) -> str:
    c = str(col).strip()
    if c.startswith("POT"):
        return "POT"
    return "AI" if c.startswith("AI") else "TAM"


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Peptide rows x sample columns, numeric."""
    df = df.rename(columns={df.columns[0]: "Peptide"})
    df = df[df["Peptide"].notna() & df["Peptide"].astype(str).str.startswith("H")]
    return df.set_index("Peptide").apply(pd.to_numeric, errors="coerce")


def run(cfg: Config, make_figure: bool = True) -> dict:
    log = get_logger("wp0", cfg.paths.logs)
    cfg.paths.ensure_outputs()

    report: dict = {
        "step": "wp0",
        "generated": datetime.now().isoformat(timespec="seconds"),
    }

    banner(log, "WP0 — HISTONE PTM EROSION (Rosano Table S8)")

    mcf7 = _tidy(load_rosano_table(cfg.paths.rosano_dir, 8, sheet="MCF7 "))
    t47d = _tidy(load_rosano_table(cfg.paths.rosano_dir, 8, sheet="T47D "))
    log.info(f"MCF7: {mcf7.shape[0]} peptides x {mcf7.shape[1]} samples")
    log.info(f"T47D: {t47d.shape[0]} peptides x {t47d.shape[1]} samples")
    log.info("")

    meta = pd.DataFrame(
        {"stage": [_stage(c) for c in mcf7.columns],
         "arm": [_arm(c) for c in mcf7.columns]},
        index=mcf7.columns,
    )

    def cells(stages, arm=None):
        m = meta.stage.isin(stages)
        if arm:
            m &= (meta.arm == arm) | (meta.arm == "POT")
        return meta.index[m]

    # --- stage means ----------------------------------------------------
    banner(log, "1. STAGE MEANS (L/H ratio)")
    means = {}
    log.info(f"{'mark':16s} " + " ".join(f"{s:>9s}" for s in STAGE_ORDER))
    for m in REPRESSIVE + ACTIVE + CONTROL:
        if m not in mcf7.index:
            continue
        row = []
        for s in STAGE_ORDER:
            v = mcf7.loc[m, cells([s])].dropna()
            row.append(float(v.mean()) if len(v) else np.nan)
        means[m] = row
        log.info(f"{m:16s} " + " ".join(f"{v:9.2f}" for v in row))
    report["stage_means"] = {"stages": STAGE_ORDER, "MCF7": means}
    log.info("")

    # --- rise then fall -------------------------------------------------
    banner(log, "2. RISE INTO DORMANCY, FALL AT AWAKENING")
    dyn = {}
    for m in REPRESSIVE + ACTIVE:
        if m not in mcf7.index:
            continue
        t0 = mcf7.loc[m, cells(["T0"])].dropna()
        dor = mcf7.loc[m, cells(["Dorm30", "Dorm60"])].dropna()
        awk = mcf7.loc[m, cells(["Awake", "TEP"])].dropna()
        if not (len(t0) and len(dor) and len(awk)):
            continue
        up = float(dor.mean() / t0.mean())
        dn = float(awk.mean() / dor.mean())
        p_up = float(stats.mannwhitneyu(dor, t0, alternative="two-sided")[1])
        p_dn = float(stats.mannwhitneyu(awk, dor, alternative="two-sided")[1])
        hit = up > 1.15 and dn < 0.85
        dyn[m] = {"fold_up": up, "fold_down": dn, "p_up": p_up,
                  "p_down": p_dn, "rise_then_fall": bool(hit)}
        log.info(f"{m:16s} T0->dorm {up:5.2f}x (p={p_up:.1e})   "
                 f"dorm->awake {dn:5.2f}x (p={p_dn:.1e})"
                 f"{'   <== rise-then-fall' if hit else ''}")
    report["dynamics"] = dyn
    log.info("")

    # --- threshold convergence -----------------------------------------
    banner(log, "3. THRESHOLD CONVERGENCE (is variance lower at awakening?)")
    log.info("A simple first-passage model predicts cells cross at a COMMON level.")
    log.info("")
    conv = {}
    for m in REPRESSIVE:
        if m not in mcf7.index:
            continue
        d = mcf7.loc[m, cells(["Dorm30", "Dorm60"])].dropna()
        a = mcf7.loc[m, cells(["Awake"])].dropna()
        t = mcf7.loc[m, cells(["TEP"])].dropna()
        cv = lambda x: float(x.std() / x.mean()) if len(x) > 1 and x.mean() else np.nan
        conv[m] = {"cv_dormancy": cv(d), "cv_awakening": cv(a), "cv_tep": cv(t),
                   "n": [len(d), len(a), len(t)]}
        log.info(f"{m:16s} CV dorm {cv(d):6.3f} (n={len(d)})   "
                 f"CV awake {cv(a):6.3f} (n={len(a)})   CV TEP {cv(t):6.3f} (n={len(t)})")
    report["convergence"] = conv
    log.info("")

    # --- arm specificity ------------------------------------------------
    banner(log, "4. ARM SPECIFICITY (AI vs TAM at dormancy)")
    arms = {}
    for m in REPRESSIVE:
        if m not in mcf7.index:
            continue
        sel = meta.stage.isin(["Dorm30", "Dorm60"])
        ai = mcf7.loc[m, meta.index[sel & (meta.arm == "AI")]].dropna()
        tam = mcf7.loc[m, meta.index[sel & (meta.arm == "TAM")]].dropna()
        if not (len(ai) and len(tam)):
            continue
        arms[m] = {"AI": float(ai.mean()), "TAM": float(tam.mean()),
                   "ratio": float(ai.mean() / tam.mean())}
        log.info(f"{m:16s} AI {ai.mean():6.2f}   TAM {tam.mean():6.2f}   "
                 f"ratio {ai.mean()/tam.mean():5.2f}")
    report["arm_specificity"] = arms
    log.info("")

    # --- cross-line concordance ----------------------------------------
    banner(log, "5. CROSS-LINE CONCORDANCE (MCF7 vs T47D)")
    t_stage = pd.Series(
        ["T0" if "POT" in str(c) else ("TEP" if "TEP" in str(c) else "Dorm")
         for c in t47d.columns],
        index=t47d.columns,
    )
    fc1, fc2, shared = [], [], []
    for m in t47d.index:
        if m not in mcf7.index:
            continue
        a = mcf7.loc[m, cells(["T0"])].dropna()
        b = mcf7.loc[m, cells(["Dorm30", "Dorm60"])].dropna()
        c = t47d.loc[m, t_stage.index[t_stage == "T0"]].dropna()
        d = t47d.loc[m, t_stage.index[t_stage == "Dorm"]].dropna()
        if len(a) and len(b) and len(c) and len(d) and a.mean() and c.mean():
            fc1.append(np.log2(b.mean() / a.mean()))
            fc2.append(np.log2(d.mean() / c.mean()))
            shared.append(m)

    r, p = stats.pearsonr(fc1, fc2)
    rho, prho = stats.spearmanr(fc1, fc2)
    log.info(f"n shared peptides = {len(fc1)}")
    log.info(f"Pearson  r   = {r:6.3f}  (p={p:.3g})")
    log.info(f"Spearman rho = {rho:6.3f}  (p={prho:.3g})")
    log.info("")
    log.info("Per-mark T0->dormancy fold change:")
    for m in REPRESSIVE + ACTIVE:
        if m not in shared:
            continue
        i = shared.index(m)
        log.info(f"   {m:16s} MCF7 {2**fc1[i]:5.2f}x   T47D {2**fc2[i]:5.2f}x")
    report["concordance"] = {
        "n_shared": len(fc1), "pearson_r": float(r), "pearson_p": float(p),
        "spearman_rho": float(rho), "spearman_p": float(prho),
        "marks": shared,
        "log2fc_mcf7": [float(x) for x in fc1],
        "log2fc_t47d": [float(x) for x in fc2],
    }
    log.info("")

    # --- verdict --------------------------------------------------------
    banner(log, "VERDICT")
    replicating = [m for m in REPRESSIVE
                   if dyn.get(m, {}).get("rise_then_fall")]
    if replicating:
        log.info(f"Rise-then-fall marks: {replicating}")
        log.info("Erosion premise SUPPORTED — build the coordinate on these marks.")
    else:
        log.info("No mark shows rise-then-fall. Erosion premise NOT supported.")

    if abs(r) < 0.3:
        log.info("")
        log.info("[!] Global cross-line concordance is weak. A model trained on the")
        log.info("    full histone profile will NOT transfer between cell lines.")
        log.info("    Restrict to the specific replicating marks.")

    report["replicating_marks"] = replicating
    report["verdict"] = "SUPPORTED" if replicating else "NOT SUPPORTED"

    out = cfg.paths.results / "wp0_histone_erosion.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info("")
    log.info(f"Written: {out}")

    if make_figure:
        try:
            from ..viz.histone import plot_erosion
            fig_path = plot_erosion(mcf7, meta, t47d, t_stage,
                                    cfg.paths.figures / "wp0_histone_erosion.svg")
            log.info(f"Written: {fig_path}")
            report["figure"] = str(fig_path)
        except Exception as e:  # noqa: BLE001
            log.info(f"[!] Figure skipped: {type(e).__name__}: {e}")

    return report
