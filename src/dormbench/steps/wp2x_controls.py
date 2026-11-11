"""WP2x — matched control signatures from the same experiment.

The Hallmark reference sets used elsewhere establish that compartment
attribution is driven by cell-type marker content rather than by the programme
a gene set is named after. They are, however, derived from entirely different
sources than the dormancy signatures, so a sceptic can reasonably ask whether
the comparison is fair.

This step derives three additional signatures from the same experiment, the
same cells and the same differential-expression pipeline as TRADITIOM, differing
only in the biology they encode:

  Culture_drift   Untreated cells at day 170 versus parental (Table S4). These
                  cells were never treated: the signature encodes long-term
                  culture and nothing about dormancy. If it behaves like the
                  dormancy signature in tissue, the compartment effect is a
                  property of in vitro derivation rather than of dormancy.

  Awakening       Awakening carbon copies versus dormancy (Tables S5, S6),
                  pooled across arms. This encodes the OPPOSITE state to
                  dormancy -- cells resuming proliferation. Two signatures
                  encoding opposite states that behave identically in tissue
                  would show the compartment preference carries no information
                  about the state at all.

  Latency         Days 7-14 versus parental (Tables S5, S6). An intermediate
                  timepoint on the same adaptation arc.

Arm-specific variants (AI and TAM) are also written, so the analysis can test
whether the effect is treatment-generic.

Sheets are laid out with a prose description in row 0 and the real header in
row 1, and gene symbols in the first column. Direction is taken from the sign
of the log2 fold change.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..genes import assert_no_dates, validate_and_recover
from ..io import _first_match
from ..logging_utils import banner, get_logger

# table -> sheet -> (signature, arm)
SPEC = {
    4: {
        "UT170vsPOT": ("Culture_drift", "untreated"),
        "UT120vsPOT": ("Culture_drift_d120", "untreated"),
    },
    5: {
        "AI7vsPOT": ("Latency", "AI"), "AI14vsPOT": ("Latency", "AI"),
        "AIalphaVsDromancy": ("Awakening", "AI"),
        "AIBetaVsDormancy": ("Awakening", "AI"),
        "AIGammaVsDormancy": ("Awakening", "AI"),
    },
    6: {
        "T7vsPOT": ("Latency", "TAM"), "T14vsPOT": ("Latency", "TAM"),
        "TAlphaVsDormancy": ("Awakening", "TAM"),
        "TBetaVsDormancy": ("Awakening", "TAM"),
        "TGammaVsDormancy": ("Awakening", "TAM"),
        "TDeltaVsDormancy": ("Awakening", "TAM"),
        "TEpsilonVsDormancy": ("Awakening", "TAM"),
        "TZetaVsDormancy": ("Awakening", "TAM"),
    },
}

MIN_FRACTION = 0.5   # for pooled sets, gene must appear in half the sheets


def _read_deg(path, sheet, log) -> pd.DataFrame | None:
    """Return gene / log2FC for a DESeq2 sheet with a prose row above the header."""
    raw = pd.read_excel(path, sheet_name=sheet, header=1)
    if raw.shape[1] < 2:
        return None
    cols = [str(c) for c in raw.columns]
    gcol = cols[0]
    lcol = next((c for c in cols
                 if "log2" in c.lower() or "fold" in c.lower()), None)
    if lcol is None:
        num = raw.select_dtypes("number").columns
        lcol = num[0] if len(num) else None
    if lcol is None:
        log.info(f"      [!] {sheet}: no fold-change column; skipped")
        return None
    out = pd.DataFrame({
        "gene": raw[gcol].astype(str).str.strip(),
        "lfc": pd.to_numeric(raw[lcol], errors="coerce"),
    }).dropna()
    out = out[~out["gene"].str.lower().isin(["nan", "na", "", "gene", "none"])]
    return out


def run(cfg: Config, min_fraction: float = MIN_FRACTION) -> dict:
    log = get_logger("wp2x", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    report: dict = {"step": "wp2x",
                    "generated": datetime.now().isoformat(timespec="seconds")}

    banner(log, "WP2x — MATCHED CONTROL SIGNATURES FROM THE SAME EXPERIMENT")

    collected: dict[tuple, list[pd.DataFrame]] = {}
    for tbl, sheets in SPEC.items():
        path = _first_match(cfg.paths.rosano_dir,
                            [f"*supplementary_table_{tbl}_*.xlsx"],
                            f"Rosano table {tbl}")
        log.info(f"Table S{tbl}: {path.name}")
        for sheet, (sig, arm) in sheets.items():
            try:
                d = _read_deg(path, sheet, log)
            except Exception as e:  # noqa: BLE001
                log.info(f"      [!] {sheet}: {type(e).__name__}: {e}")
                continue
            if d is None or d.empty:
                continue
            log.info(f"   {sheet:22s} -> {sig:20s} ({arm})  {len(d):6d} genes")
            collected.setdefault((sig, arm), []).append(d)
        log.info("")

    rows = []
    summary = []
    for (sig, arm), frames in collected.items():
        n_sheets = len(frames)
        allg = pd.concat(frames, ignore_index=True)
        # a gene must be consistent in direction and recurrent across sheets
        agg = allg.groupby("gene").agg(
            n=("lfc", "size"), mean_lfc=("lfc", "mean"),
            frac_up=("lfc", lambda s: float((s > 0).mean())))
        need = max(1, int(np.ceil(min_fraction * n_sheets)))
        keep = agg[(agg["n"] >= need) &
                   ((agg["frac_up"] >= 0.8) | (agg["frac_up"] <= 0.2))]
        for gene, r in keep.iterrows():
            rows.append({"signature": sig, "gene": gene,
                         "direction": "up" if r["mean_lfc"] > 0 else "down",
                         "source": f"Rosano2024_S4-S6_{arm}"})
            if arm in ("AI", "TAM"):
                rows.append({"signature": f"{sig}_{arm}", "gene": gene,
                             "direction": "up" if r["mean_lfc"] > 0 else "down",
                             "source": f"Rosano2024_S4-S6_{arm}"})
        n_up = int((keep["mean_lfc"] > 0).sum())
        summary.append({"signature": sig, "arm": arm, "sheets": n_sheets,
                        "genes_total": int(len(agg)),
                        "genes_kept": int(len(keep)),
                        "up": n_up, "down": int(len(keep)) - n_up})
        log.info(f"{sig:22s} ({arm:9s}) {n_sheets} sheet(s): "
                 f"{len(keep):5d}/{len(agg):6d} genes retained "
                 f"({n_up} up, {len(keep)-n_up} down)")

    new = pd.DataFrame(rows).drop_duplicates(
        subset=["signature", "gene", "direction"])

    banner(log, "GENE SYMBOL VALIDATION")
    new = validate_and_recover(new, "gene", log=log, strict=True)
    assert_no_dates(new)

    # drop genes assigned both directions within a signature
    amb = (new.groupby(["signature", "gene"])["direction"]
           .nunique().pipe(lambda s: s[s > 1]))
    if len(amb):
        log.info("")
        log.info(f"   dropping {len(amb)} genes assigned both directions")
        new = new[~new.set_index(["signature", "gene"]).index
                  .isin(set(amb.index))]
    log.info("")

    panel_path = cfg.paths.results / "signature_panel.csv"
    panel = pd.read_csv(panel_path)
    before = panel["signature"].nunique()
    panel = panel[~panel["signature"].isin(new["signature"].unique())]
    merged = pd.concat([panel, new], ignore_index=True)

    backup = cfg.paths.results / "signature_panel_before_wp2x.csv"
    if not backup.exists():
        pd.read_csv(panel_path).to_csv(backup, index=False)
        log.info(f"   backed up to {backup.name}")
    merged.to_csv(panel_path, index=False)

    banner(log, "MERGED PANEL")
    tbl = (merged.groupby(["signature", "direction"]).size()
           .rename("n_genes").reset_index())
    log.info(tbl.to_string(index=False))
    log.info("")
    log.info(f"   {before} -> {merged['signature'].nunique()} signatures, "
             f"{len(merged)} rows")
    log.info(f"   Written: {panel_path}")

    pd.DataFrame(summary).to_csv(
        cfg.paths.results / "wp2x_control_signatures.csv", index=False)

    huge = tbl.groupby("signature")["n_genes"].sum()
    huge = huge[huge > 5000]
    if len(huge):
        banner(log, "SIZE WARNING")
        for sig, n in huge.items():
            log.info(f"   {sig}: {int(n)} genes")
        log.info("")
        log.info("   A set this large is close to 'anything that changes under")
        log.info("   treatment' rather than a signature, and its matched null")
        log.info("   will be very narrow. Keep it as a size-extreme reference;")
        log.info("   do not treat it as comparable to a curated programme.")
        log.info("")
        report_large = huge.index.tolist()
    else:
        report_large = []

    banner(log, "WHAT THESE TEST")
    log.info("Culture_drift  — untreated cells, long-term culture, no dormancy.")
    log.info("                 Matched to TRADITIOM on cell line, laboratory and")
    log.info("                 DE pipeline; differs only in the biology encoded.")
    log.info("                 If it behaves the same in tissue, the compartment")
    log.info("                 effect is a property of in vitro derivation, not")
    log.info("                 of dormancy.")
    log.info("")
    log.info("Awakening      — the opposite state to dormancy, same experiment.")
    log.info("                 Identical behaviour would show the compartment")
    log.info("                 preference carries no information about state.")
    log.info("")
    log.info("Latency        — intermediate timepoint on the same arc.")
    log.info("")
    log.info("NEXT: re-run wp2b, wp2c, wp2d, wp2e, wp2f and wp3 so the new")
    log.info("signatures are scored alongside the existing panel.")

    report["oversized_signatures"] = report_large
    report["signatures_added"] = sorted(new["signature"].unique().tolist())
    report["panel_size"] = int(merged["signature"].nunique())
    report["summary"] = summary
    report["verdict"] = "OK" if len(new) else "EMPTY"
    with open(cfg.paths.results / "wp2x_controls.json", "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    return report
