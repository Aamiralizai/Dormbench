"""WP2a — build the signature panel from Rosano supplementary tables.

Produces two artefacts that every later step depends on:

  results/signature_panel.csv    tidy: signature, gene, direction, source
  results/traditiom_expression.csv   1,550 dormancy genes x 51 samples

Signatures extracted
--------------------
S7  TRADITIOM_dormancy   821 up / 729 down. Consensus dormancy DEGs vs POT
                         across AI d30/60/90 and TAM d30/60.
S3  PreAdapted           311 up / 1,241 down (Hong et al. 2019 SWNE).
S3  G2M                  200 genes, proliferation control.

Deliberately NOT treated as a signature
---------------------------------------
S3 'GeoMx CK+' is 2,639 genes -- that is the set detected in CK+ segments,
i.e. the assay's gene universe, not a dormancy signature. Scoring it would be
meaningless. It is exported separately as a background gene set, which is the
correct null for enrichment tests against the GeoMx data.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from ..config import Config
from ..genes import assert_no_dates, validate_and_recover
from ..io import load_rosano_table
from ..logging_utils import banner, get_logger

# S7 sample groups, used to annotate the expression matrix
SAMPLE_GROUPS = {
    "POT": "T0",
    "U": "Untreated",
    "T7": "TAM_latency", "T14": "TAM_latency",
    "T30": "TAM_dormancy", "T60": "TAM_dormancy",
    "AI7": "AI_latency", "AI14": "AI_latency",
    "AI30": "AI_dormancy", "AI60": "AI_dormancy", "AI90": "AI_dormancy",
}


def _classify_sample(col: str) -> str:
    c = str(col)
    if "TEP" in c:
        return "TEP"
    if c.startswith(("T.", "AI.")):
        return "Awakening"
    for prefix, group in SAMPLE_GROUPS.items():
        if c.startswith(prefix) and not c.startswith(("T.", "AI.")):
            if prefix == "U" and not c[1:2].isdigit():
                continue
            return group
    return "Other"


def _clean_genes(series: pd.Series) -> list[str]:
    v = series.dropna().astype(str).str.strip()
    v = v[~v.str.lower().isin(["nan", "na", "", "none"])]
    return sorted(set(v))


def run(cfg: Config) -> dict:
    log = get_logger("wp2a", cfg.paths.logs)
    cfg.paths.ensure_outputs()

    report: dict = {
        "step": "wp2a",
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    rows: list[dict] = []

    banner(log, "WP2a — SIGNATURE PANEL FROM ROSANO SUPPLEMENTARY TABLES")

    # ---- S7: TRADITIOM dormancy signature + expression matrix ----------
    banner(log, "1. S7 — TRADITIOM dormancy signature")
    expr_parts = []
    for sheet, direction in (("Dormancy Up", "up"), ("Dormancy Down", "down")):
        d = load_rosano_table(cfg.paths.rosano_dir, 7, sheet=sheet)
        # row 0 is a long prose description; the real header is row 1
        d = d.rename(columns={d.columns[0]: "_desc"})
        header = d.iloc[0].tolist()
        body = d.iloc[1:].copy()
        body.columns = ["gene"] + [str(c) for c in header[1:]]
        body = body[body["gene"].notna()]
        genes = _clean_genes(body["gene"])
        log.info(f"   {sheet:16s} {len(genes):5d} genes")
        for g in genes:
            rows.append({"signature": "TRADITIOM_dormancy", "gene": g,
                         "direction": direction, "source": "Rosano2024_S7"})
        body["_direction"] = direction
        expr_parts.append(body)

    expr = pd.concat(expr_parts, ignore_index=True)
    sample_cols = [c for c in expr.columns if c not in ("gene", "_direction")]
    for c in sample_cols:
        expr[c] = pd.to_numeric(expr[c], errors="coerce")
    log.info(f"   expression matrix: {expr.shape[0]} genes x {len(sample_cols)} samples")

    groups = pd.Series({c: _classify_sample(c) for c in sample_cols})
    log.info("")
    log.info("   sample groups:")
    for g, n in groups.value_counts().items():
        members = list(groups.index[groups == g])
        log.info(f"      {g:16s} n={n:2d}  {members[:6]}{' ...' if n > 6 else ''}")
    report["expression_shape"] = [int(expr.shape[0]), len(sample_cols)]
    report["sample_groups"] = groups.value_counts().to_dict()

    # per-carbon-copy awakening samples are the scarce resource
    carbon = [c for c in sample_cols if c.startswith(("T.", "AI.")) and "TEP" not in c]
    log.info("")
    log.info(f"   awakening carbon copies with transcriptome: n={len(carbon)}")
    log.info(f"      {carbon}")
    report["carbon_copies"] = carbon
    log.info("")

    # ---- S3: pre-adapted, G2M, and the GeoMx background ----------------
    banner(log, "2. S3 — pre-adapted signature, G2M control, GeoMx background")
    s3 = load_rosano_table(cfg.paths.rosano_dir, 3)
    mapping = {
        "PA up": ("PreAdapted", "up", "Hong2019_via_Rosano_S3"),
        "PA down": ("PreAdapted", "down", "Hong2019_via_Rosano_S3"),
        "G2M": ("G2M_control", "up", "Rosano2024_S3"),
    }
    for col, (sig, direction, src) in mapping.items():
        if col not in s3.columns:
            log.info(f"   [!] column '{col}' not found in S3")
            continue
        genes = _clean_genes(s3[col])
        log.info(f"   {col:10s} -> {sig:16s} ({direction:4s}) {len(genes):5d} genes")
        for g in genes:
            rows.append({"signature": sig, "gene": g,
                         "direction": direction, "source": src})

    background = _clean_genes(s3["GeoMx CK+"]) if "GeoMx CK+" in s3.columns else []
    log.info("")
    log.info(f"   GeoMx CK+ : {len(background)} genes")
    log.info("   NOT a signature — this is the gene universe detected in CK+")
    log.info("   segments. Exported separately as the background set for")
    log.info("   enrichment tests; do not score it.")
    log.info("")

    # ---- write -----------------------------------------------------------
    banner(log, "3. OUTPUT")
    panel = pd.DataFrame(rows).drop_duplicates()

    banner(log, "GENE SYMBOL VALIDATION")
    log.info("Spreadsheet software converts SEPT/MARCH/DEC gene symbols to")
    log.info("dates. The corruption is present in the published supplementary")
    log.info("workbooks, so recovery is required rather than optional.")
    log.info("")
    panel = validate_and_recover(panel, "gene", log=log, strict=True)
    assert_no_dates(panel)
    log.info("   panel clean")

    # a gene appearing in both directions of one signature is ambiguous
    dupes = (panel.groupby(["signature", "gene"])["direction"]
             .nunique().pipe(lambda s: s[s > 1]))
    if len(dupes):
        log.info(f"   [!] {len(dupes)} genes appear in BOTH directions of a "
                 "signature; dropping them as ambiguous")
        bad = set(dupes.index)
        panel = panel[~panel.set_index(["signature", "gene"]).index.isin(bad)]
    report["ambiguous_dropped"] = int(len(dupes))

    panel_path = cfg.paths.results / "signature_panel.csv"
    panel.to_csv(panel_path, index=False)

    expr_out = expr.rename(columns={"_direction": "direction"})
    expr_path = cfg.paths.results / "traditiom_expression.csv"
    expr_out.to_csv(expr_path, index=False)

    bg_path = cfg.paths.results / "geomx_background_genes.txt"
    bg_path.write_text("\n".join(background), encoding="utf-8")

    summary = (panel.groupby(["signature", "direction"]).size()
               .rename("n_genes").reset_index())
    log.info("")
    log.info(summary.to_string(index=False))
    report["panel"] = summary.to_dict("records")
    report["n_signatures"] = int(panel["signature"].nunique())

    log.info("")
    log.info(f"   Written: {panel_path}")
    log.info(f"   Written: {expr_path}")
    log.info(f"   Written: {bg_path}")

    # ---- what is still missing ------------------------------------------
    banner(log, "4. STILL NEEDED FOR THE BENCHMARK")
    log.info("A benchmark of 3 signatures is thin. Add from source papers:")
    log.info("   - NR2F1 / BHLHE41 dormancy markers (Sosa et al.)")
    log.info("   - 22-gene quiescence signature")
    log.info("   - EMT/stromal late-recurrence signature")
    log.info("   - Hallmark E2F_TARGETS as a second proliferation control")
    log.info("")
    log.info("Append them to signature_panel.csv in the same tidy format:")
    log.info("   signature,gene,direction,source")
    report["verdict"] = "OK"

    out = cfg.paths.results / "wp2a_signature_panel.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info("")
    log.info(f"Written: {out}")

    return report
