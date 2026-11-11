"""Command-line interface.

    dormbench doctor          check config and data availability
    dormbench all             run the whole pipeline in dependency order
    dormbench report          write one compact summary of everything so far
    dormbench wp0             histone PTM erosion analysis (Rosano S8)
    dormbench wp1             inspect the single-cell reference
    dormbench wp2a            build the signature panel
    dormbench wp2ext          extend the panel with external signatures
    dormbench wp2x            matched control signatures from Rosano S4-S6
    dormbench wp2b            score signatures across compartments (Figure 1)
    dormbench wp2c            cell-cycle deconfounding (Figure 2)
    dormbench wp2d            direction-resolved attribution (Figure 3)
    dormbench wp2e            size/expression-matched null control
    dormbench wp2y            does marker content explain attribution?
    dormbench wp2f            leading-edge drivers and enrichment
    dormbench wp2g            in vitro validation of the leading edge
    dormbench wp2h            hypoxia vs endothelial identity
    dormbench wp2i            CCLE cross-check in cancer cell lines
    dormbench wp2m            cell-cycle filtering multiverse
    dormbench wp2n            coherent-module (Hallmark) null
    dormbench wp2r            rare TRADITIOM-high malignant-cell state
    dormbench compare         which manuscript numbers the review steps change
    dormbench wp3             bulk cohorts: composition vs dormancy score
    dormbench wp4             sensitivity: leave-one-gene / leave-one-donor
    dormbench wp4s            survival: does dormancy add to CTS5?
    dormbench wp5             dormancy vs published TCGA hypoxia scores
    dormbench wp6             POETIC paired on-treatment biopsies
    dormbench wp7             compartment-invariant signature trade-off
    dormbench tcgamap         resolve GDC file UUIDs to TCGA barcodes
    dormbench list            show registered steps

Also runnable without installing:  python -m dormbench doctor
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .config import ConfigError, load_config
from .logging_utils import (banner, end_session, get_logger, session_path,
                            start_session)


def cmd_doctor(args) -> int:
    """Verify config, data locations, and dependencies before any real work."""
    log = get_logger("doctor")
    banner(log, "DORMBENCH DOCTOR")

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        log.info(f"[X] {e}")
        return 1

    log.info(f"Config     : {args.config or 'config/config.yaml'}")
    log.info(f"Package    : {cfg.root}")
    log.info(f"Data root  : {cfg.paths.data_root}")
    log.info("")

    log.info("DATA DIRECTORIES")
    ok = True
    checks = [
        ("data_root", cfg.paths.data_root, True),
        ("atlas (GSE176078)", cfg.paths.atlas_dir, True),
        ("METABRIC", cfg.paths.metabric_dir, True),
        ("SCAN-B (GSE96058)", cfg.paths.scanb_dir, False),
        ("Rosano tables", cfg.paths.rosano_dir, False),
        ("DepMap/CCLE", cfg.paths.depmap_dir, False),
        ("TCGA-BRCA", cfg.paths.tcga_dir, False),
        ("POETIC control (GSE105777)", cfg.paths.poetic_sampling_control_dir, False),
        ("Extended letrozole (GSE111563)", cfg.paths.extended_letrozole_dir, False),
    ]
    for name, path, required in checks:
        if path.exists():
            n = sum(1 for _ in path.rglob("*") if _.is_file())
            log.info(f"   [OK] {name:22s} {path}  ({n} files)")
        else:
            mark = "[X]" if required else "[!]"
            log.info(f"   {mark}  {name:22s} MISSING: {path}")
            if required:
                ok = False
    log.info("")

    log.info("KEY FILES")
    from . import io as dio

    probes = [
        ("atlas .h5ad", lambda: dio.find_atlas_h5ad(cfg.paths.atlas_dir)),
        ("METABRIC expression", lambda: dio._first_match(
            cfg.paths.metabric_dir, dio.METABRIC_EXPR_PATTERNS, "expr")),
        ("METABRIC clinical", lambda: dio._first_match(
            cfg.paths.metabric_dir, ["data_clinical_patient.txt"], "clinical")),
        ("SCAN-B expression", lambda: dio.load_scanb(cfg.paths.scanb_dir)["expr_path"]),
    ]
    for name, fn in probes:
        try:
            p = fn()
            size = p.stat().st_size / 1e6
            log.info(f"   [OK] {name:22s} {p.name}  ({size:,.1f} MB)")
        except Exception as e:  # noqa: BLE001
            log.info(f"   [!]  {name:22s} {type(e).__name__}: {str(e).splitlines()[0]}")
    log.info("")

    log.info("DEPENDENCIES")
    for mod in ("numpy", "pandas", "scipy", "yaml", "anndata", "scanpy",
                "matplotlib", "decoupler"):
        try:
            __import__(mod)
            log.info(f"   [OK] {mod}")
        except ImportError:
            log.info(f"   [!]  {mod} not installed")
    log.info("")

    log.info("OUTPUT DIRECTORIES")
    cfg.paths.ensure_outputs()
    for p in (cfg.paths.results, cfg.paths.figures, cfg.paths.logs, cfg.paths.processed):
        log.info(f"   [OK] {p}")

    log.info("")
    log.info("READY" if ok else "NOT READY — fix the [X] items above")
    return 0 if ok else 1


PIPELINE = ["wp0", "wp1", "wp2a", "wp2ext", "wp2x", "wp2b", "wp2c", "wp2d",
            "wp2e", "wp2m", "wp2n", "wp2r", "wp2y", "wp2f", "wp2g", "wp2h",
            "wp2i", "wp3u", "wp3", "wp4", "wp4s", "wp5", "wp6", "wp7", "wp7n", "wp8",
            "wp8p", "wp9", "wp9b", "wp10"]

# Steps added in v0.2.0 in response to the hostile-review stress test.
# wp3r is NOT in PIPELINE: it needs a second atlas that may not be present,
# and tcgaer needs network, so both are run explicitly.
REVIEW_STEPS = ["wp2m", "wp2n", "wp2r", "wp7n", "wp8p"]


def cmd_all(args) -> int:
    """Run the full pipeline in dependency order.

    Order matters in one place: wp2a rewrites signature_panel.csv from the
    Rosano tables, and wp2ext extends it. Running wp2a alone would silently
    drop the panel back to three signatures and change every downstream result
    without raising an error.
    """
    from .steps import REGISTRY

    cfg = load_config(args.config)
    cfg.paths.ensure_outputs()
    sess = start_session(cfg.paths.logs, "session")
    log = get_logger("all", cfg.paths.logs)
    banner(log, "FULL PIPELINE")
    log.info(f"started  : {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info(f"package  : {cfg.root}")
    log.info(f"data root: {cfg.paths.data_root}")
    log.info(f"session log: {sess}")

    steps = [s for s in PIPELINE if s in REGISTRY]
    if args.From:
        if args.From not in steps:
            log.info(f"[X] unknown step: {args.From}")
            return 1
        steps = steps[steps.index(args.From):]
    if args.skip:
        steps = [s for s in steps if s not in set(args.skip)]

    log.info("Steps: " + " -> ".join(steps))
    log.info("")

    results = {}
    for name in steps:
        desc, fn = REGISTRY[name]
        log.info("")
        log.info("#" * 74)
        log.info(f"# {name}  —  {desc}")
        log.info("#" * 74)
        try:
            results[name] = fn(cfg)
        except Exception as e:  # noqa: BLE001
            log.info("")
            log.info(f"[X] {name} failed: {type(e).__name__}: {e}")
            if not args.keep_going:
                log.info("    Stopping. Use --keep-going to continue past failures.")
                return 2
            results[name] = {"verdict": f"FAILED: {type(e).__name__}"}

    banner(log, "PIPELINE SUMMARY")
    log.info("   Each label below is that step's SCIENTIFIC VERDICT (its")
    log.info("   outcome), not a pass/fail status. Words like MIXED, PARTIAL,")
    log.info("   DEPENDENT, SENSITIVE, UNRESOLVED and CONFOUNDED are findings.")
    log.info("   A step that actually errored is shown as 'FAILED: <Error>'.")
    log.info("")
    n_error = 0
    for name in steps:
        v = results.get(name, {}).get("verdict", "-")
        if isinstance(v, str) and v.startswith("FAILED:"):
            n_error += 1
        log.info(f"   {name:8s} {v}")
    log.info("")
    if n_error:
        log.info(f"   {n_error} step(s) errored (see 'FAILED:' above); the rest")
        log.info("   completed and emitted a verdict.")
    else:
        log.info("   All steps completed successfully (no errors); every label")
        log.info("   above is a scientific outcome.")
    log.info("")
    log.info(f"Full log written to: {sess}")
    log.info("Attach that single file rather than pasting console output.")
    end_session()
    return 0


def cmd_report(args) -> int:
    """Write one compact, shareable summary of every step that has run.

    The step logs are verbose by design -- wp2f alone prints 30 genes per
    signature. This reads the results/*.json instead and emits verdicts plus
    headline numbers only, which is what is actually useful when someone else
    needs to see where a run got to.
    """
    import json

    cfg = load_config(args.config)
    cfg.paths.ensure_outputs()
    log = get_logger("report")

    order = ["wp0_histone_erosion", "wp1_reference_inventory",
             "wp2a_signature_panel", "wp2ext_panel", "wp2x_controls", "wp2b_compartment",
             "wp2c_deconfounding", "wp2d_direction", "wp2e_null",
             "wp2m_multiverse", "wp2n_module_null", "wp2r_rare_state",
             "wp2y_marker_overlap", "wp2f_drivers", "wp2g_invitro", "wp2h_mechanism", "wp2i_ccle",
             "wp3u_universe", "wp3_bulk", "wp4_sensitivity", "wp4s_survival",
             "wp5_hypoxia",
             "wp6_paired", "wp6_poetic", "wp7_invariant", "wp8_donor_level", "wp9_purity", "wp9b_official", "wp10_response"]

    lines = [
        "# dormbench run report",
        f"generated : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"package   : {cfg.root}",
        f"data root : {cfg.paths.data_root}",
        "",
        "## verdicts",
        "",
    ]

    found = {}
    for stem in order:
        f = cfg.paths.results / f"{stem}.json"
        if not f.exists():
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                found[stem] = json.load(fh)
        except Exception:  # noqa: BLE001
            continue

    if not found:
        log.info("No results/*.json found. Run the pipeline first.")
        return 1

    for stem, d in found.items():
        step = d.get("step", stem.split("_")[0])
        lines.append(f"- **{step}** ({stem}): `{d.get('verdict', '-')}`")
    lines.append("")

    # headline numbers, hand-picked per step so the file stays short
    HEADLINE = {
        "wp0_histone_erosion": ["replicating_marks", "concordance"],
        "wp1_reference_inventory": ["n_obs", "n_vars", "counts_location",
                                    "gene_id_type", "warnings"],
        "wp2a_signature_panel": ["panel", "expression_shape", "carbon_copies"],
        "wp2ext_panel": ["n_signatures", "hallmark"],
        "wp2x_controls": ["signatures_added", "panel_size", "summary"],
        "wp2b_compartment": ["n_cancer_top", "n_dormancy_scores"],
        "wp2c_deconfounding": ["g2m_variance_explained"],
        "wp2d_direction": ["per_signature", "mean_corr_g2m_up",
                           "mean_corr_g2m_down", "n_cancer_dominant"],
        "wp2e_null": ["n_beyond_null", "underpowered", "results"],
        "wp2f_drivers": ["concentration"],
        "wp2y_marker_overlap": ["rho_overlap_auroc_literature",
                                "rho_overlap_auroc_atlas",
                                "n_moved_toward_null_literature"],
        "wp2g_invitro": ["median_dorm_vs_T0", "median_untr_vs_T0",
                         "median_dorm_vs_untr", "wilcoxon_p"],
        "wp2h_mechanism": ["hif_median", "endo_median", "coverage"],
        "wp2i_ccle": ["n_silent", "n_tested", "pot_vs_ccle",
                      "median_rho_hypoxia"],
        "wp3_bulk": ["cohorts", "e2f_r_stroma", "traditiom_r_stroma",
                     "null_baseline_abs_r", "n_beyond_null",
                     "n_signatures_tested", "cross_cohort_consistency"],
        "wp4_sensitivity": ["max_gene_delta", "max_donor_delta",
                            "n_gene_fragile", "n_donor_fragile", "n_tested"],
        "wp5_hypoxia": ["hypoxia_hallmark_r", "e2f_r", "traditiom_r",
                        "traditiom_partial_r", "n_patients"],
        "wp9_purity": ["verdict", "mean_r_independent_stromal",
                       "mean_partial_r2_purity", "mean_partial_r2_proliferation"],
        "wp10_response": ["verdict", "label_source", "validation"],
        "wp8_donor_level": ["donors_total", "donors_by_subtype", "verdict",
                            "ER_dormancy_consistent_compartments",
                            "ER_dormancy_significant_compartments"],
        "wp7_invariant": ["verdict", "tradeoff"],
        "wp6_paired": ["n_pairs", "n_treated", "median_r_with_stroma_change",
                       "dormancy_fraction_retained",
                       "control_fraction_retained", "by_contrast"],
        "wp6_poetic": ["n_pairs", "n_treated", "median_r_with_stroma_change",
                       "dormancy_fraction_retained",
                       "control_fraction_retained"],
    }

    lines += ["## key numbers", ""]
    for stem, d in found.items():
        keys = [k for k in HEADLINE.get(stem, []) if k in d]
        if not keys:
            continue
        lines.append(f"### {stem}")
        for k in keys:
            v = d[k]
            if isinstance(v, list) and v and isinstance(v[0], dict):
                lines.append(f"- {k}:")
                for row in v[:14]:
                    short = {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                             for kk, vv in row.items()}
                    lines.append(f"    - {short}")
                if len(v) > 14:
                    lines.append(f"    - ... {len(v) - 14} more rows")
            elif isinstance(v, dict):
                # drop bulky member lists; the report is a summary, not a dump
                trimmed = {kk: vv for kk, vv in v.items()
                           if not (isinstance(vv, list) and len(vv) > 8)}
                omitted = [kk for kk in v if kk not in trimmed]
                txt = json.dumps(trimmed, default=str)[:500]
                if omitted:
                    txt += f"  (omitted long lists: {', '.join(omitted)})"
                lines.append(f"- {k}: {txt}")
            else:
                lines.append(f"- {k}: {v}")
        lines.append("")

    figs = sorted(cfg.paths.figures.glob("*.svg"))
    lines += ["## figures", ""] + [f"- {f.name}" for f in figs] + [""]

    sess = sorted(cfg.paths.logs.glob("session_*.log"))
    if sess:
        lines += ["## logs", "",
                  f"- latest session log: `{sess[-1].name}`", ""]

    out = cfg.paths.results / "RUN_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")

    PREVIEW = 40
    log.info("\n".join(lines[:PREVIEW]))
    if len(lines) > PREVIEW:
        log.info("")
        log.info(f"... console preview truncated at {PREVIEW} of {len(lines)} "
                 "lines ...")
    log.info("")
    log.info(f"FULL REPORT WRITTEN TO: {out}")
    log.info(f"({len(lines)} lines) Attach that FILE, not this console output.")
    return 0


def cmd_compare(args) -> int:
    """Old vs new: which manuscript numbers the review-response steps change."""
    from .compare import run as compare_run

    cfg = load_config(args.config)
    compare_run(cfg, get_logger("compare"))
    return 0


def cmd_list(args) -> int:
    from .steps import REGISTRY

    log = get_logger("list")
    banner(log, "REGISTERED STEPS")
    for key, (desc, _) in REGISTRY.items():
        log.info(f"   {key:6s} {desc}")
    return 0


def cmd_step(args) -> int:
    from .steps import REGISTRY

    cfg = load_config(args.config)
    # wp6-poetic is a convenience alias for wp6 --cohort poetic --tag poetic
    command = args.command
    if command == "wp6-poetic":
        command = "wp6"
        args.cohort = "poetic"
        if not getattr(args, "tag", None):
            args.tag = "poetic"
        args.no_merge = True
    _, fn = REGISTRY[command]
    kwargs = {}
    if getattr(args, "h5ad", None):
        kwargs["h5ad"] = args.h5ad
    if getattr(args, "top", None):
        kwargs["top"] = args.top
    if getattr(args, "no_figure", False):
        kwargs["make_figure"] = False
    if getattr(args, "skip_hallmark", False):
        kwargs["skip_hallmark"] = True
    if getattr(args, "signature", None):
        kwargs["signature"] = args.signature
    if getattr(args, "no_merge", False):
        kwargs["no_merge"] = True
    if getattr(args, "tag", None):
        kwargs["tag"] = args.tag
    if getattr(args, "n_boot", None):
        kwargs["n_boot"] = args.n_boot
    if getattr(args, "method", None) and args.command == "wp8":
        kwargs["method"] = args.method
    if getattr(args, "top_n", None):
        kwargs["top_n"] = args.top_n
    for k in ("n_boot", "min_genes"):
        if getattr(args, k, None):
            kwargs[k] = getattr(args, k)
    if getattr(args, "common_genes", False):
        kwargs["common_genes"] = True
    if getattr(args, "series_dir", None):
        kwargs["series_dir"] = args.series_dir
    if getattr(args, "cohort", None):
        kwargs["cohort"] = args.cohort
    if getattr(args, "only", None):
        kwargs["only"] = args.only
    if getattr(args, "n_null", None):
        kwargs["n_null"] = args.n_null
    if getattr(args, "n_markers", None):
        kwargs["n_markers"] = args.n_markers
    if getattr(args, "n_perm", None):
        kwargs["n_perm"] = args.n_perm
    for k in ("study", "atlas_dir", "celltype_col", "donor_col",
              "subtype_col", "size_tolerance", "n_permute"):
        if getattr(args, k, None) is not None:
            kwargs[k] = getattr(args, k)
    if getattr(args, "inspect", False):
        kwargs["inspect"] = True
    # wp2r: tail fraction and the optional marker panel
    if getattr(args, "threshold", None) is not None:
        kwargs["q"] = args.threshold
    if getattr(args, "no_stromal", False):
        kwargs["stromal"] = False
    # wp4s: landmark survival options
    if getattr(args, "endocrine_only", False):
        kwargs["endocrine_only"] = True
    if getattr(args, "landmark_years", None) is not None:
        kwargs["landmark_years"] = args.landmark_years
    if getattr(args, "horizon_years", None) is not None:
        kwargs["horizon_years"] = args.horizon_years
    if getattr(args, "endpoint", None) is not None:
        kwargs["endpoint"] = args.endpoint
    # --all-subtypes is the negation of the steps' er_only default
    if getattr(args, "all_subtypes", False):
        kwargs["er_only"] = False
    for k in ("subsample", "er_only", "method"):
        if getattr(args, k, None):
            kwargs[k] = getattr(args, k)
    result = fn(cfg, **kwargs)
    return 0 if result.get("verdict") != "BLOCKED" else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dormbench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check config, data and dependencies")

    a = sub.add_parser("all", help="run the full pipeline in order")
    a.add_argument("--from", dest="From", default=None,
                   help="start at this step instead of the beginning")
    a.add_argument("--skip", nargs="*", default=None, help="steps to skip")
    a.add_argument("--keep-going", action="store_true",
                   help="continue past a failing step")
    sub.add_parser("list", help="list registered steps")
    sub.add_parser("report", help="write one compact shareable summary")
    sub.add_parser("compare", help="old vs new numbers for the review-response "
                   "steps; says which manuscript claims must change")

    w0 = sub.add_parser("wp0", help="histone PTM erosion analysis (Rosano S8)")
    w0.add_argument("--no-figure", action="store_true", help="skip the SVG figure")

    sub.add_parser("wp2a", help="build signature panel from Rosano tables")

    sub.add_parser("wp2x", help="matched control signatures from Rosano S4-S6")

    w2e = sub.add_parser("wp2ext", help="extend panel with external signatures")
    w2e.add_argument("--skip-hallmark", action="store_true",
                     help="skip the MSigDB fetch (no network)")

    sub.add_parser("wp2c", help="cell-cycle deconfounding (Figure 2)")
    sub.add_parser("wp2d", help="direction-resolved attribution (Figure 3)")

    sub.add_parser("wp2h", help="hypoxia vs endothelial identity")
    sub.add_parser("wp2i", help="CCLE cross-check in cancer cell lines")

    sub.add_parser("wp5", help="dormancy vs published TCGA hypoxia scores")
    sub.add_parser("wp9", help="independent purity validation (ESTIMATE, MCP-counter)")
    sub.add_parser("wp9b", help="official ESTIMATE/MCP-counter validation (needs R script output)")
    sub.add_parser("wp10", help="dormant vs acquired-resistant longitudinal model")

    w8 = sub.add_parser("wp8",
                        help="donor-level inference, ER-positive primary")
    w8.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap replicates for donor-level CIs")
    w8.add_argument("--method", default="AUCell",
                    help="scoring method column prefix (AUCell or scanpy)")

    w6 = sub.add_parser("wp6", help="paired on-treatment biopsies "
                        "(default: extended-letrozole cohort)")
    w6.add_argument("--series-dir", default=None,
                    help="folder holding a GEO series matrix (overrides "
                         "--cohort)")
    w6.add_argument("--cohort", choices=["extended", "poetic"],
                    default="extended",
                    help="which cohort's default directory to use: "
                         "'extended' -> extended_letrozole_dir (GSE111563), "
                         "'poetic' -> poetic_sampling_control_dir (GSE105777, "
                         "analysed with --no-merge)")
    w6.add_argument("--no-merge", action="store_true",
                    help="analyse only the chosen directory; do not pool "
                         "sibling accessions (implied by --cohort poetic)")
    w6.add_argument("--tag", default=None,
                    help="name the outputs wp6_<tag>_*.csv so two cohorts can "
                         "be analysed without overwriting each other")
    # convenience alias: dormbench wp6-poetic
    sub.add_parser("wp6-poetic", help="POETIC randomised sampling control "
                   "(shortcut for wp6 --cohort poetic --tag poetic)")
    sub.add_parser("wp7", help="compartment-invariant signature trade-off")
    sub.add_parser("tcgamap", help="resolve GDC UUIDs to TCGA barcodes (network)")

    # --- review-response steps (v0.2.0) -------------------------------
    te = sub.add_parser("tcgaer", help="clinical ER status (IHC) for TCGA "
                        "(network); replaces the PAM50 proxy in wp3")
    te.add_argument("--study", default="brca_tcga",
                    help="cBioPortal study id (default brca_tcga)")

    w8p = sub.add_parser("wp8p", help="true donor x compartment raw-count "
                         "pseudobulk (review point 12)")
    w8p.add_argument("--n-boot", type=int, default=2000)
    w8p.add_argument("--all-subtypes", dest="all_subtypes",
                     action="store_true",
                     help="do not restrict to ER+ donors")
    w8p.add_argument("--tag", default=None)

    w2m = sub.add_parser("wp2m", help="cell-cycle filtering multiverse "
                         "(review point 9)")
    w2m.add_argument("--method", default="AUCell",
                     help="score column prefix (AUCell or scanpy)")

    w2n = sub.add_parser("wp2n", help="coherent-module Hallmark null "
                         "(review point 7)")
    w2n.add_argument("--size-tolerance", dest="size_tolerance", type=float,
                     default=0.5,
                     help="admit Hallmark sets within this size ratio")

    w2r = sub.add_parser("wp2r", help="rare TRADITIOM-high malignant-cell "
                         "state (review point 2.1)")
    w2r.add_argument("--method", default="AUCell",
                     help="score column prefix (AUCell or scanpy)")
    w2r.add_argument("--threshold", type=float, default=None,
                     help="single tail fraction to run (default: the full grid "
                          "1/5/10/20%%)")
    w2r.add_argument("--n-null", dest="n_null", type=int, default=200,
                     help="random same-size subsets per donor and threshold")
    w2r.add_argument("--all-subtypes", dest="all_subtypes",
                     action="store_true",
                     help="do not restrict to ER+ donors")
    w2r.add_argument("--no-stromal", dest="no_stromal", action="store_true",
                     help="skip the stromal-marker contamination panel (avoids "
                          "reading the atlas)")

    w7n = sub.add_parser("wp7n", help="fully nested compartment-neutral "
                         "redesign (review points 25-26)")
    w7n.add_argument("--n-permute", dest="n_permute", type=int, default=20,
                     help="permuted-label runs for the log-loss floor")

    w3r = sub.add_parser("wp3r", help="donor-level replication in a second "
                         "atlas (review point 4)")
    w3r.add_argument("--atlas-dir", dest="atlas_dir", default=None,
                     help="directory holding the replication atlas .h5ad")
    w3r.add_argument("--tag", default="pal", help="name for the output files")
    w3r.add_argument("--celltype-col", dest="celltype_col", default=None)
    w3r.add_argument("--donor-col", dest="donor_col", default=None)
    w3r.add_argument("--subtype-col", dest="subtype_col", default=None)
    w3r.add_argument("--all-subtypes", dest="all_subtypes",
                     action="store_true",
                     help="do not restrict to ER+ donors")
    w3r.add_argument("--inspect", action="store_true",
                     help="print the cell metadata and stop, so the columns "
                          "can be chosen before any long run")
    w3r.add_argument("--n-boot", type=int, default=2000)

    w4 = sub.add_parser("wp4", help="sensitivity analyses (Figure 8)")
    w4.add_argument("--n-boot", type=int, default=200)
    w4.add_argument("--min-genes", type=int, default=20)

    w4s = sub.add_parser("wp4s", help="survival: does dormancy add to CTS5 for "
                         "late recurrence? (Figure 14)")
    w4s.add_argument("--cohort", default="METABRIC", choices=["METABRIC", "TCGA"],
                     help="METABRIC (primary, CTS5 benchmark) or TCGA "
                          "(reduced age+stage replication)")
    w4s.add_argument("--endocrine-only", dest="endocrine_only",
                     action="store_true",
                     help="restrict to patients recorded as receiving adjuvant "
                          "hormone therapy")
    w4s.add_argument("--landmark-years", dest="landmark_years", type=float,
                     default=5.0, help="disease-free landmark in years")
    w4s.add_argument("--horizon-years", dest="horizon_years", type=float,
                     default=10.0,
                     help="absolute year on which calibration/AUC are reported")
    w4s.add_argument("--endpoint", default="OS", choices=["OS", "DFS"],
                     help="TCGA endpoint; DFS carries too few events in this "
                          "clinical export to support a 5-year landmark")
    w4s.add_argument("--tag", default=None,
                     help="name the outputs wp4s_<tag>_* so a sensitivity run "
                          "does not overwrite the primary analysis")

    sub.add_parser("wp3u", help="build the common gene universe for wp3")

    w3 = sub.add_parser("wp3", help="bulk cohorts: composition vs score (Figure 7)")
    w3.add_argument("--n-markers", type=int, default=50,
                    help="marker genes per compartment (default 50)")
    w3.add_argument("--n-boot", type=int, default=200,
                    help="patient bootstrap replicates (default 200)")
    w3.add_argument("--common-genes", action="store_true",
                    help="restrict all cohorts to results/wp3_common_genes.txt")
    w3.add_argument("--only", default=None,
                    help="run one cohort only: METABRIC, SCAN-B or TCGA")
    w3.add_argument("--n-null", type=int, default=100,
                    help="matched random gene sets per signature (default 100)")

    w2g = sub.add_parser("wp2g", help="in vitro validation of leading edge (Figure 6)")
    w2g.add_argument("--signature", default="TRADITIOM_dormancy")

    sub.add_parser("wp2y", help="marker-content explanation of attribution")

    w2f = sub.add_parser("wp2f", help="leading-edge drivers and enrichment (Figure 5)")
    w2f.add_argument("--top-n", type=int, default=30,
                     help="leading-edge genes to report per signature")

    w2e2 = sub.add_parser("wp2e", help="size/expression-matched null control")
    w2e2.add_argument("--n-perm", type=int, default=1000,
                      help="permutations per signature (default 1000)")

    w2b = sub.add_parser("wp2b", help="score signatures across compartments (Figure 1)")
    w2b.add_argument("--subsample", type=int, default=None,
                     help="use N random cells (fast test run)")
    w2b.add_argument("--er-only", action="store_true", help="restrict to ER+ tumours")
    w2b.add_argument("--method", choices=["both", "aucell", "scanpy"], default="both")

    w1 = sub.add_parser("wp1", help="inspect the single-cell reference")
    w1.add_argument("--h5ad", default=None, help="override atlas path")
    w1.add_argument("--top", type=int, default=25, help="categories to print")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return cmd_doctor(args)
    if args.command == "all":
        return cmd_all(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "compare":
        return cmd_compare(args)
    if args.command == "list":
        return cmd_list(args)
    return cmd_step(args)


if __name__ == "__main__":
    sys.exit(main())
