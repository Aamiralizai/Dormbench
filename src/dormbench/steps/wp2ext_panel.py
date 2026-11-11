"""WP2ext — extend the signature panel with external gene sets.

WP2a yields three signatures from the Rosano tables. That is a comparison, not
a benchmark. This step adds:

  1. Curated dormancy marker sets typed from the primary literature. These are
     SMALL and MUST be checked against the source papers before publication --
     they are provided as a starting point, not an authority. Each carries its
     citation in the `source` column.

  2. MSigDB Hallmark sets fetched via decoupler/OmniPath (needs network):
     E2F_TARGETS and G2M_CHECKPOINT as proliferation controls,
     EPITHELIAL_MESENCHYMAL_TRANSITION for the EMT/stromal axis,
     TGF_BETA_SIGNALING and HYPOXIA as dormancy-adjacent programmes.

  3. Anything you drop into config/extra_signatures.csv, in the same tidy
     format: signature,gene,direction,source

Rationale for keeping (1) short and explicit: a benchmark whose gene lists were
silently reconstructed from memory is not reproducible. Small hand-entered sets
can be eyeballed against a paper in minutes; a 200-gene list cannot.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from ..config import Config
from ..logging_utils import banner, get_logger

# --- Curated marker sets. VERIFY against the cited source before publishing. ---
CURATED: dict[str, dict] = {
    "NR2F1_axis": {
        "source": "Sosa et al. Nat Commun 2015; Borgen et al. BCR 2018 — VERIFY",
        "up": ["NR2F1", "SOX9", "RARB", "ZFP281", "NANOG", "SOX2",
               "CDKN1A", "CDKN1B", "TGFB2", "BMP4", "BMP7"],
        "down": [],
    },
    "BHLHE41_dormancy": {
        "source": "Adam et al. / Aguirre-Ghiso p38-DEC2 axis — VERIFY",
        "up": ["BHLHE41", "MAPK14", "ATF6", "PERK", "EIF2AK3",
               "CDKN1B", "THBS1", "TGFB2"],
        "down": ["MAPK1", "MAPK3"],
    },
    "Quiescence_core": {
        "source": "Cell-cycle-exit markers, generic — VERIFY",
        "up": ["CDKN1A", "CDKN1B", "CDKN2A", "CDKN2B", "RB1",
               "MXI1", "FOXO3", "TP53"],
        "down": ["MKI67", "CCNB1", "CCNA2", "CDK1", "PCNA", "TOP2A",
                 "AURKA", "AURKB", "PLK1", "BUB1"],
    },
}

# Names are matched with or without a "HALLMARK_" prefix: OmniPath serves
# them bare (E2F_TARGETS) while MSigDB GMT files carry the prefix.
HALLMARK_WANTED = {
    "E2F_TARGETS": ("E2F_control", "up"),
    "G2M_CHECKPOINT": ("G2M_hallmark_control", "up"),
    "EPITHELIAL_MESENCHYMAL_TRANSITION": ("EMT_hallmark", "up"),
    "TGF_BETA_SIGNALING": ("TGFbeta_hallmark", "up"),
    "HYPOXIA": ("Hypoxia_hallmark", "up"),
    "P53_PATHWAY": ("P53_hallmark", "up"),
    "TNFA_SIGNALING_VIA_NFKB": ("TNFa_NFkB_hallmark", "up"),
}


def _norm(name: str) -> str:
    """Normalise a gene-set name for matching across sources."""
    n = str(name).strip().upper()
    return n[9:] if n.startswith("HALLMARK_") else n


def run(cfg: Config, skip_hallmark: bool = False) -> dict:
    log = get_logger("wp2ext", cfg.paths.logs)
    cfg.paths.ensure_outputs()

    report: dict = {
        "step": "wp2ext",
        "generated": datetime.now().isoformat(timespec="seconds"),
    }

    banner(log, "WP2ext — EXTEND THE SIGNATURE PANEL")

    panel_path = cfg.paths.results / "signature_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"{panel_path} not found. Run `dormbench wp2a` first.")
    panel = pd.read_csv(panel_path)
    log.info(f"Existing panel: {panel['signature'].nunique()} signatures, "
             f"{len(panel)} rows")
    log.info("")

    added: list[dict] = []

    # ---- 1. curated marker sets ----------------------------------------
    banner(log, "1. CURATED MARKER SETS")
    log.info("These are hand-entered and SMALL by design so you can verify them")
    log.info("against the source papers. Do not publish without checking.")
    log.info("")
    for sig, spec in CURATED.items():
        for direction in ("up", "down"):
            for g in spec[direction]:
                added.append({"signature": sig, "gene": g,
                              "direction": direction, "source": spec["source"]})
        log.info(f"   {sig:22s} {len(spec['up']):3d} up / {len(spec['down']):3d} down")
        log.info(f"      {spec['source']}")
    log.info("")

    # ---- 2. MSigDB Hallmark --------------------------------------------
    banner(log, "2. MSigDB HALLMARK (via decoupler/OmniPath)")
    if skip_hallmark:
        log.info("   skipped (--skip-hallmark)")
        report["hallmark"] = "skipped"
    else:
        try:
            import decoupler as dc

            log.info("   fetching ...")
            hm = dc.op.hallmark(organism="human")
            cols = list(hm.columns)
            col_set = next((c for c in cols if c.lower() in
                            ("geneset", "source", "collection", "term")), cols[0])
            col_gene = next((c for c in cols if c.lower() in
                             ("genesymbol", "target", "gene", "symbol")), cols[1])
            log.info(f"   {hm[col_set].nunique()} sets, columns "
                     f"({col_set}, {col_gene})")

            # build a normalised lookup so prefixed and bare names both match
            norm_map: dict[str, str] = {}
            for raw in hm[col_set].unique():
                norm_map.setdefault(_norm(raw), raw)

            found = 0
            for msig_name, (sig, direction) in HALLMARK_WANTED.items():
                raw = norm_map.get(_norm(msig_name))
                if raw is None:
                    log.info(f"   [!] {msig_name} not found "
                             f"(available e.g. {list(norm_map)[:3]})")
                    continue
                genes = hm.loc[hm[col_set] == raw, col_gene].astype(str).unique()
                if len(genes) == 0:
                    log.info(f"   [!] {msig_name} matched but empty")
                    continue
                for g in genes:
                    added.append({"signature": sig, "gene": g,
                                  "direction": direction, "source": "MSigDB_Hallmark"})
                log.info(f"   {sig:24s} {len(genes):4d} genes")
                found += 1
            report["hallmark"] = f"{found}/{len(HALLMARK_WANTED)} sets"
        except Exception as e:  # noqa: BLE001
            log.info(f"   [!] fetch failed: {type(e).__name__}: {e}")
            log.info("   Needs network access to OmniPath. Either re-run with")
            log.info("   network, or download the Hallmark GMT from MSigDB and")
            log.info("   append manually to config/extra_signatures.csv")
            report["hallmark"] = f"failed: {type(e).__name__}"
    log.info("")

    # ---- 3. user-supplied ----------------------------------------------
    banner(log, "3. USER-SUPPLIED SIGNATURES")
    extra_path = cfg.root / "config" / "extra_signatures.csv"
    if extra_path.exists():
        extra = pd.read_csv(extra_path)
        need = {"signature", "gene", "direction", "source"}
        if not need.issubset(extra.columns):
            log.info(f"   [!] {extra_path.name} must have columns: {sorted(need)}")
        else:
            added.extend(extra.to_dict("records"))
            log.info(f"   {extra_path.name}: {extra['signature'].nunique()} "
                     f"signatures, {len(extra)} rows")
    else:
        log.info(f"   none ({extra_path.name} not present)")
        log.info("   Create it to add your own; same columns as signature_panel.csv")
    log.info("")

    # ---- merge ----------------------------------------------------------
    banner(log, "4. MERGED PANEL")
    new = pd.DataFrame(added)
    merged = pd.concat([panel, new], ignore_index=True).drop_duplicates(
        subset=["signature", "gene", "direction"])

    # drop genes listed in both directions of the same signature
    amb = (merged.groupby(["signature", "gene"])["direction"]
           .nunique().pipe(lambda s: s[s > 1]))
    if len(amb):
        log.info(f"   [!] dropping {len(amb)} genes listed in both directions")
        merged = merged[~merged.set_index(["signature", "gene"]).index
                        .isin(set(amb.index))]

    summary = (merged.groupby(["signature", "direction"]).size()
               .rename("n_genes").reset_index()
               .sort_values(["signature", "direction"]))
    log.info("")
    log.info(summary.to_string(index=False))
    log.info("")
    log.info(f"   {merged['signature'].nunique()} signatures, {len(merged)} rows")

    backup = cfg.paths.results / "signature_panel_rosano_only.csv"
    if not backup.exists():
        panel.to_csv(backup, index=False)
        log.info(f"   Backed up original: {backup.name}")
    merged.to_csv(panel_path, index=False)
    log.info(f"   Written: {panel_path}")

    report["n_signatures"] = int(merged["signature"].nunique())
    report["n_rows"] = int(len(merged))
    report["panel"] = summary.to_dict("records")
    report["verdict"] = "OK" if merged["signature"].nunique() >= 6 else "THIN"

    log.info("")
    if report["verdict"] == "THIN":
        log.info("[!] Fewer than 6 signatures. Add more before treating this as")
        log.info("    a benchmark rather than a comparison.")
    else:
        log.info("Panel is benchmark-sized. Re-run `dormbench wp2b`.")

    out = cfg.paths.results / "wp2ext_panel.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {out}")

    return report
