"""TCGAER - fetch real clinical ER status (IHC) for TCGA-BRCA.

Review point 17. wp3 currently falls back to the PAM50 SUBTYPE column as an ER
proxy (Luminal = ER+, Basal = ER-). PAM50 luminal-versus-basal is not
interchangeable with immunohistochemical ER status: roughly 10-20% of PAM50
luminal tumours are ER-negative by IHC and a minority of basal-like tumours are
ER-positive, so the proxy misclassifies patients and the resulting analysis
cannot honestly be called ER-stratified.

The real annotation exists and is public. cBioPortal exposes the TCGA-BRCA
clinical attribute ER_STATUS_BY_IHC through its REST API, keyed by patient
barcode, which is the same key wp3 already joins on.

    dormbench tcgaer          # writes <tcga_dir>/tcga_er_ihc.csv
    dormbench wp3             # picks the file up automatically

If the file is present wp3 prefers it over any clinical column and over the
PAM50 proxy, and says so in its log. If it is absent, behaviour is unchanged.

Needs network access to www.cbioportal.org. No authentication: this is open
metadata for public data. Use --study to point at a different cBioPortal study
id (brca_tcga_pan_can_atlas_2018 is the other common one).
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from ..config import Config
from ..logging_utils import banner, get_logger

BASE = "https://www.cbioportal.org/api/studies"
ATTRIBUTE = "ER_STATUS_BY_IHC"


def _fetch(study: str, attribute: str, timeout: int = 90) -> list[dict]:
    """GET the clinical attribute for every patient in a cBioPortal study."""
    url = (f"{BASE}/{study}/clinical-data"
           f"?clinicalDataType=PATIENT&attributeId={attribute}"
           f"&projection=SUMMARY&pageSize=100000")
    try:
        import requests

        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except ImportError:
        import ssl
        import urllib.request

        try:
            import certifi

            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:  # noqa: BLE001
            ctx = None
        with urllib.request.urlopen(url, timeout=timeout, context=ctx) as fh:
            return json.loads(fh.read().decode("utf-8"))


def run(cfg: Config, study: str = "brca_tcga",
        attribute: str = ATTRIBUTE) -> dict:
    log = get_logger("tcgaer", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    report: dict = {"step": "tcgaer",
                    "generated": datetime.now().isoformat(timespec="seconds"),
                    "params": {"study": study, "attribute": attribute}}

    banner(log, "TCGAER - CLINICAL ER STATUS (IHC) FOR TCGA-BRCA")
    log.info("Replaces the PAM50 luminal/basal proxy with the actual")
    log.info("immunohistochemical ER call, which is what 'ER-stratified'")
    log.info("has to mean in the manuscript.")
    log.info("")

    if not cfg.paths.tcga_dir.exists():
        log.info(f"[X] TCGA directory does not exist: {cfg.paths.tcga_dir}")
        log.info("    Set paths.tcga_dir in config/config.yaml first.")
        report["verdict"] = "BLOCKED"
        return report

    log.info(f"   study     : {study}")
    log.info(f"   attribute : {attribute}")
    try:
        data = _fetch(study, attribute)
    except Exception as e:  # noqa: BLE001
        log.info(f"[X] fetch failed: {type(e).__name__}: {e}")
        log.info("    Check network access to www.cbioportal.org, or download")
        log.info("    data_clinical_patient.txt from the study page by hand and")
        log.info("    place it in the TCGA directory.")
        report["verdict"] = "BLOCKED"
        return report

    if not data:
        log.info("[X] the API returned no rows. The study id or attribute id")
        log.info("    may be wrong for this cBioPortal release.")
        report["verdict"] = "BLOCKED"
        return report

    df = pd.DataFrame([{"patient": d["patientId"], "er_status": d["value"]}
                       for d in data])
    df["er_status"] = df["er_status"].astype(str).str.strip().str.title()
    # Keep only calls that mean something. 'Indeterminate' and
    # 'Not Performed' are not ER-negative and must not be scored as such.
    usable = df[df["er_status"].isin(["Positive", "Negative"])].copy()

    counts = df["er_status"].value_counts().to_dict()
    log.info(f"   patients returned : {len(df)}")
    log.info(f"   value counts      : {counts}")
    log.info(f"   usable (Pos/Neg)  : {len(usable)}")
    log.info("")

    out = cfg.paths.tcga_dir / "tcga_er_ihc.csv"
    usable.to_csv(out, index=False)
    log.info(f"Written: {out}")
    log.info("")
    log.info("wp3 will now prefer this file over the PAM50 proxy. Re-run:")
    log.info("   dormbench wp3 --only TCGA")
    log.info("")

    report["n_patients"] = int(len(df))
    report["n_usable"] = int(len(usable))
    report["value_counts"] = {k: int(v) for k, v in counts.items()}
    report["path"] = str(out)
    report["verdict"] = ("ER_IHC_OBTAINED" if len(usable) >= 500
                         else "ER_IHC_PARTIAL")
    log.info(f"VERDICT: {report['verdict']}")

    with open(cfg.paths.results / "tcgaer.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    return report
