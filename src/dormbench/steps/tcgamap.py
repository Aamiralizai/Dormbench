"""TCGAMAP — resolve GDC file UUIDs to TCGA barcodes via the GDC API.

A GDC download names each folder by file UUID; the barcode and sample type live
only in the sample sheet exported from the cart. If that sheet is missing, or
was exported from a different cart than the files (easy to do, and it fails
silently because UUIDs simply do not match), the assembled matrix cannot be
joined to any clinical data.

This queries the GDC files endpoint for the UUIDs actually present on disk and
writes a sample sheet in the format the assembler expects. It cannot fetch the
wrong project, because the UUIDs come from the folders themselves.

    dormbench tcgamap
    # then delete tcga_expression_matrix.parquet and re-run wp3 / wp5

Needs network access to api.gdc.cancer.gov. No authentication: this is open
metadata for public data.
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request
from datetime import datetime

import pandas as pd

from ..config import Config
from ..logging_utils import banner, get_logger

ENDPOINT = "https://api.gdc.cancer.gov/files"
FIELDS = ",".join([
    "file_id", "file_name",
    "cases.project.project_id",
    "cases.submitter_id",
    "cases.samples.submitter_id",
    "cases.samples.sample_type",
    "cases.samples.tissue_type",
])


def _query_requests(uuids: list[str], timeout: int = 60) -> list[dict]:
    """Same query via requests, which ships its own CA bundle."""
    import requests

    payload = {
        "filters": {"op": "in",
                    "content": {"field": "file_id", "value": uuids}},
        "fields": FIELDS, "format": "JSON", "size": str(len(uuids) + 10),
    }
    r = requests.post(ENDPOINT, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["data"]["hits"]


def _ssl_context(log=None):
    """Build an SSL context that works on Anaconda/Windows.

    Python on Windows loads CAs from the system certificate store, and on some
    Anaconda installs that parse fails with
    `ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]`. certifi's bundle sidesteps it
    entirely and is already present in most scientific environments.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        if log:
            log.info("   [!] certifi unavailable; using the system CA store")
            log.info("       If this fails with an ASN1 error, run:")
            log.info("       pip install certifi")
        return None


def _query(uuids: list[str], timeout: int = 60, ctx=None) -> list[dict]:
    payload = {
        "filters": {"op": "in",
                    "content": {"field": "file_id", "value": uuids}},
        "fields": FIELDS,
        "format": "JSON",
        "size": str(len(uuids) + 10),
    }
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    kw = {"timeout": timeout}
    if ctx is not None:
        kw["context"] = ctx
    with urllib.request.urlopen(req, **kw) as r:
        return json.loads(r.read())["data"]["hits"]


def run(cfg: Config, batch: int = 200) -> dict:
    log = get_logger("tcgamap", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    report: dict = {"step": "tcgamap",
                    "generated": datetime.now().isoformat(timespec="seconds")}

    banner(log, "TCGAMAP — RESOLVE FILE UUIDs TO TCGA BARCODES")

    tdir = cfg.paths.tcga_dir
    folders = [p.name for p in tdir.rglob("*")
               if p.is_dir() and len(p.name) == 36 and p.name.count("-") == 4]
    if not folders:
        folders = sorted({p.parent.name for p in tdir.rglob("*star_gene_counts.tsv")})
    if not folders:
        raise FileNotFoundError(
            f"No GDC file folders under {tdir}. Expected UUID-named "
            "directories containing *star_gene_counts.tsv")
    log.info(f"   {len(folders)} file UUIDs found on disk")
    log.info(f"   querying {ENDPOINT} in batches of {batch} ...")
    ctx = _ssl_context(log)
    try:
        import requests  # noqa: F401

        use_requests = True
        log.info("   using requests (bundles its own CA set)")
    except ImportError:
        use_requests = False

    hits: list[dict] = []
    for i in range(0, len(folders), batch):
        chunk = folders[i:i + batch]
        try:
            hits.extend(_query_requests(chunk) if use_requests
                        else _query(chunk, ctx=ctx))
        except ssl.SSLError as e:
            log.info(f"   [X] TLS error: {e}")
            log.info("       This is an Anaconda/Windows certificate-store")
            log.info("       problem, not a GDC one. Try:")
            log.info("         pip install --upgrade certifi")
            log.info("       or use the requests library, which bundles its own")
            log.info("       CA set:")
            log.info("         pip install requests")
            raise
        except urllib.error.URLError as e:
            log.info(f"   [X] network error: {e}")
            log.info("       api.gdc.cancer.gov must be reachable. Behind a")
            log.info("       proxy, export the Sample Sheet from the GDC cart")
            log.info("       page instead.")
            raise
        log.info(f"      {min(i + batch, len(folders))}/{len(folders)}")

    rows = []
    for h in hits:
        cases = h.get("cases") or []
        if not cases:
            continue
        c = cases[0]
        samples = c.get("samples") or []
        s = samples[0] if samples else {}
        rows.append({
            "File ID": h.get("file_id"),
            "File Name": h.get("file_name"),
            "Project ID": (c.get("project") or {}).get("project_id"),
            "Case ID": c.get("submitter_id"),
            "Sample ID": s.get("submitter_id"),
            "Sample Type": s.get("sample_type"),
            "Tissue Type": s.get("tissue_type"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("GDC returned no case information for these UUIDs.")

    log.info("")
    log.info(f"   resolved {len(df)}/{len(folders)} files")
    if "Project ID" in df:
        for k, v in df["Project ID"].value_counts().items():
            log.info(f"      {k}: {v}")
        projects = df["Project ID"].dropna().unique()
        if len(projects) > 1:
            log.info("   [!] more than one project in this folder — check that")
            log.info("       the download is what you think it is")
    for col in ("Sample Type", "Tissue Type"):
        if col in df and df[col].notna().any():
            log.info(f"   {col}:")
            for k, v in df[col].value_counts().items():
                log.info(f"      {k}: {v}")

    out = tdir / "gdc_sample_sheet.tsv"
    df.to_csv(out, sep="\t", index=False)
    log.info("")
    log.info(f"Written: {out}")

    # also write the sidecar directly, so wp3 can type samples without
    # re-assembling if the matrix is already barcode-keyed
    typ = df.dropna(subset=["Sample ID"])[["Sample ID", "Sample Type"]]
    typ.columns = ["sample", "sample_type"]
    typ = typ.drop_duplicates("sample")
    typ.to_csv(tdir / "tcga_sample_types.csv", index=False)
    log.info(f"Written: {tdir / 'tcga_sample_types.csv'}")
    log.info("")
    log.info("NEXT: delete tcga_expression_matrix.parquet so the matrix is")
    log.info("rebuilt with barcodes, then re-run wp3 and wp5.")

    report.update({"n_files": len(folders), "n_resolved": int(len(df)),
                   "projects": df["Project ID"].value_counts().to_dict()
                   if "Project ID" in df else {}})
    report["verdict"] = "OK" if len(df) >= 0.9 * len(folders) else "PARTIAL"
    with open(cfg.paths.results / "tcgamap.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    return report
