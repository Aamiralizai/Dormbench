"""WP1 — inspect the single-cell reference.

Answers one question: is the downloaded atlas usable as a deconvolution
reference, or do we need the GEO tarball?

Read-only and assumption-free. It reports what is in the file rather than
assuming a schema, because CELLxGENE standardisation silently renames and
drops fields.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import Config
from ..io import find_atlas_h5ad
from ..logging_utils import banner, get_logger

HINTS = {
    "celltype": ["celltype", "cell_type", "cluster", "annotation", "lineage"],
    "patient": ["patient", "donor", "sample", "subject", "case", "orig.ident"],
    "subtype": ["subtype", "receptor", "pam50", "er_status", "molecular"],
}


def _guess_columns(obs_cols: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for role, hints in HINTS.items():
        hits = []
        for col in obs_cols:
            low = col.lower()
            for rank, h in enumerate(hints):
                if h in low:
                    hits.append((rank, col))
                    break
        out[role] = [c for _, c in sorted(hits)]
    return out


def _describe_matrix(X) -> dict:
    """Classify a matrix as raw counts vs normalised, from a subsample."""
    n = min(2000, X.shape[0])
    sub = X[:n]
    if hasattr(sub, "toarray"):
        sub = sub.toarray()
    sub = np.asarray(sub, dtype=np.float64)
    finite = sub[np.isfinite(sub)]
    if finite.size == 0:
        return {"verdict": "empty", "max": None, "integer_like": None, "pct_zero": None}

    mx = float(finite.max())
    nz = finite[finite > 0]
    integer_like = bool(np.allclose(nz, np.round(nz), atol=1e-6)) if nz.size else False

    # Integer-valued expression is essentially always counts; normalised or
    # log-transformed values are continuous. Magnitude alone is unreliable.
    if integer_like:
        verdict = "raw counts"
    elif mx < 30:
        verdict = "log-normalised"
    else:
        verdict = "normalised (not logged)"

    return {
        "verdict": verdict,
        "max": round(mx, 3),
        "integer_like": integer_like,
        "pct_zero": round(float((sub == 0).mean()) * 100, 1),
    }


def run(cfg: Config, top: int = 25, h5ad: str | None = None) -> dict:
    log = get_logger("wp1", cfg.paths.logs)
    cfg.paths.ensure_outputs()

    import anndata as ad

    path = Path(h5ad) if h5ad else find_atlas_h5ad(cfg.paths.atlas_dir)
    report: dict = {
        "step": "wp1",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "file": str(path),
    }

    banner(log, "WP1 — REFERENCE INSPECTION")
    log.info(f"File : {path}")
    log.info(f"Size : {path.stat().st_size / 1e9:.2f} GB")
    log.info("")
    log.info("Opening in backed mode (matrix stays on disk)...")

    adata = ad.read_h5ad(path, backed="r")
    log.info(f"Shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    report["n_obs"], report["n_vars"] = int(adata.n_obs), int(adata.n_vars)
    log.info("")

    # 1. metadata inventory
    banner(log, "1. CELL METADATA (.obs)")
    obs_cols = list(adata.obs.columns)
    report["obs_columns"] = obs_cols
    log.info(f"{len(obs_cols)} columns:")
    for c in obs_cols:
        s = adata.obs[c]
        log.info(f"   {c:38s} {str(s.dtype):12s} {s.nunique(dropna=True):>6} unique")

    cands = _guess_columns(obs_cols)
    report["column_candidates"] = cands
    log.info("")
    log.info("Candidate columns by role:")
    for role, cols in cands.items():
        log.info(f"   {role:10s} -> {cols if cols else '*** NONE FOUND ***'}")
    log.info("")

    # 2. annotation content
    banner(log, "2. ANNOTATION CONTENT")
    report["categories"] = {}
    show = []
    for role in ("celltype", "subtype"):
        show.extend(cands[role][:3])
    for c in dict.fromkeys(show):
        vc = adata.obs[c].value_counts()
        report["categories"][c] = {str(k): int(v) for k, v in vc.head(60).items()}
        log.info(f"--- {c}  ({len(vc)} levels) ---")
        for k, v in vc.head(top).items():
            log.info(f"   {str(k)[:52]:54s} {v:>8,}")
        if len(vc) > top:
            log.info(f"   ... {len(vc) - top} more")
        log.info("")

    # 3. where are the counts
    banner(log, "3. WHERE ARE THE RAW COUNTS?")
    log.info("Deconvolution needs counts. CELLxGENE usually puts normalised")
    log.info("values in .X and keeps counts in .raw or a layer.")
    log.info("")

    mats = {"X": _describe_matrix(adata.X)}
    d = mats["X"]
    log.info(f"   {'.X':18s} {d['verdict']:24s} max={d['max']}  zeros={d['pct_zero']}%")

    layers = [k for k in adata.layers.keys() if isinstance(k, str)]
    for lay in layers:
        d = _describe_matrix(adata.layers[lay])
        mats[f"layers/{lay}"] = d
        log.info(f"   {f'.layers[{lay!r}]':18s} {d['verdict']:24s} "
                 f"max={d['max']}  zeros={d['pct_zero']}%")

    has_raw = adata.raw is not None
    if has_raw:
        d = _describe_matrix(adata.raw.X)
        mats["raw"] = d
        log.info(f"   {'.raw.X':18s} {d['verdict']:24s} max={d['max']}  "
                 f"zeros={d['pct_zero']}%  ({adata.raw.n_vars:,} genes)")
    else:
        log.info(f"   {'.raw':18s} ABSENT")

    report["matrices"], report["layers"], report["has_raw"] = mats, layers, bool(has_raw)
    log.info("")

    # 4. gene identifiers
    banner(log, "4. GENE IDENTIFIERS")
    var_cols = list(adata.var.columns)
    log.info(f".var columns: {var_cols}")
    log.info(f".var_names[:5]: {[str(x) for x in adata.var_names[:5]]}")
    ens = sum(str(x).startswith("ENSG") for x in adata.var_names[:500])
    id_type = "Ensembl" if ens > 250 else "symbols (or mixed)"
    log.info(f"Identifier type: {id_type}")
    sym_cols = [c for c in var_cols
                if any(h in c.lower() for h in ("symbol", "name", "hgnc"))]
    if ens > 250:
        log.info(f"Symbol column: {sym_cols if sym_cols else '*** NONE — map manually ***'}")
    report["var_columns"], report["gene_id_type"] = var_cols, id_type
    report["symbol_columns"] = sym_cols
    log.info("")

    # 5. verdict
    banner(log, "5. VERDICT")
    problems, warnings = [], []

    if not cands["celltype"]:
        problems.append("No cell-type column — unusable as a reference.")
    elif len(cands["celltype"]) < 2:
        warnings.append(
            "Only one cell-type column. Wu et al. has major/minor/subset levels; "
            "the CAF and PVL subdivisions matter for the stromal confound. "
            "If minor labels are missing, prefer the GEO tarball."
        )

    if not cands["subtype"]:
        warnings.append(
            "No clinical subtype column. You need receptor status (ER+), not "
            "histology (ductal/lobular). May need to join external metadata."
        )

    counts_in = [k for k, v in mats.items() if v["verdict"] == "raw counts"]
    if counts_in:
        log.info(f"Raw counts available in: {counts_in}")
    else:
        warnings.append(
            "No raw counts detected. Most deconvolution methods want counts; "
            "check whether your chosen method accepts normalised input."
        )
    report["counts_location"] = counts_in

    if id_type == "Ensembl":
        warnings.append(
            "Ensembl IDs — signature genes are symbols. Build the mapping once "
            "in WP2 and reuse it; do not map ad hoc per signature."
        )

    log.info("")
    if problems:
        log.info("BLOCKING:")
        for p in problems:
            log.info(f"   [X] {p}")
    if warnings:
        log.info("CHECK:")
        for w in warnings:
            log.info(f"   [!] {w}")
    if not problems and not warnings:
        log.info("   [OK] No issues. Proceed to WP2.")

    report["problems"], report["warnings"] = problems, warnings
    report["verdict"] = "BLOCKED" if problems else ("REVIEW" if warnings else "OK")
    log.info("")
    log.info(f"VERDICT: {report['verdict']}")

    out = cfg.paths.results / "wp1_reference_inventory.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"\nWritten: {out}")

    return report
