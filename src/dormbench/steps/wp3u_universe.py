"""WP3u — build the gene universe shared by all bulk cohorts.

Null baselines in WP3 are not comparable between cohorts when the gene
universes differ. TCGA ships 59,427 genes including many barely-detected ones;
random sets drawn from those track sequencing depth, which itself tracks
stromal content, pushing its null to about +0.32. METABRIC's 20,385-gene
microarray gives a null near -0.30. Restricting every cohort to the genes they
share, and that are actually expressed, removes most of that difference and
makes z comparable across platforms.

Writes results/wp3_common_genes.txt. Then run:
    dormbench wp3 --common-genes
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..io import (DataNotFound, load_metabric, load_scanb, load_tcga,
                  tidy_metabric_expression)
from ..logging_utils import banner, get_logger


def run(cfg: Config, min_expr_quantile: float = 0.25) -> dict:
    log = get_logger("wp3u", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    report = {"step": "wp3u",
              "generated": datetime.now().isoformat(timespec="seconds")}

    banner(log, "WP3u — COMMON GENE UNIVERSE ACROSS BULK COHORTS")

    sets = {}
    try:
        mb = load_metabric(cfg.paths.metabric_dir)
        e = tidy_metabric_expression(mb["expr"])
        sets["METABRIC"] = _expressed(e, min_expr_quantile)
        log.info(f"   METABRIC : {e.shape[0]} genes -> "
                 f"{len(sets['METABRIC'])} expressed")
        del e, mb
    except (DataNotFound, FileNotFoundError, KeyError) as ex:
        log.info(f"   [!] METABRIC: {ex}")

    try:
        sb = load_scanb(cfg.paths.scanb_dir)
        e = pd.read_csv(sb["expr_path"], index_col=0, low_memory=False)
        e = e.apply(pd.to_numeric, errors="coerce", downcast="float")
        sets["SCAN-B"] = _expressed(e, min_expr_quantile)
        log.info(f"   SCAN-B   : {e.shape[0]} genes -> "
                 f"{len(sets['SCAN-B'])} expressed")
        del e, sb
    except (DataNotFound, FileNotFoundError, KeyError, ValueError) as ex:
        log.info(f"   [!] SCAN-B: {ex}")

    try:
        tc = load_tcga(cfg.paths.tcga_dir, log=log)
        e = tc["expr"]
        sets["TCGA-BRCA"] = _expressed(e, min_expr_quantile)
        log.info(f"   TCGA     : {e.shape[0]} genes -> "
                 f"{len(sets['TCGA-BRCA'])} expressed")
        del e, tc
    except (DataNotFound, FileNotFoundError, KeyError, ValueError) as ex:
        log.info(f"   [!] TCGA: {ex}")

    if len(sets) < 2:
        raise FileNotFoundError("Need at least two cohorts to intersect.")

    common = set.intersection(*sets.values())
    log.info("")
    log.info(f"Shared and expressed in all {len(sets)} cohorts: "
             f"{len(common)} genes")
    out = cfg.paths.results / "wp3_common_genes.txt"
    out.write_text("\n".join(sorted(common)), encoding="utf-8")
    log.info(f"Written: {out}")
    log.info("")
    log.info("Now run:  dormbench wp3 --common-genes")

    report["n_common"] = len(common)
    report["per_cohort"] = {k: len(v) for k, v in sets.items()}
    report["verdict"] = "OK"
    with open(cfg.paths.results / "wp3u_universe.json", "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    return report


def _expressed(e: pd.DataFrame, q: float) -> set:
    """Genes above the q-th quantile of mean expression in this cohort."""
    m = e.mean(axis=1)
    thr = float(np.nanquantile(m, q))
    return set(m.index[m > thr].astype(str))
