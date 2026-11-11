"""WP2f — why do these signatures score where they score?

WP2e established that the TRADITIOM up programme is depleted in cancer
epithelium relative to a matched null (z = -4.86) and scores highest in
endothelium. That is an observation without an explanation. A signature derived
entirely in MCF7 cells, where no stroma exists, should not preferentially mark
endothelial cells in primary tissue.

This step asks what is actually driving that, in three ways:

  1. Per-gene compartment specificity. For every gene in every signature,
     compute mean expression per compartment and the log ratio between the
     signature's top-scoring compartment and cancer epithelium. A score is a
     mean over genes, so its compartment preference must be traceable to a
     subset of members.

  2. Leading edge. Rank signature genes by that ratio and report the top
     contributors. If a handful of highly expressed stromal genes dominate,
     the signature is not measuring a distributed cancer programme -- it is
     being dragged by contamination or by genes with dual roles.

  3. Concentration. Report what fraction of the compartment difference is
     carried by the top 10 and top 50 genes. A programme distributed across
     hundreds of genes means something different from one carried by five.

Functional enrichment (Hallmark ORA) is attempted for the leading edge if
network access is available, but the compartment arithmetic above is the part
that does not depend on any external resource.

Caching: the per-gene x per-compartment mean expression table is written to
results/ on first run and reused, so this step reloads the atlas only once.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..io import find_atlas_h5ad
from ..logging_utils import banner, get_logger

CANCER = "Cancer Epithelial"
CYCLING = ("Cancer Cycling", "Cycling T-cells", "Cycling")


def _compartment_means(cfg: Config, log) -> pd.DataFrame:
    """Mean log-normalised expression per gene per compartment (cached)."""
    cache = cfg.paths.results / "wp2f_gene_compartment_means.csv.gz"
    if cache.exists():
        log.info(f"   using cached {cache.name}")
        return pd.read_csv(cache, index_col=0)

    import anndata as ad
    import scanpy as sc

    log.info("   loading atlas (first run only) ...")
    adata = ad.read_h5ad(find_atlas_h5ad(cfg.paths.atlas_dir))
    if adata.raw is not None:
        adata = ad.AnnData(X=adata.raw.X.copy(), obs=adata.obs.copy(),
                           var=adata.raw.var.copy()
                           if adata.raw.var is not None else adata.var.copy())
    if "feature_name" in adata.var.columns:
        adata.var_names = pd.Index(adata.var["feature_name"].astype(str))
        adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    det = np.asarray((adata.X > 0).mean(axis=0)).ravel()
    adata = adata[:, det > 0.01].copy()

    # non-cycling only, to match the WP2d/WP2e comparison
    minor = (adata.obs["celltype_minor"].astype(str)
             if "celltype_minor" in adata.obs
             else pd.Series(["?"] * adata.n_obs, index=adata.obs_names))
    keep = ~minor.str.contains("|".join(CYCLING), case=False, na=False).values
    adata = adata[keep].copy()
    log.info(f"   {adata.n_obs:,} non-cycling cells x {adata.n_vars:,} genes")

    ct = adata.obs["celltype_major"].astype(str).values
    out = {}
    for comp in np.unique(ct):
        sub = adata.X[ct == comp]
        out[comp] = np.asarray(sub.mean(axis=0)).ravel()
    df = pd.DataFrame(out, index=adata.var_names)
    df.to_csv(cache)
    log.info(f"   cached to {cache.name}")
    return df


def _load_hallmark(cfg, log, timeout_s: int = 60):
    """Return the MSigDB Hallmark table, preferring a local cache.

    ``decoupler.op.hallmark()`` fetches from OmniPath over the network with no
    timeout, so a slow or unreachable server hangs the whole pipeline
    indefinitely rather than failing. Three defences, in order:

      1. a local CSV cache, so this is a network call exactly once;
      2. an opt-out via DORMBENCH_NO_NETWORK=1 for offline runs;
      3. a hard timeout on a daemon thread, so a hang degrades to a skip.

    A daemon thread is used rather than signal.alarm because the latter is
    POSIX-only and this pipeline runs on Windows.
    """
    import os
    import threading

    cache = cfg.paths.processed / "hallmark_human.csv"
    if cache.exists():
        try:
            hm = pd.read_csv(cache)
            log.info(f"   Hallmark: loaded from cache ({cache.name}, "
                     f"{len(hm)} rows) — no network call")
            return hm
        except Exception as e:  # noqa: BLE001
            log.info(f"   [!] cache unreadable ({type(e).__name__}); refetching")

    if os.environ.get("DORMBENCH_NO_NETWORK", "").strip() not in ("", "0"):
        log.info("   DORMBENCH_NO_NETWORK set and no cache present — skipping")
        return None

    log.info(f"   fetching Hallmark from OmniPath (timeout {timeout_s}s) ...")
    box: dict = {}

    def _fetch():
        try:
            import decoupler as dc
            box["hm"] = dc.op.hallmark(organism="human")
        except Exception as e:  # noqa: BLE001
            box["err"] = e

    t = threading.Thread(target=_fetch, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        log.info(f"   [!] no response after {timeout_s}s — skipping enrichment.")
        log.info("   The concentration analysis above does not depend on it.")
        log.info("   Re-run `dormbench wp2f` later, or set")
        log.info("   DORMBENCH_NO_NETWORK=1 to skip this step permanently.")
        return None
    if "err" in box:
        raise box["err"]

    hm = box.get("hm")
    if hm is not None:
        try:
            cfg.paths.processed.mkdir(parents=True, exist_ok=True)
            hm.to_csv(cache, index=False)
            log.info(f"   cached to {cache} for future runs")
        except Exception as e:  # noqa: BLE001
            log.info(f"   [!] could not write cache: {type(e).__name__}")
    return hm


def run(cfg: Config, top_n: int = 30) -> dict:
    log = get_logger("wp2f", cfg.paths.logs)
    cfg.paths.ensure_outputs()

    report: dict = {"step": "wp2f",
                    "generated": datetime.now().isoformat(timespec="seconds")}

    banner(log, "WP2f — WHAT DRIVES THE COMPARTMENT PREFERENCE?")

    panel = pd.read_csv(cfg.paths.results / "signature_panel.csv")
    means = _compartment_means(cfg, log)
    log.info("")

    # which compartment each signature prefers, from wp2d if available
    prefs = {}
    d_path = cfg.paths.results / "wp2d_direction_attribution.csv"
    if d_path.exists():
        d = pd.read_csv(d_path)
        for (sig, direction), grp in d.groupby(["signature", "direction"]):
            prefs[(sig, direction)] = grp["top_noncycling"].mode().iat[0]

    all_rows = []
    lead_rows = []

    for (sig, direction), grp in panel.groupby(["signature", "direction"]):
        if "G2M" in sig or "E2F" in sig:
            continue
        genes = [g for g in grp["gene"].astype(str) if g in means.index]
        if len(genes) < 5:
            continue

        sub = means.loc[genes]
        target = prefs.get((sig, direction))
        # the reference cannot also be the target: for sets that prefer cancer
        # epithelium, compare against the highest-scoring other compartment
        if target is None or target not in sub.columns or target == CANCER:
            others = [c for c in sub.columns if c != CANCER]
            target = sub[others].mean(axis=0).idxmax()
            if prefs.get((sig, direction)) == CANCER:
                log.info(f"   note: this set prefers {CANCER}; comparing "
                         f"against {target} instead")

        # log ratio: preferred compartment vs cancer epithelium
        ratio = np.log2((sub[target] + 0.01) / (sub[CANCER] + 0.01))
        ranked = ratio.sort_values(ascending=False)

        # how concentrated is the difference?
        diff = (sub[target] - sub[CANCER]).sort_values(ascending=False)
        # denominator is the total POSITIVE difference, so the share cannot
        # exceed 100% when many genes run the other way
        pos_total = float(diff[diff > 0].sum())
        top10 = float(diff.head(10).clip(lower=0).sum() / pos_total) if pos_total else np.nan
        top50 = float(diff.head(50).clip(lower=0).sum() / pos_total) if pos_total else np.nan
        frac_pos = float((diff > 0).mean())
        # the statistic is undefined when the set is smaller than the window
        n_ok = len(genes) >= 100

        banner(log, f"{sig} | {direction}   ({len(genes)} genes)")
        log.info(f"   preferred compartment : {target}")
        log.info(f"   genes higher in {target} than in cancer: "
                 f"{frac_pos*100:.0f}%")
        if n_ok:
            log.info(f"   share of positive gap in top 10 genes : {top10*100:5.1f}%")
            log.info(f"   share in top 50 genes                 : {top50*100:5.1f}%")
        else:
            log.info(f"   concentration: undefined ({len(genes)} genes < 50; "
                     "every gene is in the 'top 50')")
        log.info("")
        log.info(f"   leading edge (highest {target} / cancer ratio):")
        for g in ranked.head(top_n).index:
            log.info(f"      {g:16s} log2 ratio {ranked[g]:+6.2f}   "
                     f"{target[:12]:12s} {sub.loc[g, target]:6.3f}   "
                     f"cancer {sub.loc[g, CANCER]:6.3f}")
            lead_rows.append({"signature": sig, "direction": direction,
                              "gene": g, "target": target,
                              "log2_ratio": float(ranked[g]),
                              "expr_target": float(sub.loc[g, target]),
                              "expr_cancer": float(sub.loc[g, CANCER])})
        log.info("")

        all_rows.append({
            "signature": sig, "direction": direction, "n_genes": len(genes),
            "preferred_compartment": target,
            "frac_genes_higher_in_target": frac_pos,
            "gap_share_top10": top10 if n_ok else np.nan,
            "gap_share_top50": top50 if n_ok else np.nan,
            "concentration_interpretable": bool(n_ok),
            "median_log2_ratio": float(ranked.median()),
        })

    summary = pd.DataFrame(all_rows)
    lead = pd.DataFrame(lead_rows)
    summary.to_csv(cfg.paths.results / "wp2f_concentration.csv", index=False)
    lead.to_csv(cfg.paths.results / "wp2f_leading_edge.csv", index=False)

    banner(log, "CONCENTRATION SUMMARY")
    log.info("If a small number of genes carries most of the compartment gap,")
    log.info("the signature is being dragged rather than reporting a")
    log.info("distributed programme.")
    log.info("")
    log.info(f"{'signature':22s} {'dir':5s} {'n':>5s} {'target':18s} "
             f"{'%>cancer':>9s} {'top10':>7s} {'top50':>7s}")
    for _, r in summary.sort_values("gap_share_top50").iterrows():
        log.info(f"{r['signature']:22s} {r['direction']:5s} "
                 f"{int(r['n_genes']):5d} {r['preferred_compartment']:18s} "
                 f"{r['frac_genes_higher_in_target']*100:8.0f}% "
                 f"{r['gap_share_top10']*100:6.1f}% {r['gap_share_top50']*100:6.1f}%")
    report["concentration"] = summary.to_dict("records")
    log.info("")

    # ---- functional enrichment of the leading edge ----------------------
    banner(log, "FUNCTIONAL ENRICHMENT (Hallmark ORA)")
    try:
        hm = _load_hallmark(cfg, log)
        if hm is None:
            raise RuntimeError("Hallmark table unavailable (cache miss + "
                               "network timeout)")
        cols = list(hm.columns)
        col_set = next((c for c in cols if c.lower() in
                        ("geneset", "source", "collection", "term")), cols[0])
        col_gene = next((c for c in cols if c.lower() in
                         ("genesymbol", "target", "gene", "symbol")), cols[1])
        log.info(f"   Hallmark: {hm[col_set].nunique()} sets, "
                 f"columns ({col_set}, {col_gene})")

        universe = set(means.index)
        enr_rows = []
        for (sig, direction), grp in lead.groupby(["signature", "direction"]):
            q = set(grp["gene"]) & universe
            if len(q) < 5:
                continue
            best = []
            for term, tg in hm.groupby(col_set):
                t = set(tg[col_gene].astype(str)) & universe
                if len(t) < 10:
                    continue
                k = len(q & t)
                if k < 2:
                    continue
                from scipy.stats import hypergeom
                p = float(hypergeom.sf(k - 1, len(universe), len(t), len(q)))
                best.append((p, term, k, len(t)))
            best.sort()
            log.info("")
            log.info(f"   {sig} | {direction}  (leading edge, {len(q)} genes)")
            for p, term, k, nt in best[:5]:
                log.info(f"      {term[:52]:54s} k={k:3d}/{nt:4d}  p={p:.2e}")
                enr_rows.append({"signature": sig, "direction": direction,
                                 "term": term, "overlap": k, "term_size": nt,
                                 "p": p})
        if enr_rows:
            pd.DataFrame(enr_rows).to_csv(
                cfg.paths.results / "wp2f_enrichment.csv", index=False)
            report["enrichment"] = "written"
        else:
            log.info("   no terms with sufficient overlap")
            report["enrichment"] = "empty"
    except Exception as e:  # noqa: BLE001
        log.info(f"   [!] unavailable: {type(e).__name__}: {e}")
        log.info("   Needs network access to OmniPath. The concentration")
        log.info("   analysis above does not depend on it.")
        report["enrichment"] = f"failed: {type(e).__name__}"
    log.info("")

    # ---- verdict ---------------------------------------------------------
    banner(log, "VERDICT")
    up = summary[summary["direction"] == "up"]
    testable = up[up["concentration_interpretable"]]
    small = up[~up["concentration_interpretable"]]
    dragged = testable[testable["gap_share_top50"] > 0.60]

    if len(small):
        log.info("Concentration undefined (fewer than 100 genes):")
        for _, r in small.iterrows():
            log.info(f"   {r['signature']:22s} {int(r['n_genes']):4d} genes "
                     "-- every member is in the 'top 50'")
        log.info("")

    log.info(f"Of signatures large enough to test, >60% of the positive gap in "
             f"the top 50: {len(dragged)}/{len(testable)}")
    for _, r in testable.sort_values("gap_share_top50", ascending=False).iterrows():
        tag = "  <== concentrated" if r["gap_share_top50"] > 0.60 else ""
        log.info(f"   {r['signature']:22s} top50 carries "
                 f"{r['gap_share_top50']*100:5.1f}%{tag}")
    log.info("")

    if len(testable) == 0:
        verdict = "UNTESTABLE"
        log.info("No signature has enough genes for the concentration statistic.")
        log.info("Report the leading-edge tables directly instead.")
    elif len(dragged) == len(testable):
        verdict = "CONCENTRATED"
        log.info("Every signature's compartment preference is carried by a small")
        log.info("minority of genes. These are not distributed programmes")
        log.info("mislocalising; they are scores dragged by a few members.")
        log.info("Report the leading-edge genes explicitly -- that list is the")
        log.info("actual finding.")
    elif len(dragged) == 0 and len(testable):
        verdict = "DISTRIBUTED"
        log.info("The compartment preference is spread across the whole gene")
        log.info("set. The signatures genuinely describe programmes that are")
        log.info("more active outside cancer epithelium, which is a stronger")
        log.info("and more interesting claim than contamination.")
    else:
        verdict = "MIXED"
        log.info("Some signatures are dragged by a few genes, others are")
        log.info("distributed. Treat them separately; they are not the same")
        log.info("kind of object.")

    report["verdict"] = verdict
    log.info("")
    log.info(f"VERDICT: {verdict}")

    out = cfg.paths.results / "wp2f_drivers.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {out}")
    log.info(f"Written: {cfg.paths.results / 'wp2f_leading_edge.csv'}")

    try:
        from ..viz.drivers import plot_drivers
        fig = plot_drivers(summary, lead, means,
                           cfg.paths.figures / "fig5_drivers.svg")
        log.info(f"Written: {fig}")
        report["figure"] = str(fig)
    except Exception as e:  # noqa: BLE001
        log.info(f"[!] Figure skipped: {type(e).__name__}: {e}")

    return report
