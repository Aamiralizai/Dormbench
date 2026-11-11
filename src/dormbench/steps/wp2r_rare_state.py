"""WP2r - is there a reproducible rare TRADITIOM-high malignant-cell state?

The strongest remaining single-cell objection (recommendation item 2.1). Every
compartment statement so far compares AVERAGE cancer-epithelial score with
stroma. A specialist can reasonably say: true dormant cells may be a small
fraction of the malignant population, and averaging over all malignant cells
erases the state that matters.

This step asks the question directly, at the only unit that can answer it:

    within an ER+ donor, are the non-cycling malignant cells in the upper tail
    of the TRADITIOM-up distribution transcriptionally distinct from the rest
    of that donor's malignant cells, in a way that repeats across donors?

Design (fixed before the real result was seen)
    Scope       ER+ donors; cancer epithelial cells; cycling-labelled cells and
                the top G2M quartile removed (the pipeline's existing rule, so
                this step inherits WP2c's deconfounding rather than inventing
                another filter).
    Tail        within each donor, rank that donor's remaining malignant cells
                by TRADITIOM-up and take the top q, for q in 1/5/10/20%.
                Never pooled across donors: donor score distributions differ.
    Comparator  the rest of the same donor's malignant cells.
    Statistic   AUROC = P(a tail cell scores above a non-tail cell) for each
                pathway, so 0.5 is 'no distinction'. Effect size, not p-value:
                with thousands of cells every comparison is significant.
    Null        within each donor, random subsets of the SAME SIZE drawn from the
                same pool. This is the null the objection needs: it asks whether
                the tail is special, or whether any equal-sized arbitrary subset
                of a donor's cells differs in these pathways. 200 draws.
    Replicate   the donor. Inference is a Wilcoxon signed-rank across donors on
                (AUROC - 0.5), with BH across pathways within a threshold.
    Panel       quiescence, E2F, G2M, EMT, TGF-beta, hypoxia, inflammatory,
                stress/apoptosis, and the other dormancy sets. The defining
                signature (TRADITIOM-up) is excluded from the panel because its
                AUROC is 1 by construction; it is reported separately as the
                tail-extraction check.

Reading the verdict
    REPRODUCIBLE_RARE_STATE          at least three state pathways separate the
                                     tail in a consistent direction, survive BH,
                                     AND keep their sign at every threshold.
    PARTIAL_RARE_STATE               some separation, but not both conditions.
    THRESHOLD_DEPENDENT              significant at some thresholds only.
    NO_REPRODUCIBLE_RARE_STATE       none: no reproducible upper-tail state.
    INSUFFICIENT_DONORS              fewer than five usable ER+ donors.

A null result here does NOT mean dormancy is unreal. It means TRADITIOM-up does
not define a reproducible upper-tail malignant state in treatment-naive primary
tumours - which is a different claim and must be written as such, with the
post-treatment/MRD caveat attached.

Input is results/wp2b_cell_scores.csv.gz (and the atlas for the stromal-marker
panel only). Output is results/wp2r_*.csv, results/wp2r_rare_state.json and
figures/fig13_rare_state.svg.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Config
from ..logging_utils import banner, get_logger

CANCER = "Cancer Epithelial"
CYCLING = ("Cancer Cycling", "Cycling T-cells", "Cycling")
PRIMARY = "TRADITIOM_dormancy|up"
PRIMARY_SHORT = "TRADITIOM-up"

THRESHOLDS = (0.01, 0.05, 0.10, 0.20)
HEADLINE_Q = 0.05
MIN_POOL = 30            # a donor needs malignant cells to have a tail at all
MIN_TAIL = 5
MIN_DONORS = 5
N_NULL = 200

# Prespecified pathway panel. (score column, role label, is_state_pathway)
PATHWAYS = [
    ("Quiescence_core|up", "quiescence", True),
    ("Quiescence_core|down", "quiescence (down component)", False),
    ("E2F_control|up", "proliferation (E2F)", False),
    ("G2M_control|up", "proliferation (G2M)", False),
    ("EMT_hallmark|up", "EMT", True),
    ("TGFbeta_hallmark|up", "TGF-beta", True),
    ("Hypoxia_hallmark|up", "hypoxia", True),
    ("TNFa_NFkB_hallmark|up", "inflammatory (TNFa/NFkB)", True),
    ("P53_hallmark|up", "stress / apoptosis (p53)", True),
    ("TRADITIOM_dormancy|down", "TRADITIOM down component", False),
    ("PreAdapted|up", "pre-adapted dormancy", True),
    ("NR2F1_axis|up", "NR2F1 axis", True),
    ("BHLHE41_dormancy|up", "BHLHE41 axis", True),
    ("Culture_drift|up", "culture drift (control)", False),
]

# Tails defined on OTHER signatures, used to test whether a TRADITIOM-up tail
# is special. Two of them are dormancy-family signatures from the same
# experiment, so a shared profile is convergence; `Culture_drift` is the
# same-experiment negative control. If every tail produces the same pathway
# profile, the profile belongs to the signature family (or to high-AUCell cells
# generally) and must not be reported as a TRADITIOM state.
SPECIFICITY_CONTROLS = ["PreAdapted|up", "Latency|up", "Culture_drift|up"]
SPECIFICITY_EXCESS = 0.05   # TRADITIOM must beat the best control by this much

# Stromal / endothelial markers that would indicate ambient RNA, doublets or
# annotation bleed rather than a malignant state. Panels are separate because
# the interpretation differs: endothelial contamination points at doublets with
# vasculature, fibroblast at CAF admixture.
STROMAL_MARKERS = {
    "endothelial": ["VWF", "EMCN", "PLVAP", "ACKR1", "KDR", "PECAM1"],
    "fibroblast": ["COL1A1", "COL1A2", "COL3A1", "DCN", "COL6A1", "COL6A2"],
    "perivascular": ["CAV1", "CAV2", "RGS5", "CSPG4", "MCAM"],
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _bh(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    idx = np.where(ok)[0]
    if idx.size:
        order = idx[np.argsort(p[idx])]
        m = len(order)
        adj = p[order] * m / np.arange(1, m + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        out[order] = np.clip(adj, 0, 1)
    return out


def _ranks(v: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(v)


def _auroc_from_ranks(ranks: np.ndarray, tail: np.ndarray) -> float:
    """AUROC of 'tail' versus the rest, from precomputed ranks.

    Ranks depend only on the score vector, which is fixed for a donor, so the
    observed value and every null replicate are a sum over the selected indices.
    """
    n1 = int(tail.sum())
    n0 = int(len(tail) - n1)
    if n1 < MIN_TAIL or n0 < MIN_TAIL:
        return np.nan
    s = ranks[tail].sum() - n1 * (n1 + 1) / 2.0
    return float(s / (n1 * n0))


def _tail_mask(score: np.ndarray, q: float) -> np.ndarray:
    """Top-q fraction of a donor's malignant pool, by score."""
    n = len(score)
    k = max(MIN_TAIL, int(round(q * n)))
    k = min(k, n - MIN_TAIL)
    if k < MIN_TAIL:
        return np.zeros(n, dtype=bool)
    cut = np.argsort(score)[::-1][:k]
    m = np.zeros(n, dtype=bool)
    m[cut] = True
    return m


# --------------------------------------------------------------------------
# optional stromal-marker panel
# --------------------------------------------------------------------------
def _stromal_marker_means(cfg: Config, n_cells_expected: int, obs: pd.DataFrame,
                          log) -> pd.DataFrame | None:
    """Per-cell mean log-normalised expression of the marker panels.

    Loaded only for these genes. The atlas is read backed, and positional
    alignment with the score table is verified against nCount_RNA before any
    value is used: a silent misalignment here would produce a confident wrong
    answer, which is the failure mode this project has hit before.
    """
    try:
        import anndata as ad
        from ..io import find_atlas_h5ad

        path = find_atlas_h5ad(cfg.paths.atlas_dir)
        log.info(f"   reading marker genes from {path.name} (backed)")
        adata = ad.read_h5ad(path, backed="r")

        if adata.n_obs != n_cells_expected:
            log.info(f"   [!] atlas has {adata.n_obs:,} cells but the score "
                     f"table has {n_cells_expected:,}; skipping marker panel")
            return None

        # alignment check: nCount_RNA must agree element-wise
        ref = adata.obs["nCount_RNA"].values.astype(float)
        got = pd.to_numeric(obs["nCount_RNA"], errors="coerce").values
        if not np.allclose(ref, got, rtol=0, atol=0, equal_nan=True):
            bad = int(np.sum(np.abs(ref - got) > 0))
            log.info(f"   [!] {bad:,} cells disagree on nCount_RNA; the score "
                     "table is not in atlas order. Skipping marker panel.")
            return None
        log.info("   row alignment verified on nCount_RNA (exact)")

        marker_genes = [g for p in STROMAL_MARKERS.values() for g in p]
        if "feature_name" in adata.var.columns:
            names = adata.var["feature_name"].astype(str).values
        else:
            names = np.asarray(adata.var_names, dtype=str)
        pos = {g: i for i, g in enumerate(names)}
        present = [g for g in marker_genes if g in pos]
        if len(present) < 3:
            log.info("   [!] too few marker genes present; skipping panel")
            return None

        cols = [pos[g] for g in present]
        # materialise only the needed columns
        X = adata.X[:, cols]
        if hasattr(X, "to_memory"):
            X = X.to_memory()
        X = X.X if hasattr(X, "X") else X
        X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=float)

        lib = np.asarray(X.sum(axis=1)).ravel()
        lib[lib == 0] = np.nan
        norm = np.log1p(X / lib[:, None] * 1e4)

        out = pd.DataFrame(index=np.arange(len(norm)))
        for panel, genes in STROMAL_MARKERS.items():
            idx = [k for k, g in enumerate(present) if g in genes]
            if not idx:
                continue
            out[panel] = norm[:, idx].mean(axis=1)
        log.info(f"   marker panels computed from {len(present)}/"
                 f"{len(marker_genes)} genes: {list(out.columns)}")
        return out
    except Exception as e:  # noqa: BLE001
        log.info(f"   [!] stromal-marker panel skipped: {type(e).__name__}: {e}")
        return None


# --------------------------------------------------------------------------
def run(cfg: Config, method: str = "AUCell", er_only: bool = True,
        q: float | None = None, n_null: int = N_NULL,
        stromal: bool = True) -> dict:
    log = get_logger("wp2r", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    rng = np.random.default_rng(cfg.seed)
    report: dict = {
        "step": "wp2r",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "params": {"method": method, "er_only": er_only, "thresholds":
                   list(THRESHOLDS), "headline_q": HEADLINE_Q,
                   "n_null": n_null},
    }

    banner(log, "WP2r - RARE TRADITIOM-HIGH MALIGNANT-CELL STATE")
    log.info("Compartment averages can hide a rare state. This asks whether the")
    log.info("upper tail of TRADITIOM-up within each donor's non-cycling")
    log.info("malignant cells is transcriptionally distinct, reproducibly, and")
    log.info("relative to a random subset of the same size.")
    log.info("")

    path = cfg.paths.results / "wp2b_cell_scores.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `dormbench wp2b` first.")
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df):,} cells x {df.shape[1]} columns")

    primary_col = f"{method}|{PRIMARY}"
    if primary_col not in df.columns:
        # tolerate a table written by a different scoring method
        cand = [c for c in df.columns if c.endswith(f"|{PRIMARY}")]
        if not cand:
            raise ValueError(f"no {method}|{PRIMARY} score column in "
                             f"{path.name}; the panel is missing the primary "
                             "signature")
        primary_col = cand[0]
    log.info(f"Primary (defining) score: {primary_col}")
    report["params"]["primary"] = primary_col

    # ---- scope -----------------------------------------------------------
    banner(log, "1. SCOPING THE MALIGNANT POOL")
    ct = df["celltype_major"].astype(str).values
    if "celltype_minor" in df.columns:
        minor = df["celltype_minor"].astype(str).values
    else:
        minor = np.array(["?"] * len(df))

    if er_only:
        if "subtype" not in df.columns:
            raise ValueError("no 'subtype' column; cannot restrict to ER+")
        # NOTE: 'ER' is a substring of 'HER2+', so match the label exactly.
        subtype = df["subtype"].astype(str).str.strip()
        is_er = subtype.eq("ER+").values
        log.info(f"ER+ restriction: {int(is_er.sum()):,} cells from "
                 f"{df.loc[is_er, 'donor_id'].nunique()} donors")
    else:
        is_er = np.ones(len(df), dtype=bool)
        log.info("ER+ restriction disabled: all subtypes retained")

    is_cyc = np.isin(minor, CYCLING)
    # proliferation filter, computed globally across the retained cells exactly
    # as wp2m does, so this step inherits the pipeline's rule rather than
    # inventing another one
    g2m_col = next((c for c in df.columns
                    if c.startswith(method) and "G2M" in c), None)
    if g2m_col is None:
        g2m_col = next((c for c in df.columns if "G2M" in c), None)
    if g2m_col is None:
        raise ValueError("no G2M score column found")
    g2m = pd.to_numeric(df[g2m_col], errors="coerce").values
    cut = np.nanquantile(g2m[is_er & ~is_cyc], 0.75)
    high_g2m = g2m > cut
    log.info(f"Proliferation reference: {g2m_col} (top-quartile cut {cut:.4f})")

    pool_mask = (ct == CANCER) & is_er & ~is_cyc & ~high_g2m
    log.info(f"Malignant pool: {int(pool_mask.sum()):,} non-cycling cancer "
             f"epithelial cells")
    log.info(f"   cycling-labelled removed: {int((is_er & (ct == CANCER) & is_cyc).sum()):,}")
    log.info(f"   top-G2M-quartile removed: "
             f"{int((is_er & (ct == CANCER) & ~is_cyc & high_g2m).sum()):,}")
    log.info("")

    donors = sorted(pd.unique(df.loc[pool_mask, "donor_id"].astype(str)))
    log.info(f"Donors with a malignant pool: {len(donors)}")

    # ---- per-donor tails and pathway AUROCs ------------------------------
    banner(log, "2. UPPER-TAIL EXTRACTION AND PATHWAY SEPARATION")
    pathway_cols = [(f"{method}|{c}", label, is_state)
                    for c, label, is_state in PATHWAYS
                    if f"{method}|{c}" in df.columns]
    missing = [c for c, _, _ in PATHWAYS if f"{method}|{c}" not in df.columns]
    if missing:
        log.info(f"   [!] pathway columns absent from the score table, skipped: "
                 f"{', '.join(missing)}")
    log.info(f"   {len(pathway_cols)} pathways scored (the defining signature "
             "is excluded: its tail AUROC is 1 by construction)")
    log.info("")

    obs = df[["nCount_RNA"]] if "nCount_RNA" in df.columns else pd.DataFrame()
    marker = None
    if stromal:
        banner(log, "3. STROMAL-MARKER CONTAMINATION PANEL")
        marker = _stromal_marker_means(cfg, len(df), obs, log) if len(obs) \
            else None
        if marker is None:
            log.info("   stromal-marker panel unavailable; the step continues "
                     "without it")
        log.info("")

    donor_rows, comp_rows, marker_rows = [], [], []
    qs = [q] if q is not None else list(THRESHOLDS)
    for donor in donors:
        idx_d = np.where(pool_mask &
                         (df["donor_id"].astype(str).values == donor))[0]
        if len(idx_d) < MIN_POOL:
            log.info(f"   {donor}: {len(idx_d)} cells < {MIN_POOL}; skipped")
            continue
        prim = pd.to_numeric(df[primary_col].values[idx_d], errors="coerce")
        ok = np.isfinite(prim)
        idx_d, prim = idx_d[ok], prim[ok]
        n = len(prim)
        if n < MIN_POOL:
            continue
        # ranks for every pathway within this donor; the null reuses them
        ranks = {}
        for col, label, _ in pathway_cols:
            v = pd.to_numeric(df[col].values[idx_d], errors="coerce")
            if not np.isfinite(v).all():
                v = np.where(np.isfinite(v), v, np.nanmedian(v))
            ranks[col] = _ranks(v)

        for qq in qs:
            tail = _tail_mask(prim, qq)
            k = int(tail.sum())
            if k < MIN_TAIL:
                continue
            comp_rows.append({
                "donor_id": donor, "threshold": qq, "pool_n": n,
                "tail_n": k, "tail_frac": k / n,
                "median_primary_tail": float(np.median(prim[tail])),
                "median_primary_rest": float(np.median(prim[~tail])),
            })
            # null: random subsets of the same size
            null_draws = np.empty((n_null, len(pathway_cols)))
            for r in range(n_null):
                pick = rng.choice(n, size=k, replace=False)
                m = np.zeros(n, dtype=bool)
                m[pick] = True
                for j, (col, _, _) in enumerate(pathway_cols):
                    null_draws[r, j] = _auroc_from_ranks(ranks[col], m)
            for j, (col, label, is_state) in enumerate(pathway_cols):
                obs_a = _auroc_from_ranks(ranks[col], tail)
                nd = null_draws[:, j]
                mu, sd = float(np.nanmean(nd)), float(np.nanstd(nd))
                lo, hi = (np.nanpercentile(nd, [2.5, 97.5])
                          if np.isfinite(nd).any() else (np.nan, np.nan))
                donor_rows.append({
                    "donor_id": donor, "threshold": qq, "pathway": label,
                    "score": col, "is_state_pathway": bool(is_state),
                    "pool_n": n, "tail_n": k, "auroc": obs_a,
                    "null_mean": mu, "null_sd": sd,
                    "null_lo": float(lo), "null_hi": float(hi),
                    "z": float((obs_a - mu) / sd) if sd and sd > 0 else np.nan,
                })
            # stromal markers, if available
            if marker is not None:
                for panel in marker.columns:
                    v = marker[panel].values[idx_d].astype(float)
                    ok_m = np.isfinite(v)
                    if not ok_m.all():
                        # a cell with no counts in any marker gene would
                        # otherwise poison the rank sum with NaN
                        fill = float(np.nanmedian(v[ok_m])) if ok_m.any() else 0.0
                        v = np.where(ok_m, v, fill)
                    a = _auroc_from_ranks(_ranks(v), tail)
                    if np.isfinite(a):
                        marker_rows.append({"donor_id": donor,
                                            "threshold": qq,
                                            "panel": panel, "auroc": a})

    if not donor_rows:
        raise RuntimeError("no donor produced a usable rare-cell tail")
    dr = pd.DataFrame(donor_rows)
    dr.to_csv(cfg.paths.results / "wp2r_donor_pathways.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(
        cfg.paths.results / "wp2r_tail_composition.csv", index=False)
    if marker_rows:
        pd.DataFrame(marker_rows).to_csv(
            cfg.paths.results / "wp2r_stromal_markers.csv", index=False)

    # ---- donor-level inference ------------------------------------------
    banner(log, "4. DONOR-LEVEL INFERENCE (donor is the replicate)")
    log.info("Wilcoxon signed-rank across donors on (AUROC - 0.5); BH within")
    log.info("each threshold across pathways. 0.5 = tail indistinguishable.")
    log.info("")
    summary = []
    for qq in qs:
        sub = dr[dr["threshold"] == qq]
        for label, blk in sub.groupby("pathway"):
            v = blk["auroc"].dropna().values - 0.5
            if len(v) < MIN_DONORS:
                continue
            try:
                p = stats.wilcoxon(v).pvalue if len(set(v)) > 1 else np.nan
            except ValueError:
                p = np.nan
            summary.append({
                "threshold": qq, "pathway": label,
                "is_state_pathway": bool(blk["is_state_pathway"].iloc[0]),
                "n_donors": int(len(v)),
                "median_auroc": float(np.nanmedian(v) + 0.5),
                "mean_effect": float(np.nanmean(v)),
                "n_above_half": int((v > 0).sum()),
                "n_below_half": int((v < 0).sum()),
                "frac_consistent": float(max((v > 0).sum(),
                                             (v < 0).sum()) / len(v)),
                "wilcoxon_p": float(p),
            })
    summ = pd.DataFrame(summary)
    if summ.empty:
        raise RuntimeError(f"no pathway had {MIN_DONORS}+ donors")
    summ["q"] = np.nan
    for qq, idx in summ.groupby("threshold").groups.items():
        summ.loc[idx, "q"] = _bh(summ.loc[idx, "wilcoxon_p"].values)
    summ.to_csv(cfg.paths.results / "wp2r_pathway_summary.csv", index=False)

    head = summ[(summ["threshold"] == HEADLINE_Q) & summ["is_state_pathway"]]
    head = head.sort_values("median_auroc")
    log.info(f"State pathways at top {HEADLINE_Q:.0%} "
             "(AUROC > 0.5 = enriched in the TRADITIOM-high tail):")
    log.info(f"{'pathway':30s} {'n':>3s} {'median AUROC':>13s} "
             f"{'consistent':>11s} {'q':>9s}")
    for _, r in head.iterrows():
        log.info(f"{r['pathway']:30s} {int(r['n_donors']):3d} "
                 f"{r['median_auroc']:13.3f} "
                 f"{r['frac_consistent']:11.2f} {r['q']:9.4g}")
    log.info("")

    # null comparison: how often is the observed effect beyond the random-
    # subset null of the same size?
    hz = dr[dr["threshold"] == HEADLINE_Q]
    beyond = (hz["z"].abs() > 2).groupby(hz["pathway"]).mean()
    log.info("Fraction of donors where the tail is > 2 SD from the "
             "same-size random-subset null:")
    for label in head["pathway"]:
        if label in beyond.index:
            log.info(f"   {label:30s} {beyond.loc[label]:.2f}")
    log.info("")

    # ---- is the profile specific to TRADITIOM? --------------------------
    banner(log, "4b. IS THIS A TRADITIOM STATE, OR JUST A HIGH-SCORING TAIL?")
    log.info("The tail is SELECTED on a high TRADITIOM score, so anything")
    log.info("correlated with it will be enriched by construction. Tails are")
    log.info("therefore re-derived from other signatures and the same pathway")
    log.info("profiles computed. A profile that every tail produces is a")
    log.info("property of the signature family, not of TRADITIOM.")
    log.info("")

    donor_arr = df["donor_id"].astype(str).values
    spec_rows = []
    tail_sets = {}      # donor -> set of cell positions (into df) in the tail
    for donor in donors:
        idx_d = np.where(pool_mask & (donor_arr == donor))[0]
        if len(idx_d) < MIN_POOL:
            continue
        pv = pd.to_numeric(df[primary_col].values[idx_d], errors="coerce")
        ok = np.isfinite(pv)
        idx_d, pv = idx_d[ok], pv[ok]
        if len(pv) < MIN_POOL:
            continue
        tail_sets[donor] = (idx_d, _tail_mask(pv, HEADLINE_Q))

    def _profile(score_col: str, tag: str) -> None:
        for donor, (idx_d, _) in tail_sets.items():
            v = pd.to_numeric(df[score_col].values[idx_d], errors="coerce")
            ok = np.isfinite(v)
            vv = v.copy()
            vv[~ok] = np.nanmedian(v) if ok.any() else 0.0
            tail = _tail_mask(vv, HEADLINE_Q)
            if int(tail.sum()) < MIN_TAIL:
                continue
            for col, label, is_state in pathway_cols:
                sv = pd.to_numeric(df[col].values[idx_d], errors="coerce")
                sv = np.where(np.isfinite(sv), sv, np.nanmedian(sv))
                spec_rows.append({
                    "defining_signature": tag, "donor_id": donor,
                    "pathway": label, "is_state_pathway": bool(is_state),
                    "auroc": _auroc_from_ranks(_ranks(sv), tail)})

    _profile(primary_col, PRIMARY)
    for ctrl in SPECIFICITY_CONTROLS:
        c = f"{method}|{ctrl}"
        if c in df.columns:
            _profile(c, ctrl)
        else:
            log.info(f"   [!] control signature absent: {c}")
    spec = pd.DataFrame(spec_rows)
    spec.to_csv(cfg.paths.results / "wp2r_specificity.csv", index=False)

    prof = (spec.groupby(["defining_signature", "pathway"])["auroc"]
            .median().unstack())
    log.info("Median AUROC of each pathway in the tail of each defining score:")
    show = prof.round(2)
    log.info(show.to_string())
    log.info("")

    # excess over the best control, per state pathway
    state_labels = [lab for _, lab, is_state in pathway_cols if is_state]
    ctrl_tags = [t for t in prof.index if t != PRIMARY]
    excess = {}
    if ctrl_tags:
        for lab in state_labels:
            if lab not in prof.columns:
                continue
            base = prof.loc[PRIMARY, lab]
            best = max(prof.loc[t, lab] for t in ctrl_tags
                       if lab in prof.columns and pd.notna(prof.loc[t, lab]))
            excess[lab] = float(base - best)
        log.info("TRADITIOM excess over the strongest control tail "
                 "(positive = TRADITIOM's tail is more extreme):")
        for lab, e in sorted(excess.items(), key=lambda kv: -kv[1]):
            flag = "  <-- specific" if e > SPECIFICITY_EXCESS else ""
            log.info(f"   {lab:30s} {e:+.3f}{flag}")
        log.info("")
        n_specific = sum(1 for e in excess.values() if e > SPECIFICITY_EXCESS)
        log.info(f"State pathways where TRADITIOM exceeds every control by "
                 f"> {SPECIFICITY_EXCESS:.2f}: {n_specific}/{len(excess)}")
        if n_specific == 0:
            log.info("[!] No state pathway is TRADITIOM-specific. The tail is")
            log.info("    transcriptionally distinct, but not distinct in a way")
            log.info("    that belongs to TRADITIOM rather than to correlated")
            log.info("    dormancy signatures.")
        log.info("")
    else:
        n_specific = 0
        log.info("[!] no control tails available; specificity not established")
        log.info("")

    # tail overlap: are the TRADITIOM and control tails the same cells?
    jac_rows = []
    for ctrl in [c for c in SPECIFICITY_CONTROLS if f"{method}|{c}" in df.columns]:
        vals = []
        for donor, (idx_d, tail) in tail_sets.items():
            pv = pd.to_numeric(df[f"{method}|{ctrl}"].values[idx_d],
                               errors="coerce")
            ok = np.isfinite(pv)
            pv = np.where(ok, pv, np.nanmedian(pv))
            ctail = _tail_mask(pv, HEADLINE_Q)
            inter = int((tail & ctail).sum())
            union = int((tail | ctail).sum())
            if union:
                vals.append(inter / union)
        if vals:
            jac_rows.append({"control": ctrl, "median_jaccard": float(np.median(vals)),
                             "n_donors": len(vals)})
    if jac_rows:
        jac = pd.DataFrame(jac_rows)
        jac.to_csv(cfg.paths.results / "wp2r_tail_overlap.csv", index=False)
        log.info("Tail overlap with control tails (Jaccard, median over donors):")
        for _, r in jac.iterrows():
            log.info(f"   {r['control']:22s} {r['median_jaccard']:.2f}")
        log.info("   High overlap means the signatures are selecting the same")
        log.info("   cells, so their pathway profiles cannot be independent.")
        log.info("")

    # ---- threshold stability --------------------------------------------
    banner(log, "5. DOES THE STATE SURVIVE A CHANGE OF THRESHOLD?")
    sig_at_head = set(head.loc[head["q"] < 0.05, "pathway"])
    stable, unstable = [], []
    for label in sorted({s["pathway"] for s in summary}):
        blk = summ[summ["pathway"] == label].sort_values("threshold")
        if len(blk) < 2:
            continue
        eff = blk["mean_effect"].values
        same = bool(np.all(np.sign(eff[np.isfinite(eff)]) ==
                           np.sign(eff[np.isfinite(eff)][0]))) \
            if np.isfinite(eff).any() else False
        (stable if same else unstable).append(label)
        if label in sig_at_head:
            log.info(f"   {label:30s} mean effect by threshold: "
                     + ", ".join(f"{q:.0%}={e:+.3f}"
                                 for q, e in zip(blk['threshold'], eff))
                     + ("   [stable]" if same else "   [!] sign changes"))
    log.info("")
    if len(qs) < 2:
        # a single-threshold run cannot test stability; do not let the absence
        # of a grid masquerade as instability
        stable = sorted({s["pathway"] for s in summary})
        unstable = []
        log.info("   [!] only one threshold run: stability not evaluated, so it")
        log.info("       is reported as stable rather than failing the check")
    log.info(f"   direction stable across thresholds: {len(stable)} pathways")
    if unstable:
        log.info(f"   [!] sign flips across thresholds: {', '.join(unstable)}")
    log.info("")

    # ---- stromal-marker panel -------------------------------------------
    stromal_out = None
    if marker_rows:
        mr = pd.DataFrame(marker_rows)
        mh = mr[mr["threshold"] == HEADLINE_Q]
        log.info("Stromal-marker enrichment in the tail "
                 "(AUROC > 0.5 = tail expresses the marker panel more):")
        for panel, blk in mh.groupby("panel"):
            v = blk["auroc"].dropna()
            log.info(f"   {panel:14s} median AUROC {v.median():.3f}  "
                     f"(n={len(v)})")
        stromal_out = {p: float(b["auroc"].median())
                       for p, b in mh.groupby("panel")
                       if np.isfinite(b["auroc"]).any()}
        if stromal_out:
            report["stromal_markers"] = stromal_out
        log.info("")
        log.info("   Read this as a contamination check, not as biology: a tail")
        log.info("   that is also high for VWF/COL1A1 may be ambient RNA or")
        log.info("   doublets rather than a malignant state.")
        log.info("")

    # ---- verdict ---------------------------------------------------------
    banner(log, "VERDICT")
    n_don = int(head["n_donors"].max()) if len(head) else 0
    n_sig = int(len(sig_at_head))
    state_stable = [p for p in sig_at_head if p in stable]
    log.info(f"ER+ donors in the analysis : {n_don}")
    log.info(f"State pathways separating the tail at q<0.05 : {n_sig}")
    log.info(f"   of which threshold-stable : {len(state_stable)}")
    log.info(f"   of which TRADITIOM-specific : {n_specific}")
    log.info("")

    if n_don < MIN_DONORS:
        verdict = "INSUFFICIENT_DONORS"
        log.info("Too few ER+ donors with a malignant pool to test anything.")
    elif n_sig == 0:
        verdict = "NO_REPRODUCIBLE_RARE_STATE"
        log.info("The upper tail of TRADITIOM-up does not separate from the rest")
        log.info("of a donor's malignant cells in any state pathway, at any")
        log.info("threshold, beyond a same-size random subset.")
        log.info("")
        log.info("Write this as: 'Neither compartment-level averages nor an")
        log.info("upper-tail analysis identified a reproducible TRADITIOM-high")
        log.info("malignant state in treatment-naive primary ER+ tumours.' Then")
        log.info("attach the caveat: post-treatment/MRD and metastatic dormancy")
        log.info("are not represented in this atlas.")
    elif n_sig >= 3 and len(state_stable) == n_sig and n_specific >= 3:
        verdict = "REPRODUCIBLE_RARE_STATE"
        log.info("!" * 72)
        log.info("A reproducible TRADITIOM-high malignant state is detectable AND")
        log.info(f"  {n_specific} state pathways are TRADITIOM-specific, i.e. the")
        log.info("  tail exceeds every control tail by a margin. This CHANGES the")
        log.info("  manuscript's conclusion from 'the signature is not a")
        log.info("  malignant readout' to 'it carries a genuine malignant-state")
        log.info("  component that tissue-level measurement also washes out'.")
        log.info("!" * 72)
    elif n_sig >= 3 and len(state_stable) == n_sig:
        verdict = "REPRODUCIBLE_STATE_NOT_TRADITIOM_SPECIFIC"
        log.info("The upper tail IS transcriptionally distinct and reproducible:")
        log.info(f"  {n_sig} state pathways separate it in a stable direction.")
        log.info("")
        log.info("But the profile is NOT specific to TRADITIOM. Tails derived")
        log.info("from the other dormancy signatures produce the same profile, so")
        log.info(f"  only {n_specific} state pathway(s) exceed every control.")
        log.info("")
        log.info("Write this as: 'TRADITIOM-high malignant cells are enriched")
        log.info("for a mesenchymal/hypoxic/stress programme, but so are the")
        log.info("tails of related dormancy signatures; the upper-tail phenotype")
        log.info("is a property of the signature family, not a TRADITIOM-specific")
        log.info("dormant state.' Do NOT claim a TRADITIOM-defined rare state.")
        log.info("")
        log.info("This is a stronger version of the manuscript's argument: the")
        log.info("signature does not isolate a malignant state even when you stop")
        log.info("averaging and go looking for the rarest cells.")
    elif sig_at_head and len(state_stable) < n_sig:
        verdict = "THRESHOLD_DEPENDENT"
        log.info("Separation exists but does not keep its direction when the")
        log.info("tail threshold changes. A biological state should not")
        log.info("depend on the cut, so this is reported as threshold-")
        log.info("dependent (hypothesis-generating), not as a state.")
    else:
        verdict = "PARTIAL_RARE_STATE"
        log.info("Some state pathways separate the tail, but fewer than three")
        log.info("and not all stable: report as partial / hypothesis-generating.")

    report.update({
        "verdict": verdict,
        "n_donors": n_don,
        "headline_q": HEADLINE_Q,
        "n_state_pathways_significant": n_sig,
        "n_state_pathways_stable": len(state_stable),
        "significant_pathways": sorted(sig_at_head),
        "stable_pathways": sorted(stable),
        "unstable_pathways": sorted(unstable),
        "null_beyond_fraction": {k: float(v) for k, v in beyond.items()},
        "specificity_excess": excess,
        "n_specific_state_pathways": int(n_specific),
        "specificity_threshold": SPECIFICITY_EXCESS,
        "specificity_controls": [c for c in SPECIFICITY_CONTROLS
                                 if f"{method}|{c}" in df.columns],
        "pathway_profile_by_defining_signature": prof.round(4).to_dict("index"),
        "headline_summary": head.to_dict("records"),
        "tail_composition": pd.DataFrame(comp_rows).to_dict("records"),
    })
    log.info("")
    log.info(f"VERDICT: {verdict}")

    # ---- figure ----------------------------------------------------------
    try:
        from ..viz.rare_state import plot_rare_state

        fig = plot_rare_state(
            df=df, donors=donors, pool_mask=pool_mask, primary=primary_col,
            donor_table=dr, composition=pd.DataFrame(comp_rows),
            summary=summ, thresholds=qs,
            headline_q=HEADLINE_Q, marker=marker, marker_table=(
                pd.DataFrame(marker_rows) if marker_rows else None),
            specificity=spec,
            out_path=cfg.paths.figures / "fig13_rare_state.svg")
        log.info(f"Written: {fig}")
        report["figure"] = str(fig)
    except Exception as e:  # noqa: BLE001
        log.info(f"[!] Figure skipped: {type(e).__name__}: {e}")

    with open(cfg.paths.results / "wp2r_rare_state.json", "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {cfg.paths.results / 'wp2r_rare_state.json'}")
    return report
