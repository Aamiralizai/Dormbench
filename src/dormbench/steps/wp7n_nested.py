"""WP7n - fully nested compartment-neutral redesign.

Review points 25 and 26, which together are the weakest part of the manuscript.

WP7 computes gene compartment-specificity on the Wu atlas, picks the specificity
cap by looking at the resulting curve, and evaluates the AUROC on that same
atlas. Its own log already says so. Its donor cross-fitting block improves on
that but still fits the elastic net once on all TRADITIOM samples and fixes the
cap at 0.1 before the folds begin, so the choice of cap is not nested.

Separately, the trade-off table reports CV accuracy 1.000 at every threshold.
That is a ceiling on a near-trivial contrast, not evidence of no performance
cost, and the manuscript must not claim the latter.

This step removes both problems:

  Nesting.   For each held-out donor fold, gene specificity is computed on the
             TRAINING donors only, EVERY candidate cap is evaluated on training
             donors only, the best cap is selected there, and only then is the
             resulting signature scored on the held-out donor. The held-out
             donor influences nothing that produced the signature.

  Honest performance. In vitro accuracy is recomputed with balanced accuracy,
             ROC AUC and log loss instead of raw accuracy, over grouped folds,
             with feature selection repeated inside each fold. Log loss is not
             bounded above by a ceiling, so it registers a cost that accuracy
             cannot. A permuted-label run gives the floor those metrics take on
             when there is no signal at all, which is what makes 1.000 readable.

What this can and cannot establish. A nested estimate on held-out donors of the
SAME atlas measures generalisation across donors, not across datasets. It does
not license the word "validated"; only a second atlas does that (see wp3r).
It does establish that the invariance result is not an artefact of selecting the
operating point on the data it is scored on, which is the specific charge.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import Config
from ..genes import resolve_against
from ..io import find_atlas_h5ad
from ..logging_utils import banner, get_logger
from .wp7_invariant import _auroc, _label

CANCER = "Cancer Epithelial"
STROMAL = ["CAFs", "PVL", "Endothelial"]
CYCLING = ("Cancer Cycling", "Cycling T-cells", "Cycling")
CAPS = (np.inf, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05)
MIN_CELLS = 20


def _clf(seed):
    """An UNFITTED elastic-net logistic model, matching wp7's configuration.

    sklearn >=1.8 deprecates `penalty=`; l1_ratio alone selects the elastic-net
    mix and is forward-compatible. Older versions still require penalty=.
    """
    from sklearn.linear_model import LogisticRegression

    kw = dict(solver="saga", l1_ratio=0.5, C=0.5, max_iter=4000,
              random_state=seed)
    try:
        LogisticRegression(**kw)
        return LogisticRegression(**kw)
    except TypeError:
        return LogisticRegression(penalty="elasticnet", **kw)


def _fit(X, y, seed):
    return _clf(seed).fit(X, y)


def _specificity(adata, cells_mask, var_names) -> pd.Series:
    """max |compartment mean - cancer mean| over compartments, on a cell subset."""
    sub = adata[cells_mask]
    ct = sub.obs["celltype_major"].astype(str).values
    means = {}
    for comp in [CANCER] + STROMAL:
        m = ct == comp
        if m.sum() >= MIN_CELLS:
            means[comp] = np.asarray(sub.X[m].mean(axis=0)).ravel()
    if CANCER not in means or len(means) < 2:
        return None
    other = np.vstack([v for k, v in means.items() if k != CANCER])
    return pd.Series(np.abs(other - means[CANCER]).max(axis=0), index=var_names)


def run(cfg: Config, n_genes: int = 200, n_permute: int = 20) -> dict:
    log = get_logger("wp7n", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    report: dict = {"step": "wp7n",
                    "generated": datetime.now().isoformat(timespec="seconds"),
                    "params": {"n_genes": n_genes, "n_permute": n_permute,
                               "caps": [None if np.isinf(c) else c
                                        for c in CAPS]}}

    banner(log, "WP7n - FULLY NESTED COMPARTMENT-NEUTRAL REDESIGN")
    log.info("Threshold selection happens INSIDE each donor fold, so the")
    log.info("held-out donor influences neither gene specificity nor the cap.")
    log.info("")

    import anndata as ad
    import scanpy as sc
    from scipy import sparse
    from sklearn.metrics import (balanced_accuracy_score, log_loss,
                                 roc_auc_score)
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # ---- TRADITIOM training data ----------------------------------------
    expr = pd.read_csv(cfg.paths.results / "traditiom_expression.csv")
    expr = expr.set_index(expr.columns[0])
    expr = expr.drop(columns=[c for c in ("direction",) if c in expr.columns])
    labels = pd.Series({c: _label(c) for c in expr.columns}).dropna()
    X_all = expr[labels.index].apply(pd.to_numeric, errors="coerce").dropna()
    y = (labels == "dormant").astype(int).values
    groups = np.array([re.sub(r"\.\d+$", "", c) for c in labels.index])
    log.info(f"   TRADITIOM: {X_all.shape[0]} genes x {len(labels)} samples, "
             f"{len(set(groups))} biological groups")
    log.info("")

    # ---- atlas -----------------------------------------------------------
    log.info("Loading atlas ...")
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
    if "donor_id" not in adata.obs.columns:
        raise ValueError("atlas lacks donor_id; nested folds are impossible")
    ct = adata.obs["celltype_major"].astype(str).values
    donor = adata.obs["donor_id"].astype(str).values
    minor = (adata.obs["celltype_minor"].astype(str)
             if "celltype_minor" in adata.obs
             else pd.Series(["?"] * adata.n_obs, index=adata.obs_names))
    keep = ~minor.str.contains("|".join(CYCLING), case=False, na=False).values
    Xa = adata.X.tocsc() if sparse.issparse(adata.X) else sparse.csc_matrix(adata.X)
    gidx = {g: i for i, g in enumerate(adata.var_names)}
    present = set(adata.var_names)
    log.info(f"   {adata.n_obs:,} cells, {len(set(donor))} donors")
    log.info("")

    # ---- nested donor folds ---------------------------------------------
    banner(log, "1. NESTED CAP SELECTION, EVALUATED OUT OF SAMPLE")
    log.info(f"{'held out':14s} {'cap chosen':>10s} {'train |0.5-AU|':>15s} "
             f"{'TEST AUROC':>11s} {'genes':>6s}")
    fold_rows = []
    for held in pd.unique(donor):
        tr_mask = donor != held
        te_mask = donor == held
        spec_tr = _specificity(adata, tr_mask, adata.var_names)
        if spec_tr is None:
            continue

        best = None
        for cap in CAPS:
            allowed = [g for g in X_all.index
                       if g not in spec_tr.index or spec_tr.get(g, 0) <= cap]
            if len(allowed) < 30:
                continue
            Xt = X_all.loc[allowed].T.values
            clf = _fit(StandardScaler().fit_transform(Xt), y, cfg.seed)
            coef = pd.Series(clf.coef_.ravel(), index=allowed)
            sel = coef[coef != 0].abs().sort_values(ascending=False).head(n_genes)
            up = [g for g in sel.index if coef[g] > 0]
            g_at = resolve_against(up, present)
            idx = [gidx[g] for g in g_at]
            if len(idx) < 5:
                continue
            # cap is scored on TRAINING donors only
            v = np.asarray(Xa[:, idx].mean(axis=1)).ravel()
            pm = tr_mask & keep & (ct == CANCER)
            nm = tr_mask & keep & np.isin(ct, STROMAL)
            if pm.sum() < MIN_CELLS or nm.sum() < MIN_CELLS:
                continue
            au_tr = _auroc(v[pm], v[nm])
            score = abs(0.5 - au_tr)
            if best is None or score < best["train_invariance"]:
                best = {"cap": cap, "train_invariance": score,
                        "idx": idx, "n_up": len(idx)}
        if best is None:
            continue

        v = np.asarray(Xa[:, best["idx"]].mean(axis=1)).ravel()
        pm = te_mask & keep & (ct == CANCER)
        nm = te_mask & keep & np.isin(ct, STROMAL)
        if pm.sum() < 10 or nm.sum() < 10:
            continue
        au_te = _auroc(v[pm], v[nm])
        fold_rows.append({"held_out_donor": held,
                          "cap_selected": (None if np.isinf(best["cap"])
                                           else best["cap"]),
                          "train_invariance": best["train_invariance"],
                          "test_auroc": au_te,
                          "test_invariance": abs(0.5 - au_te),
                          "n_up_genes": best["n_up"],
                          "n_cancer_cells": int(pm.sum()),
                          "n_stromal_cells": int(nm.sum())})
        log.info(f"{str(held)[:14]:14s} "
                 f"{str(fold_rows[-1]['cap_selected']):>10s} "
                 f"{best['train_invariance']:15.3f} {au_te:11.3f} "
                 f"{best['n_up']:6d}")

    folds = pd.DataFrame(fold_rows)
    if len(folds) < 5:
        log.info("")
        log.info("[X] fewer than 5 evaluable folds; nested estimate not usable")
        report["verdict"] = "NOT_EVALUABLE"
        return report
    folds.to_csv(cfg.paths.results / "wp7n_nested_folds.csv", index=False)

    au = folds["test_auroc"].values
    log.info("")
    log.info(f"   {len(au)} folds. Held-out AUROC mean {au.mean():.3f} "
             f"+/- {au.std():.3f}, median {np.median(au):.3f}")
    log.info(f"   range [{au.min():.3f}, {au.max():.3f}]")
    log.info(f"   |0.5 - mean| = {abs(0.5 - au.mean()):.3f} (fully nested)")
    caps_used = folds["cap_selected"].value_counts(dropna=False).to_dict()
    log.info(f"   caps selected across folds: {caps_used}")
    log.info("")

    # Review point 27: a mean near 0.5 can hide wide donor heterogeneity.
    n_stroma_biased = int((au < 0.4).sum())
    n_cancer_biased = int((au > 0.6).sum())
    log.info("   DONOR HETEROGENEITY (review point 27)")
    log.info(f"      strongly stroma-biased donors (AUROC<0.4): "
             f"{n_stroma_biased}/{len(au)}")
    log.info(f"      strongly cancer-biased donors (AUROC>0.6): "
             f"{n_cancer_biased}/{len(au)}")
    log.info("      A mean near 0.5 with a wide spread is not an invariant")
    log.info("      score; it is a score that is biased in both directions in")
    log.info("      different patients. Report the spread, not just the mean.")
    log.info("")
    report["nested"] = {
        "n_folds": int(len(au)), "auroc_mean": float(au.mean()),
        "auroc_sd": float(au.std()), "auroc_min": float(au.min()),
        "auroc_max": float(au.max()),
        "invariance": float(abs(0.5 - au.mean())),
        "n_stroma_biased": n_stroma_biased,
        "n_cancer_biased": n_cancer_biased,
        "caps_selected": {str(k): int(v) for k, v in caps_used.items()}}

    # ---- honest in vitro performance ------------------------------------
    banner(log, "2. IN VITRO PERFORMANCE WITHOUT THE ACCURACY CEILING")
    log.info("Accuracy saturates at 1.000 on this contrast, so it cannot show")
    log.info("a cost. Balanced accuracy, ROC AUC and log loss are reported")
    log.info("instead, with selection repeated inside each grouped fold.")
    log.info("")
    logo = LeaveOneGroupOut()

    def _metrics(cap, yy):
        allowed = [g for g in X_all.index]
        if np.isfinite(cap):
            spec_full = _specificity(adata, np.ones(adata.n_obs, bool),
                                     adata.var_names)
            allowed = [g for g in allowed
                       if g not in spec_full.index or spec_full.get(g, 0) <= cap]
        if len(allowed) < 30:
            return None
        Xt = X_all.loc[allowed].T.values
        pr, tr = [], []
        for tr_i, te_i in logo.split(Xt, yy, groups):
            if len(set(yy[tr_i])) < 2:
                continue
            # Scaler AND classifier are fitted inside the fold, so the
            # held-out group influences neither the scaling nor the model.
            pipe = Pipeline([("s", StandardScaler()), ("c", _clf(cfg.seed))])
            try:
                pipe.fit(Xt[tr_i], yy[tr_i])
                pr.extend(pipe.predict_proba(Xt[te_i])[:, 1])
                tr.extend(yy[te_i])
            except Exception:  # noqa: BLE001
                continue
        if len(set(tr)) < 2:
            return None
        pr, tr = np.asarray(pr), np.asarray(tr)
        return {"n_genes": len(allowed),
                "balanced_accuracy": float(
                    balanced_accuracy_score(tr, (pr > 0.5).astype(int))),
                "roc_auc": float(roc_auc_score(tr, pr)),
                "log_loss": float(log_loss(tr, np.clip(pr, 1e-6, 1 - 1e-6)))}

    perf_rows = []
    for cap in CAPS:
        m = _metrics(cap, y)
        if m is None:
            continue
        m["cap"] = None if np.isinf(cap) else cap
        perf_rows.append(m)
    perf = pd.DataFrame(perf_rows)

    # Permuted-label floor: what these metrics read when there is no signal.
    rng = np.random.default_rng(cfg.seed)
    floor = []
    for _ in range(n_permute):
        m = _metrics(np.inf, rng.permutation(y))
        if m:
            floor.append(m["log_loss"])
    if perf.empty:
        log.info("   [!] no evaluable model; skipping performance table")
    else:
        log.info(f"{'cap':>8s} {'genes':>7s} {'bal.acc':>9s} {'ROC AUC':>9s} "
                 f"{'log loss':>10s}")
        for _, r in perf.iterrows():
            log.info(f"{str(r['cap']):>8s} {int(r['n_genes']):7d} "
                     f"{r['balanced_accuracy']:9.3f} {r['roc_auc']:9.3f} "
                     f"{r['log_loss']:10.4f}")
        perf.to_csv(cfg.paths.results / "wp7n_performance.csv", index=False)
        base = perf.iloc[0]
        tight = perf.iloc[-1]
        d_ll = float(tight["log_loss"] - base["log_loss"])
        log.info("")
        log.info(f"   log loss cost of the tightest cap: {d_ll:+.4f}")
        if floor:
            log.info(f"   permuted-label log loss (no signal): "
                     f"{np.mean(floor):.4f} +/- {np.std(floor):.4f}")
            log.info("   The gap between the fitted and permuted values is the")
            log.info("   real headroom this task has; read the cost against it.")
        report["performance"] = perf.to_dict("records")
        report["log_loss_cost"] = d_ll
        report["permuted_log_loss_mean"] = (float(np.mean(floor)) if floor
                                            else None)
    log.info("")

    # ---- verdict ---------------------------------------------------------
    banner(log, "VERDICT")
    inv = abs(0.5 - au.mean())
    heterogeneous = (n_stroma_biased + n_cancer_biased) > len(au) / 3
    if inv <= 0.05 and not heterogeneous:
        verdict = "NESTED_INVARIANCE_HOLDS"
        log.info("The compartment-neutral redesign remains neutral when the")
        log.info("specificity cap is selected entirely within training donors")
        log.info("and scored on held-out donors, without excessive per-donor")
        log.info("spread. The in-sample-selection objection is answered for")
        log.info("this atlas. Call it a proof of concept that generalises")
        log.info("across donors; 'validated' still requires a second dataset.")
    elif inv <= 0.05:
        verdict = "NESTED_INVARIANCE_MEAN_ONLY"
        log.info("[!] Mean held-out AUROC is near 0.5, but individual donors")
        log.info("    are strongly biased in both directions. 'Compartment-")
        log.info("    neutral' is true on average and false per patient. The")
        log.info("    manuscript must state this; a score that is endothelial-")
        log.info("    biased in some patients and cancer-biased in others is")
        log.info("    not usable as a patient-level readout.")
    else:
        verdict = "NESTED_INVARIANCE_FAILS"
        log.info("[!] Under fully nested selection the redesigned signature is")
        log.info("    NOT compartment-neutral out of sample. The WP7 operating")
        log.info("    point was selected in-sample and does not generalise.")
        log.info("    Report this honestly and withdraw the redesign claim, or")
        log.info("    present it only as a negative result about feature")
        log.info("    selection. Do not report the in-sample curve as the")
        log.info("    headline.")
    report["verdict"] = verdict
    log.info("")
    log.info(f"VERDICT: {verdict}")

    with open(cfg.paths.results / "wp7n_nested.json", "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {cfg.paths.results / 'wp7n_nested_folds.csv'}")
    return report
