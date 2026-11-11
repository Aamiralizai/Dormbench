"""WP7 — can a dormancy signature be made compartment-invariant, and at what cost?

Everything upstream establishes that signatures derived from pure cell culture
carry compartment-associated expression into tissue. This step asks the
constructive question: if genes are selected with that confound penalised, does
the resulting signature stop scoring in stroma, and how much in vitro
discriminative power is lost?

The design is a trade-off curve rather than a single model, because the honest
answer is a cost, not a solution.

    unconstrained   elastic net on TRADITIOM dormancy vs proliferating,
                    with no penalty. Reproduces the standard approach.

    filtered        the same, after excluding genes whose expression is
                    compartment-specific in the reference atlas, at a series of
                    stringency thresholds.

Each resulting signature is evaluated twice: cross-validated accuracy in vitro,
which is what a conventional derivation optimises, and compartment attribution
(AUROC, cancer epithelium vs stroma) in the atlas, which is what determines
whether it can be interpreted in tissue.

Deliberate limitations, stated because they bound what this can show:

  * 51 bulk samples. This is a regularised linear problem; a deep model would
    fit the confound more efficiently, not less.
  * The expression matrix distributed with the source publication covers only
    the signature genes, so gene selection operates on a restricted space.
  * In vitro accuracy is not the endpoint of interest. A signature that
    separates dormant from proliferating flasks perfectly and reports stromal
    content in tissue is worthless, which is the point of plotting both.
  * Patient-level validation requires the dormant versus acquired-resistant
    labels from GSE111563, which are in the source publication rather than the
    GEO deposit. The hook is provided; the labels are not.
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

CANCER = "Cancer Epithelial"
STROMAL = ["CAFs", "PVL", "Endothelial"]
CYCLING = ("Cancer Cycling", "Cycling T-cells", "Cycling")


def _label(col: str) -> str | None:
    """Dormant vs proliferating, from TRADITIOM sample naming."""
    c = str(col).strip()
    if c.startswith("POT"):
        return "proliferating"
    if c.startswith("U") and c[1:2].isdigit():
        return "proliferating"          # untreated controls keep dividing
    if c.startswith(("T30", "T60", "AI30", "AI60", "AI90")):
        return "dormant"
    return None                          # latency, awakening, TEP excluded


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run(cfg: Config, thresholds=(np.inf, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05),
        n_genes: int = 200) -> dict:
    log = get_logger("wp7", cfg.paths.logs)
    cfg.paths.ensure_outputs()
    report: dict = {"step": "wp7",
                    "generated": datetime.now().isoformat(timespec="seconds")}

    banner(log, "WP7 — COMPARTMENT-INVARIANT SIGNATURE: THE TRADE-OFF")

    # ---- compartment specificity per gene, from the atlas ---------------
    means_path = cfg.paths.results / "wp2f_gene_compartment_means.csv.gz"
    if not means_path.exists():
        raise FileNotFoundError("Run `dormbench wp2f` first to build "
                                "wp2f_gene_compartment_means.csv.gz")
    means = pd.read_csv(means_path, index_col=0)
    # Specificity = the LARGEST absolute gap between cancer epithelium and any
    # other compartment. Taking (max stroma - cancer) and then abs() is not
    # equivalent: it misses genes expressed much LOWER in one compartment, and
    # it ignores immune and normal-epithelial populations entirely. A gene
    # neutral against endothelium can still be strongly CAF- or immune-specific.
    other_cols = [c for c in means.columns if c != CANCER]
    spec = means[other_cols].sub(means[CANCER], axis=0).abs().max(axis=1)
    per_comp = {c: means[c].sub(means[CANCER]).abs() for c in other_cols}
    log.info(f"   compartment specificity computed for {len(spec)} genes")
    log.info(f"   median max|compartment - cancer| = {spec.median():.3f}, "
             f"90th pct = {spec.quantile(0.9):.3f}")
    log.info("   per-compartment medians (each vs cancer epithelium):")
    for c, v in per_comp.items():
        log.info(f"      {c:22s} {v.median():.3f}")
    log.info("")

    # ---- TRADITIOM training data ----------------------------------------
    expr = pd.read_csv(cfg.paths.results / "traditiom_expression.csv")
    expr = expr.set_index(expr.columns[0])
    expr = expr.drop(columns=[c for c in ("direction",) if c in expr.columns])
    labels = pd.Series({c: _label(c) for c in expr.columns}).dropna()
    X_all = expr[labels.index].apply(pd.to_numeric, errors="coerce").dropna()
    y = (labels == "dormant").astype(int).values
    log.info(f"   TRADITIOM: {X_all.shape[0]} genes x {len(labels)} samples")
    log.info(f"   {int(y.sum())} dormant, {int((1 - y).sum())} proliferating")
    if len(labels) < 12 or y.sum() < 4:
        raise ValueError("too few labelled samples to train")
    log.info("")

    # ---- atlas, for compartment evaluation ------------------------------
    import anndata as ad
    import scanpy as sc
    from scipy import sparse

    log.info("Loading atlas for evaluation ...")
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
    ct = adata.obs["celltype_major"].astype(str).values
    minor = (adata.obs["celltype_minor"].astype(str)
             if "celltype_minor" in adata.obs
             else pd.Series(["?"] * adata.n_obs, index=adata.obs_names))
    keep = ~minor.str.contains("|".join(CYCLING), case=False, na=False).values
    best = max(STROMAL, key=lambda s: int((keep & (ct == s)).sum()))
    pos_m, neg_m = keep & (ct == CANCER), keep & (ct == best)
    present = set(adata.var_names)
    Xa = adata.X.tocsc() if sparse.issparse(adata.X) else sparse.csc_matrix(adata.X)
    gidx = {g: i for i, g in enumerate(adata.var_names)}
    log.info(f"   {CANCER} n={int(pos_m.sum()):,} vs {best} n={int(neg_m.sum()):,}")
    log.info("")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Replicates of one flask/timepoint (T30.1, T30.2) are not independent
    # observations. Leaving out a single sample while its own replicate stays
    # in the training set inflates accuracy. Group by condition-timepoint and
    # hold out whole groups.
    groups = np.array([re.sub(r"\.\d+$", "", c) for c in labels.index])
    log.info(f"   {len(set(groups))} biological groups for grouped CV: "
             f"{sorted(set(groups))}")
    log.info("")

    rows = []
    banner(log, "TRADE-OFF: IN VITRO ACCURACY vs COMPARTMENT INVARIANCE")
    log.info(f"{'max specificity':>16s} {'genes':>7s} {'CV acc':>8s} "
             f"{'n sig':>6s} {'AUROC':>8s} {'|0.5-AUROC|':>12s}")

    for thr in thresholds:
        allowed = [g for g in X_all.index
                   if g not in spec.index or spec.get(g, 0) <= thr]
        if len(allowed) < 30:
            log.info(f"{thr:16.2f} {len(allowed):7d}  too few genes, skipped")
            continue
        Xt = X_all.loc[allowed].T.values

        # sklearn >=1.8 deprecates `penalty=`; l1_ratio alone selects the
        # elastic-net mix and is forward-compatible to 1.10.
        try:
            clf = LogisticRegression(solver="saga", l1_ratio=0.5, C=0.5,
                                     max_iter=4000, random_state=cfg.seed)
        except TypeError:  # older sklearn still wants penalty=
            clf = LogisticRegression(penalty="elasticnet", solver="saga",
                                     l1_ratio=0.5, C=0.5, max_iter=4000,
                                     random_state=cfg.seed)
        # Scaling is fitted INSIDE each fold. Standardising the whole matrix
        # first lets held-out samples influence the feature scaling.
        pipe = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
        try:
            cv = float(np.mean(cross_val_score(
                pipe, Xt, y, cv=LeaveOneGroupOut(), groups=groups,
                scoring="accuracy")))
        except Exception:  # noqa: BLE001
            cv = np.nan
        clf.fit(StandardScaler().fit_transform(Xt), y)
        coef = pd.Series(clf.coef_.ravel(), index=allowed)
        sel = coef[coef != 0].abs().sort_values(ascending=False).head(n_genes)
        up = [g for g in sel.index if coef[g] > 0]

        # Score the learned up-signature in the atlas.
        # NOTE: this is an UNWEIGHTED mean over positive-coefficient genes,
        # whereas cv_accuracy above evaluates the full signed elastic-net
        # predictor. The two columns therefore describe related but distinct
        # objects, and the row is not a strict trade-off for one model. An
        # unweighted positive-gene score is used because the atlas is a
        # different platform from the culture matrix and the fitted
        # coefficients are not directly transportable across them. Applying
        # the full linear predictor requires rank-transformed or
        # cross-platform-standardised expression and is not done here.
        g_at = resolve_against(up, present)
        idx = [gidx[g] for g in g_at]
        if len(idx) >= 5:
            v = np.asarray(Xa[:, idx].mean(axis=1)).ravel()
            au = _auroc(v[pos_m], v[neg_m])
        else:
            au = np.nan

        rows.append({"max_specificity": None if np.isinf(thr) else thr,
                     "genes_available": len(allowed), "cv_accuracy": cv,
                     "n_selected": int(len(sel)), "n_up": len(up),
                     "n_up_in_atlas": len(idx), "auroc": au,
                     "invariance": abs(0.5 - au) if np.isfinite(au) else np.nan,
                     "signature": ";".join(up[:80])})
        lbl = "none" if np.isinf(thr) else f"{thr:.2f}"
        log.info(f"{lbl:>16s} {len(allowed):7d} {cv:8.3f} {len(up):6d} "
                 f"{au:8.3f} {abs(0.5 - au):12.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(cfg.paths.results / "wp7_tradeoff.csv", index=False)
    log.info("")

    # ---- donor cross-fitting --------------------------------------------
    # The trade-off above computes gene specificity, picks the threshold and
    # evaluates AUROC all in the same atlas, so the operating point is
    # optimistic. Here specificity is recomputed within TRAINING donors only
    # and the constrained signature's cancer-vs-stroma AUROC is measured in
    # HELD-OUT donors, over donor folds, for an out-of-sample estimate.
    if "donor_id" in adata.obs.columns and np.isfinite(res["auroc"]).any():
        banner(log, "DONOR CROSS-FITTING (out-of-sample AUROC)")
        donor = adata.obs["donor_id"].astype(str).values
        cap = 0.1
        allowed = [g for g in X_all.index
                   if g not in spec.index or spec.get(g, 0) <= cap]
        Xt = X_all.loc[allowed].T.values
        try:
            clf = LogisticRegression(solver="saga", l1_ratio=0.5, C=0.5,
                                     max_iter=4000, random_state=cfg.seed)
        except TypeError:
            clf = LogisticRegression(penalty="elasticnet", solver="saga",
                                     l1_ratio=0.5, C=0.5, max_iter=4000,
                                     random_state=cfg.seed)
        clf.fit(StandardScaler().fit_transform(Xt), y)
        coef = pd.Series(clf.coef_.ravel(), index=allowed)
        fold_au = []
        for held in pd.unique(donor):
            tr_mask = donor != held
            tr_cells = adata[tr_mask]
            ctr = tr_cells.obs["celltype_major"].astype(str).values
            cm = {}
            for compartment in [CANCER] + STROMAL:
                m = ctr == compartment
                if m.sum() >= 20:
                    cm[compartment] = np.asarray(
                        tr_cells.X[m].mean(axis=0)).ravel()
            if CANCER not in cm or len(cm) < 2:
                continue
            others = np.vstack([v for k, v in cm.items() if k != CANCER])
            spec_tr = pd.Series(np.abs(others - cm[CANCER]).max(axis=0),
                                index=adata.var_names)
            up = [g for g in coef.index if coef[g] > 0
                  and (g not in spec_tr.index or spec_tr.get(g, 0) <= cap)]
            g_at = resolve_against(up, present)
            idx = [gidx[g] for g in g_at]
            if len(idx) < 5:
                continue
            te_mask = donor == held
            te_ct = adata.obs["celltype_major"].astype(str).values[te_mask]
            te_keep = keep[te_mask]
            v = np.asarray(Xa[te_mask][:, idx].mean(axis=1)).ravel()
            pm = te_keep & (te_ct == CANCER)
            nm = te_keep & np.isin(te_ct, STROMAL)
            if pm.sum() >= 10 and nm.sum() >= 10:
                fold_au.append(_auroc(v[pm], v[nm]))
        if len(fold_au) >= 5:
            fa = np.array(fold_au)
            log.info(f"   {len(fa)} donor folds, held-out AUROC "
                     f"{fa.mean():.3f} \u00b1 {fa.std():.3f}")
            log.info(f"   median {np.median(fa):.3f}, "
                     f"range [{fa.min():.3f}, {fa.max():.3f}]")
            log.info(f"   |0.5 - mean AUROC| = {abs(0.5 - fa.mean()):.3f} "
                     "out of sample")
            report["cross_fit_auroc_mean"] = float(fa.mean())
            report["cross_fit_auroc_sd"] = float(fa.std())
            report["cross_fit_n_folds"] = int(len(fa))
            report["cross_fit_invariance"] = float(abs(0.5 - fa.mean()))
            pd.DataFrame({"held_out_auroc": fa}).to_csv(
                cfg.paths.results / "wp7_crossfit.csv", index=False)
        else:
            log.info("   too few evaluable donor folds for cross-fitting")
        log.info("")

    # ---- verdict ---------------------------------------------------------
    banner(log, "VERDICT")
    ok = res[res["auroc"].notna() & res["cv_accuracy"].notna()]
    if len(ok) < 2:
        verdict = "NOT_EVALUABLE"
        log.info("Too few evaluable models to describe a trade-off.")
    else:
        base = ok.iloc[0]
        best_inv = ok.loc[ok["invariance"].idxmin()]
        log.info(f"unconstrained : CV {base['cv_accuracy']:.3f}, "
                 f"AUROC {base['auroc']:.3f}")
        log.info(f"most invariant: CV {best_inv['cv_accuracy']:.3f}, "
                 f"AUROC {best_inv['auroc']:.3f} "
                 f"(specificity cap {best_inv['max_specificity']})")
        d_acc = base["cv_accuracy"] - best_inv["cv_accuracy"]
        d_inv = base["invariance"] - best_inv["invariance"]
        log.info("")
        log.info(f"cost: {d_acc:+.3f} in vitro accuracy for "
                 f"{d_inv:+.3f} in compartment invariance")
        log.info("")

        # Accuracy is saturated: dormant vs proliferating flasks in pure
        # culture is a near-trivial contrast, and every model scores 1.00
        # whether or not scaling is fitted in-fold and whether folds are
        # grouped. A ceiling cannot register a cost, so the absence of an
        # observed accuracy drop is NOT evidence that there is none.
        saturated = bool((ok["cv_accuracy"] >= 0.999).all())
        report["cv_saturated"] = saturated
        if saturated:
            log.info("[!] CV accuracy is 1.000 at every threshold. This is a")
            log.info("    ceiling, not a demonstration of low cost: the in")
            log.info("    vitro contrast is too easily separable for this")
            log.info("    metric to detect a performance penalty. Report the")
            log.info("    invariance gain; do not claim the cost is negligible.")
            log.info("")

        # The threshold is chosen on the same atlas used to define gene
        # specificity AND to compute the AUROC, so the selected operating
        # point is optimistic. Treat the curve as exploratory.
        report["threshold_selected_in_sample"] = True
        log.info("[!] Gene specificity, threshold choice and AUROC evaluation")
        log.info("    all use the Wu atlas. The best operating point is")
        log.info("    therefore selected in-sample and is likely optimistic.")
        log.info("    Cross-fitted or independent-atlas evaluation is required")
        log.info("    before this is presented as a validated signature.")
        log.info("")

        if d_inv > 0.1 and d_acc < 0.1:
            verdict = ("INVARIANCE_IMPROVED_COST_UNRESOLVED" if saturated
                       else "INVARIANCE_ACHIEVABLE")
            log.info("Compartment invariance improves substantially with no")
            log.info("observable loss of in vitro discrimination in this")
            log.info("dataset. Penalising compartment-specific genes during")
            log.info("derivation is worth testing further; the true")
            log.info("performance cost is not established here.")
        elif d_inv > 0.1:
            verdict = "INVARIANCE_COSTS_ACCURACY"
            log.info("Invariance is achievable but costs in vitro accuracy.")
            log.info("That trade-off is the finding: the genes that best")
            log.info("separate dormant from proliferating cells in culture are")
            log.info("substantially the same genes that carry compartment")
            log.info("information in tissue.")
        else:
            verdict = "INVARIANCE_NOT_ACHIEVABLE"
            log.info("Excluding compartment-specific genes does not make the")
            log.info("signature compartment-invariant. The confound is not")
            log.info("carried by an identifiable gene subset, consistent with")
            log.info("WP2y, and cannot be removed by feature selection alone.")

    report["verdict"] = verdict
    report["tradeoff"] = res.drop(columns=["signature"]).to_dict("records")
    log.info("")
    log.info(f"VERDICT: {verdict}")
    log.info("")
    log.info("NEXT: patient-level evaluation requires the dormant vs")
    log.info("acquired-resistant labels from Selli et al. 2019 supplementary")
    log.info("tables, as config/wp6_groups.csv (patient,group). Without them")
    log.info("this is an in vitro result with a tissue-invariance check, not a")
    log.info("clinically validated signature.")

    try:
        from ..viz.tradeoff import plot_tradeoff
        fig = plot_tradeoff(res, cfg.paths.figures / "fig11_invariance_tradeoff.svg")
        log.info(f"Written: {fig}")
        report["figure"] = str(fig)
    except Exception as e:  # noqa: BLE001
        log.info(f"[!] Figure skipped: {type(e).__name__}: {e}")

    with open(cfg.paths.results / "wp7_invariant.json", "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"Written: {cfg.paths.results / 'wp7_tradeoff.csv'}")
    return report
