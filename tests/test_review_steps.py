"""Tests for the v0.2.0 review-response steps.

These run the new steps end to end on small synthetic fixtures with a KNOWN
answer, so a logic error is caught here rather than after a multi-hour run on
the real atlas. They do not touch real atlas data.

The fixtures are built so the correct result is not in doubt:
  - a signature planted in endothelial cells MUST come out stroma-dominant;
  - a signature with no compartment structure MUST come out near 0.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("anndata")
pytest.importorskip("scanpy")

from dormbench.steps.wp2m_multiverse import _g2m_matched
from dormbench.steps.wp2m_multiverse import _auroc as auroc_m
from dormbench.steps.wp3r_replication import _canonical
from dormbench.steps.wp8p_pseudobulk import _rank_score


# --------------------------------------------------------------- wp8p ------

def test_rank_score_is_invariant_to_library_size():
    """The pseudobulk statistic must not track sequencing depth.

    This is the whole reason a rank statistic is used: donor x compartment
    libraries differ enormously in total counts, and a depth-sensitive score
    would reproduce that rather than the biology.
    """
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(200)]
    base = pd.DataFrame(rng.lognormal(size=(200, 4)), index=genes,
                        columns=list("abcd"))
    scaled = base.copy()
    scaled["b"] = scaled["b"] * 1000        # a library 1000x deeper

    sig = genes[:20]
    s1 = _rank_score(base, sig)
    s2 = _rank_score(scaled, sig)
    assert np.allclose(s1.values, s2.values), (
        "rank score changed when a library was rescaled; it is not "
        "depth-invariant")


def test_rank_score_detects_planted_signal():
    genes = [f"G{i}" for i in range(100)]
    mat = pd.DataFrame(1.0, index=genes, columns=["lo", "hi"])
    sig = genes[:10]
    mat.loc[sig, "hi"] = 50.0               # signature high in one library
    s = _rank_score(mat, sig)
    assert s["hi"] > s["lo"]


def test_rank_score_needs_enough_genes():
    genes = [f"G{i}" for i in range(50)]
    mat = pd.DataFrame(1.0, index=genes, columns=["a", "b"])
    s = _rank_score(mat, genes[:3])         # below the 5-gene minimum
    assert s.isna().all()


# --------------------------------------------------------------- wp2m ------

def test_g2m_matching_equalises_the_proliferation_profile():
    """After matching, the two compartments must have the same G2M distribution.

    That is the point of the matched scheme: it removes the confound without
    excluding cells, unlike the quantile filter.
    """
    rng = np.random.default_rng(1)
    n = 4000
    # cancer cells are systematically higher-G2M, as in the real atlas
    g2m = np.concatenate([rng.normal(0.6, 0.15, n), rng.normal(0.3, 0.15, n)])
    pos = np.zeros(2 * n, bool); pos[:n] = True
    neg = ~pos

    before = abs(g2m[pos].mean() - g2m[neg].mean())
    ip, in_ = _g2m_matched(g2m, pos, neg, rng)
    assert ip is not None and len(ip) == len(in_)
    after = abs(g2m[ip].mean() - g2m[in_].mean())
    assert after < before / 3, (
        f"G2M matching barely reduced the imbalance ({before:.3f} -> "
        f"{after:.3f})")


def test_auroc_direction_convention():
    """AUROC below 0.5 must mean stroma-dominant, the convention the
    manuscript's wording depends on."""
    cancer = np.array([0.1, 0.2, 0.3])
    stroma = np.array([0.7, 0.8, 0.9])
    assert auroc_m(cancer, stroma) < 0.5
    assert auroc_m(stroma, cancer) > 0.5


# --------------------------------------------------------------- wp3r ------

@pytest.mark.parametrize("label,expected", [
    ("Cancer Epithelial", "Cancer Epithelial"),
    ("Malignant cells", "Cancer Epithelial"),
    ("Tumour epithelium", "Cancer Epithelial"),
    ("Endothelial", "Endothelial"),
    ("CAFs", "CAFs"),
    ("Fibroblast", "CAFs"),
    ("Pericyte", "PVL"),
    ("T-cells", None),
    ("Myeloid", None),
    ("Something unheard of", None),
])
def test_compartment_mapping(label, expected):
    assert _canonical(label) == expected


def test_immune_labels_never_become_stroma():
    """An immune label silently mapped into CAFs would corrupt the
    replication comparison, so it must map to None."""
    for lab in ("T-cells", "B cell", "Myeloid", "Plasmablasts", "Mast cells"):
        assert _canonical(lab) is None


def test_smooth_muscle_is_perivascular_not_endothelial():
    """A Cell Ontology label can contain 'vascular' without being endothelium.

    "vascular associated smooth muscle cell" is vessel-wall smooth muscle, i.e.
    perivascular. Because the map is substring-based and checked in order, the
    "vascular" -> Endothelial rule used to capture it first, silently inflating
    the endothelial compartment in the replication atlas. The mapping must
    therefore test "smooth muscle" before "vascular".
    """
    assert _canonical("vascular associated smooth muscle cell") == "PVL"
    assert _canonical("smooth muscle cell") == "PVL"
    # labels that mention vasculature without smooth muscle stay endothelial
    assert _canonical("endothelial cell of vascular tree") == "Endothelial"
    assert _canonical("vascular endothelial cell") == "Endothelial"


# ------------------------------------------------------- end-to-end wp8p ---

def _fixture(tmp_path, n_donors=8, seed=0):
    """A small atlas where the answer is known by construction."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(300)]
    planted = genes[:20]          # high in endothelium -> stroma-dominant
    flat = genes[20:40]           # no structure -> near 0.5

    rows, obs = [], []
    for d in range(n_donors):
        for comp, n in [("Cancer Epithelial", 60), ("Endothelial", 40),
                        ("CAFs", 40), ("PVL", 30)]:
            for _ in range(n):
                v = rng.poisson(2.0, len(genes)).astype(float)
                if comp == "Endothelial":
                    idx = [genes.index(g) for g in planted]
                    v[idx] += rng.poisson(20.0, len(planted))
                rows.append(v)
                obs.append({"donor_id": f"D{d}", "celltype_major": comp,
                            "subtype": "ER+"})
    X = np.vstack(rows)
    adata = ad.AnnData(X=X, obs=pd.DataFrame(obs).reset_index(drop=True),
                       var=pd.DataFrame(index=genes))
    adata.obs_names = [f"c{i}" for i in range(adata.n_obs)]
    p = tmp_path / "atlas.h5ad"
    adata.write_h5ad(p)

    panel = pd.DataFrame(
        [{"signature": "PLANTED_stromal", "gene": g, "direction": "up"}
         for g in planted]
        + [{"signature": "FLAT_control", "gene": g, "direction": "up"}
           for g in flat])
    return p, panel


def test_wp8p_recovers_a_planted_stromal_signature(tmp_path):
    """End-to-end: a signature planted in endothelium must come out with a
    NEGATIVE cancer-minus-endothelium difference in every donor, and the
    unstructured control must not."""
    from dormbench.config import Config, Paths
    from dormbench.steps import wp8p_pseudobulk

    atlas, panel = _fixture(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    panel.to_csv(results / "signature_panel.csv", index=False)

    paths = Paths(data_root=tmp_path, atlas_dir=tmp_path,
                  metabric_dir=tmp_path, scanb_dir=tmp_path,
                  rosano_dir=tmp_path, depmap_dir=tmp_path,
                  tcga_dir=tmp_path, poetic_sampling_control_dir=tmp_path,
                  extended_letrozole_dir=tmp_path, results=results,
                  figures=tmp_path / "figures", logs=tmp_path / "logs",
                  processed=tmp_path / "processed")
    cfg = Config(paths=paths, er_positive_labels=["ER+"], seed=42)

    out = wp8p_pseudobulk.run(cfg, n_boot=200)
    res = pd.read_csv(results / "wp8p_donor_level.csv")

    planted = res[(res["signature"] == "PLANTED_stromal|up")
                  & (res["compartment"] == "Endothelial")].iloc[0]
    assert planted["mean_diff"] < 0, "planted stromal signature not detected"
    assert planted["n_lower_in_cancer"] == planted["n_donors"], (
        "planted signal should be lower in cancer in EVERY donor")

    flat = res[(res["signature"] == "FLAT_control|up")
               & (res["compartment"] == "Endothelial")].iloc[0]
    assert abs(flat["mean_diff"]) < abs(planted["mean_diff"]), (
        "unstructured control separated as strongly as the planted signal")
    assert out["verdict"].startswith("PSEUDOBULK")
    assert out["counts_were_integer"] is True


# ---------------------------------------------------------------- wp2r -----

def test_auroc_from_ranks_matches_brute_force():
    """The fast rank-sum AUROC must equal the definition it replaces.

    Every null replicate in wp2r is evaluated with this shortcut, so a sign or
    off-by-one error here would corrupt the whole null distribution silently.
    """
    from scipy.stats import rankdata
    from dormbench.steps.wp2r_rare_state import _auroc_from_ranks

    rng = np.random.default_rng(1)
    for _ in range(20):
        n = 200
        v = rng.normal(size=n)
        tail = np.zeros(n, dtype=bool)
        tail[rng.choice(n, size=rng.integers(10, 60), replace=False)] = True

        r = rankdata(v)
        got = _auroc_from_ranks(r, tail)

        # brute force: P(tail value > rest value), ties counted as half
        pos, neg = v[tail], v[~tail]
        brute = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
        assert abs(got - brute) < 1e-9, (got, brute)


def test_tail_mask_selects_the_high_end_and_keeps_a_comparator():
    from dormbench.steps.wp2r_rare_state import _tail_mask, MIN_TAIL

    v = np.arange(1000, dtype=float)          # ascending
    m = _tail_mask(v, 0.05)
    assert m.sum() == 50
    assert m[-50:].all() and not m[:-50].any(), "tail is not the top of the score"
    # a pool too small to leave a comparator yields no tail at all, rather
    # than a tail that consumes the pool
    small = np.arange(6, dtype=float)
    ms = _tail_mask(small, 0.5)
    assert ms.sum() == 0 or ((~ms).sum() >= MIN_TAIL and ms.sum() >= MIN_TAIL)


def test_tail_mask_takes_high_scores_not_low():
    from dormbench.steps.wp2r_rare_state import _tail_mask

    v = np.array([0.0] * 90 + [5.0] * 10)
    m = _tail_mask(v, 0.10)
    assert m[90:].all() and not m[:90].any()


def _rare_fixture(tmp_path, n_donors=8, n_cells=400, specificity=True):
    """Score table where the tail phenotype is known by construction.

    A latent mesenchymal programme drives the pathway scores. When
    `specificity` is True that programme is planted ONLY in the TRADITIOM score
    (so the TRADITIOM tail should be the only one enriched); when False it
    drives TRADITIOM, PreAdapted and Latency equally, which is the state the
    real data turned out to be in.

    Three pathways are driven hard enough that they survive BH across the panel
    with 8 donors: the verdict needs at least three significant state pathways.
    """
    rng = np.random.default_rng(7)
    rows = []
    for d in range(n_donors):
        mes = rng.normal(size=n_cells)              # latent programme
        noise = lambda: rng.normal(size=n_cells)     # noqa: E731
        trad = mes + 0.3 * noise()
        pre = (mes if not specificity else noise()) + 0.3 * noise()
        lat = (mes if not specificity else noise()) + 0.3 * noise()
        emt = mes + 0.2 * noise()
        hyp = 0.8 * mes + 0.2 * noise()
        tgf = 0.8 * mes + 0.2 * noise()
        for i in range(n_cells):
            rows.append({
                "celltype_major": "Cancer Epithelial",
                "celltype_minor": "Cancer Epithelial",
                "subtype": "ER+", "donor_id": f"D{d}",
                "nCount_RNA": 5000, "nFeature_RNA": 2000,
                "AUCell|TRADITIOM_dormancy|up": trad[i],
                "AUCell|PreAdapted|up": pre[i],
                "AUCell|Latency|up": lat[i],
                "AUCell|EMT_hallmark|up": emt[i],
                "AUCell|Hypoxia_hallmark|up": hyp[i],
                "AUCell|TGFbeta_hallmark|up": tgf[i],
                "AUCell|P53_hallmark|up": rng.normal(),
                "AUCell|TNFa_NFkB_hallmark|up": rng.normal(),
                "AUCell|Quiescence_core|up": rng.normal(),
                "AUCell|Quiescence_core|down": rng.normal(),
                "AUCell|NR2F1_axis|up": rng.normal(),
                "AUCell|BHLHE41_dormancy|up": rng.normal(),
                "AUCell|TRADITIOM_dormancy|down": rng.normal(),
                "AUCell|Culture_drift|up": rng.normal(),
                "AUCell|E2F_control|up": rng.normal(),
                "AUCell|G2M_control|up": rng.normal(),
            })
    df = pd.DataFrame(rows)
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    df.to_csv(results / "wp2b_cell_scores.csv.gz", index=False,
              compression="gzip")
    return results


def _cfg(tmp_path, results):
    from dormbench.config import Config, Paths

    paths = Paths(data_root=tmp_path, atlas_dir=tmp_path,
                  metabric_dir=tmp_path, scanb_dir=tmp_path,
                  rosano_dir=tmp_path, depmap_dir=tmp_path,
                  tcga_dir=tmp_path, poetic_sampling_control_dir=tmp_path,
                  extended_letrozole_dir=tmp_path, results=results,
                  figures=tmp_path / "figures", logs=tmp_path / "logs",
                  processed=tmp_path / "processed")
    return Config(paths=paths, er_positive_labels=["ER+"], seed=42)


def test_wp2r_calls_a_shared_phenotype_not_specific(tmp_path):
    """When the tail phenotype is shared with correlated signatures, the step
    must refuse to call it a TRADITIOM state."""
    from dormbench.steps import wp2r_rare_state

    results = _rare_fixture(tmp_path, specificity=False)
    out = wp2r_rare_state.run(_cfg(tmp_path, results), stromal=False,
                              n_null=50)

    assert out["verdict"] == "REPRODUCIBLE_STATE_NOT_TRADITIOM_SPECIFIC", \
        out["verdict"]
    assert out["n_state_pathways_significant"] >= 1
    assert out["n_specific_state_pathways"] == 0


def test_wp2r_calls_a_specific_phenotype_specific(tmp_path):
    """The mirror case: a phenotype only the TRADITIOM tail carries must be
    reported as specific, so the test above is not passing vacuously."""
    from dormbench.steps import wp2r_rare_state

    results = _rare_fixture(tmp_path, specificity=True)
    out = wp2r_rare_state.run(_cfg(tmp_path, results), stromal=False,
                              n_null=50)

    assert out["n_specific_state_pathways"] > 0, out["specificity_excess"]
    assert out["verdict"] in ("REPRODUCIBLE_RARE_STATE", "PARTIAL_RARE_STATE"), \
        out["verdict"]
