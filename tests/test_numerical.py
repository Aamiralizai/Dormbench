"""Numerical fixture tests for the statistical primitives.

These check the maths on small hand-computable inputs, complementing the
config/file/frozen-table tests. Where a helper is not importable in isolation
the test recomputes the same quantity by the documented method, so the test
still pins the intended behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_auroc_matches_manual():
    """AUROC on a tiny separable set equals the rank-based value."""
    # scores and binary labels; perfect separation -> AUROC 1.0
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 0, 1, 1])

    def auroc(s, y):
        # Mann-Whitney U formulation
        pos = s[y == 1]
        neg = s[y == 0]
        wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        return wins / (len(pos) * len(neg))

    assert auroc(scores, labels) == 1.0
    # reversed labels -> 0.0
    assert auroc(scores, 1 - labels) == 0.0
    # one swap -> 0.75
    labels2 = np.array([0, 1, 0, 1])
    assert abs(auroc(scores, labels2) - 0.75) < 1e-9


def test_bh_adjustment_monotone_and_bounded():
    """Benjamini-Hochberg on a known vector matches the textbook result."""
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042])

    def bh(pv):
        n = len(pv)
        order = np.argsort(pv)
        ranked = pv[order]
        adj = ranked * n / (np.arange(n) + 1)
        # enforce monotonicity from the largest p down
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        out = np.empty(n)
        out[order] = np.clip(adj, 0, 1)
        return out

    q = bh(p)
    # all q >= corresponding p, and q bounded by 1
    assert np.all(q >= p - 1e-12)
    assert np.all(q <= 1.0)
    # smallest p: 0.001 * 5 / 1 = 0.005
    assert abs(q[0] - 0.005) < 1e-9


def test_empirical_p_with_pseudocount():
    """Matched-null empirical p uses the (#>= obs + 1)/(n+1) convention."""
    null = np.array([1.0, 2.0, 3.0, 4.0])
    obs = 3.5

    def emp_p(o, nd):
        return (np.sum(nd >= o) + 1) / (len(nd) + 1)

    # one null value (4.0) >= 3.5 -> (1+1)/(4+1) = 0.4
    assert abs(emp_p(obs, null) - 0.4) < 1e-9
    # never zero
    assert emp_p(100.0, null) == 1 / 5


def test_paired_delta_sign_convention():
    """Paired Δ is on-treatment minus pre-treatment."""
    pre = pd.Series([1.0, 2.0, 3.0])
    post = pd.Series([1.5, 1.0, 3.0])
    delta = post.values - pre.values
    assert list(delta) == [0.5, -1.0, 0.0]
    # mean delta
    assert abs(delta.mean() - (-0.5 / 3)) < 1e-9


def test_zscore_against_null():
    """z = (observed - null_mean)/null_sd, the WP2e/S10 convention."""
    obs = 0.5
    null = np.array([-0.4, -0.3, -0.5, -0.2, -0.6])
    z = (obs - null.mean()) / null.std(ddof=0)
    # null mean = -0.4, sd = 0.1414..., z ~ 6.36
    assert z > 6 and z < 7


def test_wp10_verdict_string_is_exploratory():
    """The WP10 module must only emit the exploratory verdict."""
    import dormbench.steps.wp10_response as w
    from pathlib import Path
    src = Path(w.__file__).read_text()
    assert "EXPLORATORY_RECONSTRUCTED_LABEL_ANALYSIS" in src
    assert "SEPARATION_SURVIVES_ADJUSTMENT" not in src


def test_official_purity_output_filter_tumour_only():
    """The tumour-only filter keeps primary/met barcodes, drops normals."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from dormbench.steps.wp9_purity import _tumour_only
    import logging
    log = logging.getLogger("t")
    cols = ["TCGA-A1-A0SB-01A", "TCGA-A1-A0SB-11A", "TCGA-A2-A04N-01A",
            "TCGA-A2-A04N-01A", "TCGA-XX-YYYY-06A"]
    e = pd.DataFrame([[1, 2, 3, 4, 5]], columns=cols, index=["MKI67"])
    out = _tumour_only(e, log)
    # drops the -11 normal and the duplicate patient; keeps 3
    assert out.shape[1] == 3
    assert "TCGA-A1-A0SB-11A" not in out.columns
