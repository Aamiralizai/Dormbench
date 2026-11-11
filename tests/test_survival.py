"""Tests for the survival work package (wp4s).

The CTS5 implementation is pinned to published data rather than to constants
copied from a formula. Wimmer et al. 2024 (Breast Cancer Res Treat 205:227-239)
report the risk-group means for their three strata; a correct implementation
must place those means in the strata they report. This is the test that catches
a sign error in the quadratic size term, which is the one place two secondary
sources disagree and which silently inflates the score when wrong.
"""

from __future__ import annotations

import numpy as np

from dormbench.steps.wp4s_survival import (cts5_risk_group, cts5_score,
                                           _node_category, _ipcw_auc, _km)


def test_node_categories_match_published_ordinal_scale():
    """0, 1, 2-3, 4-9, >9 positive nodes -> 0,1,2,3,4."""
    assert _node_category(0) == 0
    assert _node_category(1) == 1
    assert _node_category(2) == 2 and _node_category(3) == 2
    assert _node_category(4) == 3 and _node_category(9) == 3
    assert _node_category(10) == 4 and _node_category(45) == 4


def test_cts5_reproduces_published_risk_group_means():
    """Wimmer 2024 Table 1 group means must land in the group they report.

    Low         nodes 0.1, size 12.0 mm, grade 1.76, age 62.5
    Intermediate nodes 0.4, size 21.0 mm, grade 2.11, age 64.6
    High        nodes 3.6, size 28.0 mm, grade 2.40, age 64.3

    With the quadratic term signed +0.001 the intermediate mean computes to
    4.23 and would be misclassified as high; with -0.001 it computes to 3.36,
    which is intermediate, as published.
    """
    low = cts5_score(0.1, 12.0, 1.76, 62.5)
    mid = cts5_score(0.4, 21.0, 2.11, 64.6)
    high = cts5_score(3.6, 28.0, 2.40, 64.3)

    assert cts5_risk_group(low) == "low", low
    assert cts5_risk_group(mid) == "intermediate", mid
    assert cts5_risk_group(high) == "high", high

    # explicit bands, so a future sign flip fails loudly rather than by class
    assert low < 3.13, low
    assert 3.13 <= mid <= 3.86, mid
    assert high > 3.86, high


def test_cts5_caps_tumour_size_at_30mm():
    """Above the cap the score must not keep rising."""
    a = cts5_score(0, 30.0, 2, 60)
    b = cts5_score(0, 60.0, 2, 60)
    assert a == b


def test_cts5_handles_missing_inputs():
    assert np.isnan(cts5_score(np.nan, 20, 2, 60))
    assert np.isnan(cts5_score(0, np.nan, 2, 60))
    assert cts5_risk_group(np.nan) == "unknown"


def test_km_survival_is_monotone_and_starts_at_one():
    rng = np.random.default_rng(0)
    t = rng.exponential(5, 200)
    e = rng.integers(0, 2, 200)
    grid = np.linspace(0.5, 10, 12)
    s = np.array([_km(t, e, x) for x in grid])
    assert np.all(s <= 1.0 + 1e-12)
    assert np.all(np.diff(s) <= 1e-12), "KM curve must not increase"


def test_ipcw_auc_detects_planted_direction():
    """A risk score proportional to the hazard must give AUC > 0.5."""
    rng = np.random.default_rng(1)
    n = 400
    hazard = rng.normal(size=n)
    t = rng.exponential(np.exp(-hazard))
    c = rng.exponential(8, n)                 # censoring
    obs = np.minimum(t, c)
    ev = (t <= c).astype(float)
    auc = _ipcw_auc(hazard, obs, ev, horizon=3.0)
    assert np.isfinite(auc)
    assert auc > 0.6, auc
