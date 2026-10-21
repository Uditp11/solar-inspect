"""Durbin-Watson and VIF, tested before either exists.

Durbin-Watson, by hand, on residuals e = [1, -1, 2, 0]:

    differences e_t - e_(t-1) = [-2, 3, -2], squared and summed = 4 + 9 + 4 = 17
    Σ e_t² = 1 + 1 + 4 + 0 = 6
    DW = 17 / 6

Alternating residuals ±1 give (n - 1)·4 / n, which tends to 4; a slow random walk
has steps far smaller than its level, so DW tends to 0.

VIF_j = 1 / (1 - R²_j), R²_j from regressing column j on the others plus an
intercept. The collinear pair reuses the hand-worked OLS case of test_ols.py:
x = [0, 1, 2, 3] against [1, 2, 2, 4] gives R² = 81/95 whichever way round, so
both VIFs are 1 / (14/95) = 95/14.
"""
from __future__ import annotations

import numpy as np
import pytest

from solar_inspect.analytics.diagnostics import durbin_watson, vif


def test_durbin_watson_hand_example():
    assert durbin_watson(np.array([1.0, -1.0, 2.0, 0.0])) == pytest.approx(17 / 6, abs=1e-15)


def test_durbin_watson_alternating_residuals_near_four():
    e = np.where(np.arange(1000) % 2 == 0, 1.0, -1.0)
    assert durbin_watson(e) == pytest.approx(4 * 999 / 1000, abs=1e-12)
    assert durbin_watson(e) > 3.99


def test_durbin_watson_slow_random_walk_near_zero():
    e = np.cumsum(np.random.default_rng(0).normal(size=10_000))
    assert 0 <= durbin_watson(e) < 0.01


def test_vif_orthogonal_columns_is_exactly_one():
    X = np.array([[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])
    assert vif(X).tolist() == [1.0, 1.0]


def test_vif_hand_built_collinear_pair():
    X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(vif(X), [95 / 14, 95 / 14], rtol=1e-12)
