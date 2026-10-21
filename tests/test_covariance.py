"""Classical and Newey-West (HAC) OLS covariance, tested before either exists.

Both are worked by hand on the OLS case of test_ols.py, x = [0, 1, 2, 3],
y = [1, 2, 2, 4], design A = [1, x], residuals u = [0.1, 0.2, -0.7, 0.4]:

    AᵀA = [[4, 6], [6, 14]], det 20, (AᵀA)⁻¹ = [[0.7, -0.3], [-0.3, 0.2]]

Classical: σ² = SSres / (n - k) = 0.70 / 2 = 0.35, so
    Var(intercept) = 0.35 · 0.7 = 0.245, Var(slope) = 0.35 · 0.2 = 0.07.

HAC with lags = 0 is White's HC0, (AᵀA)⁻¹ S (AᵀA)⁻¹ with S = Σ u_t² a_t a_tᵀ:
    u² = [0.01, 0.04, 0.49, 0.16]
    S = [[Σu², Σu²x], [Σu²x, Σu²x²]] = [[0.70, 1.50], [1.50, 3.44]]
    S (AᵀA)⁻¹ = [[0.04, 0.09], [0.018, 0.238]]
    (AᵀA)⁻¹ S (AᵀA)⁻¹ = [[0.0226, -0.0084], [-0.0084, 0.0206]]

On random data both are compared with statsmodels. Its HAC is the same
definition when `use_correction=False` (statsmodels' default for "HAC", passed
explicitly here): Bartlett weights 1 - l/(L+1) and no n/(n-k) factor.
"""
from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from solar_inspect.analytics.covariance import classical_cov, hac_cov, newey_west_lags
from solar_inspect.analytics.ols import add_intercept, normal_equations

A_HAND = add_intercept(np.array([[0.0], [1.0], [2.0], [3.0]]))
U_HAND = np.array([0.1, 0.2, -0.7, 0.4])


def ar1_data(n: int, seed: int):
    """Two regressors and AR(1) errors, so the lag terms of HAC are not negligible."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.8 * e[t - 1] + rng.normal()
    y = 1.0 + X @ np.array([2.0, -0.5]) + e
    return X, y


def test_classical_cov_hand_example():
    np.testing.assert_allclose(classical_cov(A_HAND, U_HAND),
                               [[0.245, -0.105], [-0.105, 0.07]], atol=1e-14)


def test_hac_with_zero_lags_is_hc0_by_hand():
    np.testing.assert_allclose(hac_cov(A_HAND, U_HAND, lags=0),
                               [[0.0226, -0.0084], [-0.0084, 0.0206]], atol=1e-14)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_classical_cov_matches_statsmodels(seed):
    X, y = ar1_data(500, seed)
    A = add_intercept(X)
    u = y - A @ normal_equations(X, y)
    ref = sm.OLS(y, A).fit().cov_params()
    np.testing.assert_allclose(classical_cov(A, u), ref, rtol=0, atol=1e-10)


@pytest.mark.parametrize("seed,lags", [(0, 0), (1, 1), (2, 7), (3, 25)])
def test_hac_matches_statsmodels(seed, lags):
    X, y = ar1_data(500, seed)
    A = add_intercept(X)
    u = y - A @ normal_equations(X, y)
    ref = sm.OLS(y, A).fit(cov_type="HAC",
                           cov_kwds={"maxlags": lags, "use_correction": False}).cov_params()
    np.testing.assert_allclose(hac_cov(A, u, lags), ref, rtol=0, atol=1e-10)


@pytest.mark.parametrize("lags", [1, 7])
def test_hac_within_groups_matches_statsmodels_hac_panel(lags):
    """Three series stacked one after another: lag products never cross a series boundary."""
    X, y = ar1_data(600, 4)
    groups = np.repeat([0, 1, 2], [150, 250, 200])
    A = add_intercept(X)
    u = y - A @ normal_equations(X, y)
    ref = sm.OLS(y, A).fit(cov_type="hac-panel", cov_kwds={
        "groups": groups, "maxlags": lags, "use_correction": False}).cov_params()
    np.testing.assert_allclose(hac_cov(A, u, lags, groups=groups), ref, rtol=0, atol=1e-10)
    assert np.abs(hac_cov(A, u, lags) - ref).max() > 1e-6   # the boundaries do matter


def test_hac_lags_rule_of_thumb():
    # floor(4 (n/100)^(2/9)): 4·1 = 4; 4·10^(2/9) = 6.67; 4·32^(2/9) = 4·2^(10/9) = 8.64
    assert [newey_west_lags(n) for n in (100, 1000, 3200)] == [4, 6, 8]
