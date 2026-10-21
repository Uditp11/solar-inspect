"""OLS by normal equations and by batch gradient descent, tested before either exists.

The small case is worked by hand:

    x = [0, 1, 2, 3], y = [1, 2, 2, 4]
    x̄ = 1.5, ȳ = 2.25, Sxx = Σ(x - x̄)² = 5, Sxy = Σ(x - x̄)(y - ȳ) = 4.5
    slope = Sxy / Sxx = 0.9, intercept = ȳ - 0.9 x̄ = 0.9
    ŷ = [0.9, 1.8, 2.7, 3.6], residuals = [0.1, 0.2, -0.7, 0.4]
    SSres = 0.70, SStot = 4.75, R² = 1 - 0.70/4.75 = 81/95
    adjusted R² (n = 4, p = 1) = 1 - (14/95)(3/2) = 74/95

The rest use exactly-known coefficients on noiseless data, or a matrix built from
orthonormal columns so its singular values, and so its condition number, are set
by construction.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from solar_inspect.analytics.ols import (add_intercept, adjusted_r2, gradient_descent,
                                         normal_equations, r2)

X_HAND = np.array([[0.0], [1.0], [2.0], [3.0]])
Y_HAND = np.array([1.0, 2.0, 2.0, 4.0])


def correlated(n: int, seed: int) -> np.ndarray:
    """Three features on very different scales, the first two correlated at about 0.95."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 0.95 * a + np.sqrt(1 - 0.95 ** 2) * rng.normal(size=n)
    c = rng.normal(size=n)
    return np.column_stack([0.5 + 0.3 * a, 40 + 10 * b, 25 + 5 * c])


def test_intercept_is_a_leading_column_of_ones():
    assert add_intercept(np.array([[2.0], [3.0]])).tolist() == [[1.0, 2.0], [1.0, 3.0]]


def test_normal_equations_on_the_hand_worked_case():
    assert normal_equations(X_HAND, Y_HAND) == pytest.approx([0.9, 0.9], abs=1e-12)


def test_r2_and_adjusted_r2_on_the_hand_worked_case():
    yhat = add_intercept(X_HAND) @ np.array([0.9, 0.9])
    assert r2(Y_HAND, yhat) == pytest.approx(81 / 95, abs=1e-12)
    assert adjusted_r2(81 / 95, n=4, p=1) == pytest.approx(74 / 95, abs=1e-12)


def test_a_constant_target_is_all_intercept():
    X = correlated(200, seed=1)
    assert normal_equations(X, np.full(200, 7.0)) == pytest.approx([7, 0, 0, 0], abs=1e-8)


def test_normal_equations_recover_known_coefficients_on_noiseless_data():
    X = correlated(500, seed=2)
    beta = np.array([2.0, 3.0, -0.5, 1.25])
    np.testing.assert_allclose(normal_equations(X, add_intercept(X) @ beta), beta, atol=1e-8)


def test_normal_equations_never_form_an_inverse(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("an explicit inverse was formed")
    for mod in (np.linalg, scipy.linalg):
        monkeypatch.setattr(mod, "inv", refuse)
        monkeypatch.setattr(mod, "pinv", refuse)
    normal_equations(X_HAND, Y_HAND)


def test_gradient_descent_maps_back_to_original_units():
    # Standardised internally; the coefficients that come back must be in raw units.
    X = correlated(500, seed=3)
    beta = np.array([2.0, 3.0, -0.5, 1.25])
    got, n_iter = gradient_descent(X, add_intercept(X) @ beta)
    np.testing.assert_allclose(got, beta, rtol=1e-6, atol=1e-6)
    assert isinstance(n_iter, int) and n_iter > 0


def test_gradient_descent_agrees_with_normal_equations_on_noisy_data():
    # Stated tolerance: 1e-6 relative, 1e-6 absolute, per coefficient.
    X = correlated(2000, seed=4)
    y = add_intercept(X) @ np.array([-10.0, 800.0, 4.0, -2.0])
    y = y + np.random.default_rng(5).normal(scale=30.0, size=len(y))
    gd, _ = gradient_descent(X, y)
    np.testing.assert_allclose(gd, normal_equations(X, y), rtol=1e-6, atol=1e-6)


def test_gradient_descent_raises_when_it_does_not_converge():
    X = correlated(200, seed=6)
    with pytest.raises(RuntimeError):
        gradient_descent(X, X @ np.array([1.0, 2.0, 3.0]), max_iter=3)


def test_forming_xtx_squares_the_condition_number():
    # Orthonormal columns times singular values 10, 1, 0.01: cond(X) is 1e3 by
    # construction, so cond(XᵀX) is 1e6.
    q, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(300, 3)))
    X = q * np.array([10.0, 1.0, 0.01])
    cond_x, cond_xtx = np.linalg.cond(X), np.linalg.cond(X.T @ X)
    assert cond_x == pytest.approx(1e3, rel=1e-9)
    assert cond_xtx == pytest.approx(1e6, rel=1e-6)
    assert cond_xtx == pytest.approx(cond_x ** 2, rel=1e-6)
    # Still solvable at that conditioning, to a tolerance loosened by the square.
    beta = np.array([1.0, -2.0, 0.5, 3.0])
    np.testing.assert_allclose(normal_equations(X, add_intercept(X) @ beta), beta, atol=1e-6)
