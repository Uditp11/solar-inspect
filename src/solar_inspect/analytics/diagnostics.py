"""Residual and design diagnostics for OLS: Durbin-Watson and VIF, both written here.

`durbin_watson`: Σ(e_t - e_(t-1))² / Σ e_t², in NumPy, on residuals in the order
given. About 2 for uncorrelated residuals, towards 0 for positive and towards 4
for negative first-order autocorrelation. The order is the caller's: a gap in
the series (the night break, a missing timestamp) is not filled, so the pair
across it counts as one step. Cross-checked by hand and at both limits in
tests/test_diagnostics.py.

`vif`: 1 / (1 - R²_j), R²_j from regressing column j on the other columns plus
an intercept with this project's normal-equations OLS (whose linear solve is
scipy's Cholesky). Cross-checked against hand-worked values.
"""
from __future__ import annotations

import numpy as np

from solar_inspect.analytics.ols import add_intercept, normal_equations, r2


def durbin_watson(e: np.ndarray) -> float:
    e = np.asarray(e, dtype=float)
    return float((np.diff(e) ** 2).sum() / (e ** 2).sum())


def vif(X: np.ndarray) -> np.ndarray:
    """One value per column of X (features only, no intercept column)."""
    X = np.asarray(X, dtype=float)
    out = []
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        fitted = add_intercept(others) @ normal_equations(others, X[:, j])
        out.append(1.0 / (1.0 - r2(X[:, j], fitted)))
    return np.array(out)
