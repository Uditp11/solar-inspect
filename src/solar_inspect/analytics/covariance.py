"""Coefficient covariance for OLS: classical and Newey-West (HAC), both written here.

Both take the design matrix as fitted (intercept column included) and the
residuals, rows in time order. (AᵀA)⁻¹ is the one inverse that cannot be
avoided, because it is the answer: it is computed by scipy's Cholesky solve
against the identity, never by a general inverse.

`classical_cov`: σ² (AᵀA)⁻¹ with σ² = Σu² / (n - k). Valid only if the errors
are homoscedastic and uncorrelated; 15-minute D4 residuals are neither.
Cross-checked by hand and against statsmodels `OLS(...).fit().cov_params()`.

`hac_cov`: (AᵀA)⁻¹ S (AᵀA)⁻¹ with the Newey-West meat
S = Γ_0 + Σ_(l=1..L) w_l (Γ_l + Γ_lᵀ), Γ_l = Σ_t u_t u_(t-l) a_t a_(t-l)ᵀ,
Bartlett weights w_l = 1 - l / (L + 1), and no small-sample factor. Lags count
rows, so the caller's row order is the time axis and a gap is not filled.
Lags = 0 is White's HC0. Cross-checked by hand at lags = 0 and against
statsmodels `cov_type="HAC"` with `use_correction=False` (the same definition)
to 1e-10. With `groups` (one label per row, each group's rows contiguous and in
time order), a lag product is only taken between two rows of the same group:
several series stacked one after another, as in a plant-level fit over all its
inverters. Cross-checked against statsmodels `cov_type="hac-panel"` with
`use_correction=False` to 1e-10.

`newey_west_lags`: floor(4 (n / 100)^(2/9)), the Newey-West rule of thumb.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _bread(A: np.ndarray) -> np.ndarray:
    return cho_solve(cho_factor(A.T @ A), np.eye(A.shape[1]))


def classical_cov(A: np.ndarray, u: np.ndarray) -> np.ndarray:
    A, u = np.asarray(A, dtype=float), np.asarray(u, dtype=float)
    n, k = A.shape
    return (u @ u) / (n - k) * _bread(A)


def hac_cov(A: np.ndarray, u: np.ndarray, lags: int, groups=None) -> np.ndarray:
    A, u = np.asarray(A, dtype=float), np.asarray(u, dtype=float)
    g = A * u[:, None]
    S = g.T @ g
    for lag in range(1, lags + 1):
        same = slice(None) if groups is None else groups[lag:] == groups[:-lag]
        gamma = g[lag:][same].T @ g[:-lag][same]
        S += (1 - lag / (lags + 1)) * (gamma + gamma.T)
    B = _bread(A)
    return B @ S @ B


def newey_west_lags(n: int) -> int:
    return math.floor(4 * (n / 100) ** (2 / 9))
