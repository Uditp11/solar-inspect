"""Ordinary least squares, two ways, both written here.

Both solvers take features without an intercept column and return
[intercept, slope_1, ..., slope_p]; the intercept is an explicit column of ones
added by `add_intercept`.

`normal_equations`: normal equations formed here; the linear solve is scipy's
Cholesky. XᵀX is symmetric positive definite when X has full column rank, so
Cholesky is the natural factorisation. No inverse is formed at any point. Forming
XᵀX squares the condition number of X, and on D4 irradiation and module
temperature are correlated at about 0.95, so that is where accuracy is lost first,
not the O(np² + p³) cost, which at p = 3 is nothing.

`gradient_descent`: batch gradient descent in NumPy only, on standardised
features, with the coefficients mapped back to original units. The step is 1/L,
L being the largest eigenvalue of the Hessian ZᵀZ/n of the mean squared loss.
It stops when the gradient norm falls to `tol` times its starting value, and
raises if that has not happened within `max_iter` steps.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def add_intercept(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(len(X)), X])


def normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = add_intercept(X)
    return cho_solve(cho_factor(A.T @ A), A.T @ np.asarray(y, dtype=float))


def gradient_descent(X: np.ndarray, y: np.ndarray, tol: float = 1e-10,
                     max_iter: int = 100_000) -> tuple[np.ndarray, int]:
    """Returns (coefficients in original units, number of steps taken)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu, sd = X.mean(0), X.std(0)
    Z = add_intercept((X - mu) / sd)
    n = len(y)
    lr = 1.0 / np.linalg.eigvalsh(Z.T @ Z / n).max()

    b = np.zeros(Z.shape[1])
    g = Z.T @ (Z @ b - y) / n
    stop = tol * np.linalg.norm(g)
    for step in range(max_iter):
        if np.linalg.norm(g) <= stop:
            break
        b -= lr * g
        g = Z.T @ (Z @ b - y) / n
    else:
        raise RuntimeError(f"gradient descent did not converge in {max_iter} steps "
                           f"(gradient norm {np.linalg.norm(g):.3g}, target {stop:.3g})")

    slopes = b[1:] / sd
    return np.concatenate([[b[0] - slopes @ mu], slopes]), step


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    return float(1.0 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def adjusted_r2(r2_value: float, n: int, p: int) -> float:
    """p is the number of slopes, not counting the intercept."""
    return 1.0 - (1.0 - r2_value) * (n - 1) / (n - p - 1)
