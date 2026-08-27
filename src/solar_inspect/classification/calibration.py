"""Calibration primitives: softmax at a temperature, NLL, and ECE.

Here rather than in the script that calls them because they are hand-written
metric maths and tests/test_cls_metrics.py needs to import them. Same reason
leakage.py is a module and not a block inside scripts/leakage_check_d1.py.
"""
from __future__ import annotations

import numpy as np

BINS = 15


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(z: np.ndarray, y: np.ndarray, T: float) -> float:
    p = softmax(z, T)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = BINS) -> float:
    """Expected calibration error: sum over bins of |accuracy - confidence| * share.

    Equal-width bins on the predicted probability of the predicted class. Hand
    checkable: a model that is right 70% of the time in the bin it calls 0.9
    contributes 0.2 times that bin's share of the data.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)
