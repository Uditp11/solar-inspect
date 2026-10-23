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


def at_threshold(score: np.ndarray, is_anom: np.ndarray, t: float, prevalence: float) -> dict:
    """Flag where score >= t: counts as measured, and reweighted to field prevalence.

    The reweighting is the base-rate correction's (assumptions A1 and A2): with
    the sample's own No-Anomaly share q, an anomaly weighs prevalence / (1 - q)
    and a normal (1 - prevalence) / q. Recall is unchanged by it; precision and
    the flag rate per 1,000 modules are the field figures.
    """
    is_anom = np.asarray(is_anom, bool)
    q = 1.0 - is_anom.mean()
    w = np.where(is_anom, prevalence / (1.0 - q), (1.0 - prevalence) / q)
    f = np.asarray(score) >= t
    tp, fp = (w * (f & is_anom)).sum(), (w * (f & ~is_anom)).sum()
    return {"t": float(t), "flags": int(f.sum()),
            "flags_per_1000_measured": 1000.0 * f.mean(),
            "flags_per_1000_field": 1000.0 * (tp + fp) / w.sum(),
            "recall": float((f & is_anom).sum() / is_anom.sum()),
            "precision_measured": float((f & is_anom).sum() / f.sum()) if f.any() else float("nan"),
            "precision_field": float(tp / (tp + fp)) if f.any() else float("nan")}


def pick_threshold(score: np.ndarray, is_anom: np.ndarray, prevalence: float,
                   target: float) -> float | None:
    """The highest-recall threshold whose precision at field prevalence is >= target.

    Candidates are the distinct scores. Ties in recall go to the higher
    threshold. None when no candidate reaches the target: no fallback.
    """
    best = None
    for t in np.unique(score)[::-1]:                  # descending: recall never falls
        r = at_threshold(score, is_anom, t, prevalence)
        if r["precision_field"] >= target and (best is None or r["recall"] > best[1]):
            best = (float(t), r["recall"])
    return None if best is None else best[0]
