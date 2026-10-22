"""Per-image panel-count error at a confidence threshold, NumPy only.

The domain metric: an inspection report states how many panels a frame holds, and
AP says nothing about whether that count is right. A detection counts when its
score is >= `conf`. Signed error is predicted minus ground-truth count, so negative
means panels missed and positive means extra boxes; MAE is the mean absolute error
over images.
"""
from __future__ import annotations

import numpy as np


def count_error(det_scores, gt_counts, conf: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (signed error per image, absolute error per image, MAE)."""
    pred = np.array([int((np.asarray(s) >= conf).sum()) for s in det_scores], dtype=int)
    signed = pred - np.asarray(gt_counts, dtype=int)
    absolute = np.abs(signed)
    return signed, absolute, float(absolute.mean()) if len(absolute) else float("nan")
