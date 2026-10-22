"""Pairwise intersection-over-union for axis-aligned boxes, NumPy only.

Boxes are `xyxy`: [x1, y1, x2, y2] with x2 >= x1 and y2 >= y1, in continuous
coordinates. Area is (x2 - x1)(y2 - y1), with no "+1" pixel convention, so two
boxes that share an edge intersect in a line of zero area and have IoU 0. That is
the case of panels tiled in a row, and the reason default NMS does not merge
neighbouring panels on D2.

IoU is defined as 0 when the union is 0 (two zero-area boxes), rather than 0/0.
"""
from __future__ import annotations

import numpy as np


def area(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    return (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in `a` (N, 4) with every box in `b` (M, 4), shape (N, M)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
