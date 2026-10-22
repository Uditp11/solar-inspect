"""Greedy non-maximum suppression, NumPy only.

Boxes are sorted by score, descending. The highest is kept and every remaining box
whose IoU with it is **strictly greater** than `iou_thr` is suppressed; repeat on
what is left. A box at IoU exactly equal to the threshold survives, which is
torchvision's convention (`torchvision.ops.nms`), so the two agree box for box.

Ties: the sort is stable on the negated score, so of two boxes with equal scores
the one with the lower input index is considered first. Kept indices are returned
in the order they were kept, i.e. by descending score.

O(N^2) in the worst case. At D2's few hundred boxes per frame that is nothing.
"""
from __future__ import annotations

import numpy as np

from solar_inspect.detection.boxes import iou


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Indices into `boxes` (N, 4, xyxy) that survive, by descending score."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        order = rest[iou(boxes[i], boxes[rest])[0] <= iou_thr]
    return np.array(keep, dtype=np.int64)
