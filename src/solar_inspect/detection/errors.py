"""Classes of false positive for the post-hoc error analysis, NumPy only.

`classify(det_boxes, tp, gt_boxes)` labels the detections of one frame, given the
true-positive flags `ap.match` gave them at IoU 0.5. Each false positive gets the
first class whose test it passes:

1. `duplicate`: IoU > 0.3 with a true positive of the same frame (a second box on a
   panel already found, or a box spanning two found panels);
2. `localisation`: best IoU with any ground-truth box in [0.1, 0.5);
3. `background`: best IoU with every ground-truth box < 0.1 (a false alarm or an
   unlabelled panel; the labels cannot tell which).

A false positive whose best ground-truth IoU is >= 0.5 lost that ground truth to a
higher-scoring detection; if it overlaps no true positive by more than 0.3 (two
halves of one panel do not) it passes none of the three tests and is labelled
`unclassified`, so the analysis can report how many there are.
"""
from __future__ import annotations

import numpy as np

from solar_inspect.detection.boxes import iou

DUPLICATE_IOU = 0.3
LOCALISATION_IOU = (0.1, 0.5)


def classify(det_boxes, tp, gt_boxes) -> np.ndarray:
    tp = np.asarray(tp, dtype=bool)
    out = np.where(tp, "tp", "unclassified").astype("<U12")
    fp = ~tp
    if not fp.any():
        return out
    to_tp = iou(det_boxes, np.asarray(det_boxes)[tp]).max(axis=1, initial=0.0)
    to_gt = iou(det_boxes, gt_boxes).max(axis=1, initial=0.0)
    lo, hi = LOCALISATION_IOU
    out[fp & (to_gt < lo)] = "background"
    out[fp & (to_gt >= lo) & (to_gt < hi)] = "localisation"
    out[fp & (to_tp > DUPLICATE_IOU)] = "duplicate"     # last, so it wins: tested first
    return out
