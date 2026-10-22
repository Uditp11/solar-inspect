"""Average precision at one IoU threshold (AP@0.5 by default), NumPy only.

Inputs are per image, in the same image order: `det_boxes[i]` (N_i, 4, xyxy),
`det_scores[i]` (N_i,), `gt_boxes[i]` (M_i, 4). Single class.

Matching. Detections from **all images are pooled and sorted by score globally**,
descending; the sort is stable over the concatenation in image order, so ties keep
image order and then within-image order. In that order, each detection is matched
to the **highest-IoU ground truth in its own image that is still unmatched**, if
that IoU is **>= iou_thr**; otherwise it is a false positive. A second detection on
an already matched ground truth is therefore a false positive (unless another
unmatched one qualifies, which is COCO's rule; VOC's devkit takes the best ground
truth overall and calls the detection a false positive if that one is taken).
On an exact IoU tie between two unmatched ground truths the lower index wins;
pycocotools' loop keeps the last, which is one of the reasons the two can differ
when ground truths overlap one another.

Integration. Precision and recall are taken at every rank, precision is replaced by
its envelope (the running maximum from the right, so it is monotone
non-increasing in rank), and then:

- `method="all"`: all-point interpolation, the VOC (2010+) definition: the sum over
  recall steps of the step width times the envelope there.
- `method="101"`: 101-point interpolation, the COCO definition and the one used for
  the headline number: the mean of the envelope at the first rank whose recall
  reaches r, for r = 0, 0.01, ..., 1, and 0 where recall never reaches r. It is
  also the one torchmetrics computes, so it is the one cross-checked against it.

No detections gives AP 0. No ground truth anywhere leaves AP undefined, and NaN is
returned.

`mean_ap` averages the 101-point AP over IoU thresholds, by default COCO's ten,
0.50 to 0.95 in steps of 0.05, which is AP@0.5:0.95. Each threshold is matched
afresh, so a detection can be a true positive at 0.5 and a false positive at 0.75.
"""
from __future__ import annotations

import numpy as np

from solar_inspect.detection.boxes import iou


def match(det_boxes, det_scores, gt_boxes, iou_thr: float = 0.5
          ) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (scores sorted descending, TP flag per detection in that order, n_gt)."""
    img = np.concatenate([np.full(len(s), i) for i, s in enumerate(det_scores)]
                         + [np.zeros(0, dtype=int)]).astype(int)
    idx = np.concatenate([np.arange(len(s)) for s in det_scores]
                         + [np.zeros(0, dtype=int)]).astype(int)
    scores = np.concatenate([np.asarray(s, dtype=float) for s in det_scores] + [np.zeros(0)])
    order = np.argsort(-scores, kind="stable")

    taken = [np.zeros(len(g), dtype=bool) for g in gt_boxes]
    ious = [iou(d, g) for d, g in zip(det_boxes, gt_boxes)]
    tp = np.zeros(len(order), dtype=bool)
    for r, k in enumerate(order):
        i = img[k]
        if not len(taken[i]):
            continue
        cand = np.where(taken[i], -1.0, ious[i][idx[k]])
        j = int(cand.argmax())
        if cand[j] >= iou_thr:
            taken[i][j] = True
            tp[r] = True
    return scores[order], tp, int(sum(len(g) for g in gt_boxes))


def ap_from_matches(tp: np.ndarray, n_gt: int, method: str = "all") -> float:
    if method not in ("all", "101"):
        raise ValueError(f"method must be 'all' or '101', not {method!r}")
    if n_gt == 0:
        return float("nan")
    if not len(tp):
        return 0.0
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(tp) + 1)
    env = np.maximum.accumulate(precision[::-1])[::-1]
    if method == "all":
        return float(np.sum(np.diff(recall, prepend=0.0) * env))
    rs = np.linspace(0, 1, 101)
    at = np.searchsorted(recall, rs, side="left")
    return float(np.where(at < len(env), env[np.minimum(at, len(env) - 1)], 0.0).mean())


def average_precision(det_boxes, det_scores, gt_boxes, iou_thr: float = 0.5,
                      method: str = "all") -> float:
    _, tp, n_gt = match(det_boxes, det_scores, gt_boxes, iou_thr)
    return ap_from_matches(tp, n_gt, method)


# 0.50, 0.55, ..., 0.95, as pycocotools defines them
COCO_IOU_THRESHOLDS = tuple(np.linspace(0.5, 0.95, 10))


def mean_ap(det_boxes, det_scores, gt_boxes, thresholds=COCO_IOU_THRESHOLDS) -> float:
    """101-point AP from `match` at each IoU threshold, averaged: AP@0.5:0.95 with the
    default thresholds."""
    return float(np.mean([average_precision(det_boxes, det_scores, gt_boxes, t, "101")
                          for t in thresholds]))
