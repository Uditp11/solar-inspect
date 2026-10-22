"""Library reference values for the from-scratch detection metrics.

torchmetrics' MeanAveragePrecision (pycocotools backend) is the library half of the
seam: it is the source of mAP@0.5:0.95, and at IoU 0.5 it is the cross-check for
`ap.average_precision(method="101")`.

Two traps, both measured on this torchmetrics (1.9.0) and pycocotools (2.0.11):

- The top max-detections default is 100 per image. D2 frames carry up to 584
  boxes, so at the default the reference silently scores a truncated list. It is
  raised to 1000 here.
- Once it is raised, the summary's `map` is -1: pycocotools computes that entry
  with `maxDets=100` hard-coded, and 100 is no longer one of the settings. So both
  values are read from the per-threshold precision tensor (extended summary) at
  the 1000 slot, averaging the entries that are not -1, which is what pycocotools'
  own summary does. Checked against `map` and `map_50` on inputs of at most 100
  detections per image, where both are defined: equal to float32 precision.
"""
from __future__ import annotations

import numpy as np
import torch

MAX_DETS = [1, 10, 1000]


def torchmetrics_map(det_boxes, det_scores, gt_boxes, iou_thresholds=None) -> dict:
    """{"map_50", "map_50_95" (when the default thresholds are used), "mar"} at 1000
    detections per image, over per-image xyxy arrays, single class."""
    from torchmetrics.detection import MeanAveragePrecision

    m = MeanAveragePrecision(box_format="xyxy", iou_thresholds=iou_thresholds,
                             max_detection_thresholds=MAX_DETS, extended_summary=True)
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))   # noqa: E731
    m.update([{"boxes": t(b).reshape(-1, 4), "scores": t(s),
               "labels": torch.zeros(len(s), dtype=torch.int64)}
              for b, s in zip(det_boxes, det_scores)],
             [{"boxes": t(g).reshape(-1, 4), "labels": torch.zeros(len(g), dtype=torch.int64)}
              for g in gt_boxes])
    r = m.compute()
    thr = np.asarray(m.iou_thresholds)
    p = r["precision"][:, :, 0, 0, -1].numpy()          # (IoU thresholds, 101 recall points)
    mean = lambda a: float(a[a > -1].mean()) if (a > -1).any() else float("nan")  # noqa: E731
    out = {"map_50": mean(p[np.isclose(thr, 0.5)]), "mar": float(r["mar_1000"])}
    if iou_thresholds is None:
        out["map_50_95"] = mean(p)
    return out
