"""One evaluation pass of a trained YOLO checkpoint on one split, scored three ways.

Ultralytics' validator runs inference once with the config's `conf`, `iou` and
`max_det`, reports its own mAP50 and mAP50-95, and writes the boxes it scored to
`predictions.json`. The from-scratch AP@0.5 (both interpolations), torchmetrics'
mAP@0.5 and mAP@0.5:0.95, precision, recall and panel-count error are then all
computed from that same file, so the three AP numbers differ only in how they are
computed, never in which predictions they see. The predictions are saved again as
arrays (`predictions.npz`) so later sweeps reuse them without another pass.

Ultralytics computes its numbers differently from both others, and the gaps are
expected rather than bugs: it matches detections to ground truth greedily by IoU
within an image rather than by score, and integrates a linearly interpolated
101-point curve with the trapezoid rule. `predictions.json` rounds boxes to 3
decimals and scores to 5, which creates a few score ties.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from solar_inspect.detection.ap import ap_from_matches, match
from solar_inspect.detection.count import count_error
from solar_inspect.detection.data import REPO, ground_truth, splits
from solar_inspect.detection.reference import torchmetrics_map

CONF_THRESHOLDS = (0.1, 0.25, 0.5)


def evaluate(weights: Path, split: str, cfg: dict, project: Path, name: str) -> dict:
    from ultralytics import YOLO

    data = REPO / cfg["data"]
    images = splits(data)[split]
    m = YOLO(str(weights)).val(data=str(data), split=split, imgsz=cfg["imgsz"],
                               batch=cfg["batch"], device=cfg["device"], conf=cfg["conf"],
                               iou=cfg["iou"], max_det=cfg["max_det"], save_json=True,
                               project=str(project), name=name, exist_ok=False)
    save_dir = Path(m.save_dir)

    names, gts = ground_truth(images)
    pj = save_dir / "predictions.json"
    by = {n: [] for n in names}
    for r in (json.loads(pj.read_text(encoding="utf-8")) if pj.exists() else []):
        by[r["file_name"]].append(r)
    boxes = [np.array([[x, y, x + w, y + h] for x, y, w, h in (r["bbox"] for r in by[n])],
                      dtype=float).reshape(-1, 4) for n in names]
    scores = [np.array([r["score"] for r in by[n]], dtype=float) for n in names]

    sc, tp, n_gt = match(boxes, scores, gts)
    tm = torchmetrics_map(boxes, scores, gts)
    out = {
        "split": split, "images": len(names), "boxes": n_gt,
        "detections": int(sum(map(len, scores))),
        "max_detections_per_image": int(max(map(len, scores), default=0)),
        "ultralytics": {"map50": float(m.box.map50), "map50_95": float(m.box.map),
                        "precision": float(m.box.mp), "recall": float(m.box.mr)},
        "ap50_all_point": ap_from_matches(tp, n_gt, "all"),
        "ap50_101_point": ap_from_matches(tp, n_gt, "101"),
        "torchmetrics": tm,
        "at_conf": {},
    }
    gt_counts = [len(g) for g in gts]
    for c in CONF_THRESHOLDS:
        k = sc >= c
        signed, _, mae = count_error(scores, gt_counts, c)
        out["at_conf"][str(c)] = {
            "precision": float(tp[k].mean()) if k.any() else float("nan"),
            "recall": float(tp[k].sum() / n_gt),
            "count_mae": mae, "count_signed_mean": float(signed.mean()),
            "count_signed_min": int(signed.min()), "count_signed_max": int(signed.max()),
        }

    np.savez_compressed(
        save_dir / "predictions.npz", names=np.array(names),
        image=np.concatenate([np.full(len(s), i) for i, s in enumerate(scores)] + [[]]).astype(int),
        boxes=np.concatenate(boxes + [np.zeros((0, 4))]),
        scores=np.concatenate(scores + [np.zeros(0)]),
        gt_image=np.concatenate([np.full(len(g), i) for i, g in enumerate(gts)] + [[]]).astype(int),
        gt_boxes=np.concatenate(gts + [np.zeros((0, 4))]))
    out["save_dir"] = save_dir.relative_to(REPO).as_posix()
    return out
