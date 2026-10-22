"""The from-scratch NMS on the detector's real boxes, against Ultralytics' NMS.

    python scripts/nms_crosscheck_det.py runs/det_nms_sweep_<run>

The boxes before NMS are taken from inside one Ultralytics validation pass (same
checkpoint, `conf`, `max_det` and preprocessing as the sweep), rather than from a
second inference path: `non_max_suppression` is wrapped so that each batch's raw head
output is copied before the library's NMS runs on it, and `update_metrics` is wrapped
to pair each copy with its image files. The copy is the tensor the library suppresses
(letterboxed pixel coordinates, not clipped to the frame, the model's own precision).
The candidates are the boxes the library itself keeps before NMS: score strictly
above `conf`, as Ultralytics filters.

`solar_inspect.detection.nms.nms` is then applied to each image's candidates at NMS
IoU 0.3, 0.5 and 0.7, and the per-image counts are compared with the sweep's
Ultralytics pass at the same threshold. IoU is invariant to the letterbox's uniform
scale and shift, so suppression in letterboxed coordinates is suppression in the
frame. Equal counts on every image mean the from-scratch NMS reproduces the
library's on real detections.

Reads val only. Writes runs/det_nms_xcheck_<utc>/ (gitignored): the pass and
nms_crosscheck.json.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.train import git_state   # noqa: E402
from solar_inspect.detection.evaluate import evaluate       # noqa: E402
from solar_inspect.detection.nms import nms                 # noqa: E402

CONFIG = "configs/det_yolo.yaml"
THRESHOLDS = (0.3, 0.5, 0.7)


def capture() -> dict[str, np.ndarray]:
    """Wraps Ultralytics' NMS and metric update; returns {image file name: (N, 5)
    candidates as xyxy + score}, filled during the next validation pass."""
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import nms as unms
    from ultralytics.utils.ops import xywh2xyxy

    pending, cands = [], {}
    original_nms, original_update = unms.non_max_suppression, DetectionValidator.update_metrics

    def wrapped_nms(prediction, conf_thres=0.25, *args, **kwargs):
        p = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
        p = p.detach().transpose(-1, -2).clone()                # (B, N, 4 + nc)
        p[..., :4] = xywh2xyxy(p[..., :4])
        score = p[..., 4:].amax(-1)
        pending.append([np.column_stack([b[s > conf_thres, :4].double().cpu().numpy(),
                                         s[s > conf_thres].double().cpu().numpy()])
                        for b, s in zip(p, score)])
        return original_nms(prediction, conf_thres, *args, **kwargs)

    def wrapped_update(self, preds, batch):
        # The validator calls NMS once more than it updates metrics (a batch-of-one call
        # before the loop), so each update is paired with the NMS call just before it.
        last = pending[-1]
        pending.clear()
        if len(last) != len(batch["im_file"]):
            raise SystemExit("captured NMS input does not match the batch")
        for f, c in zip(batch["im_file"], last):
            cands[Path(f).name] = c
        return original_update(self, preds, batch)

    unms.non_max_suppression = wrapped_nms
    DetectionValidator.update_metrics = wrapped_update
    return cands


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit(__doc__)
    sweep_dir = (REPO / argv[0]).resolve()
    sw = json.loads((sweep_dir / "nms_sweep.json").read_text(encoding="utf-8"))
    sha, dirty = git_state()
    cfg = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = REPO / "runs" / f"det_nms_xcheck_{run}"

    cands = capture()
    evaluate(REPO / sw["weights"], "val", {**cfg, "conf": sw["conf"], "iou": 0.7}, dest, "pass")
    names = sw["names"]
    if sorted(cands) != sorted(names):
        raise SystemExit("captured images do not match the sweep's")

    library = {r["iou"]: np.array(r["per_image_count"]) for r in sw["sweep"]}
    rows = []
    for t in THRESHOLDS:
        ours = np.array([len(nms(cands[n][:, :4], cands[n][:, 4], t)) for n in names])
        diff = ours - library[t]
        rows.append({"iou": t, "candidates": int(sum(len(c) for c in cands.values())),
                     "detections_from_scratch": int(ours.sum()),
                     "detections_ultralytics": int(library[t].sum()),
                     "images_equal": int((diff == 0).sum()), "images": len(names),
                     "differences": {names[i]: int(diff[i]) for i in np.flatnonzero(diff)}})
    out = {"run": run, "git_sha": sha, "git_dirty": dirty,
           "sweep_run": sweep_dir.relative_to(REPO).as_posix(), "weights": sw["weights"],
           "split": "val", "conf": sw["conf"], "torchvision_nms": "torchvision" in sys.modules,
           "rows": rows}
    (dest / "nms_crosscheck.json").write_text(json.dumps(out, indent=2) + "\n",
                                              encoding="utf-8", newline="\n")

    print(f"\nval ({len(names)} images), conf {sw['conf']}, "
          f"{rows[0]['candidates']} candidates before NMS")
    for r in rows:
        print(f"iou {r['iou']}: from scratch {r['detections_from_scratch']}, Ultralytics "
              f"{r['detections_ultralytics']}, images equal {r['images_equal']}/{r['images']}"
              + (f", differences {r['differences']}" if r["differences"] else ""))
    print(f"\nwrote {dest.relative_to(REPO).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
