"""Evaluate a trained detection checkpoint on the val split.

    python scripts/eval_val_det.py runs/det_<run>/weights/best.pt
    python scripts/eval_val_det.py runs/det_<run>/weights/best.pt max_det=3000

`key=value` overrides after the weights path are applied over the config and
recorded in `eval.json`, the same way `train_det.py` takes them.

One Ultralytics validation pass with the config's `conf`, `iou` and `max_det`, then
Ultralytics' mAP50 and mAP50-95, the from-scratch AP@0.5 (all-point and 101-point),
torchmetrics' mAP@0.5 and mAP@0.5:0.95, precision, recall and per-image panel-count
error at a few confidence thresholds, all from the same saved predictions (see
`solar_inspect.detection.evaluate`). Writes `runs/det_val_<run>/` with
`predictions.json`, `predictions.npz` for later sweeps, and `eval.json`.
Reads val only; `splits()` refuses if val resolves to the test tree.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.train import git_state   # noqa: E402
from solar_inspect.detection.evaluate import evaluate       # noqa: E402
from train_det import overrides                              # noqa: E402

CONFIG = "configs/det_yolo.yaml"


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit(__doc__)
    weights = Path(argv[0]).resolve()
    over = overrides(argv[1:])
    sha, dirty = git_state()
    cfg = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
    if set(over) - set(cfg):     # a mistyped key would be recorded but never applied
        raise SystemExit(f"unknown config keys: {sorted(set(over) - set(cfg))}")
    cfg.update(over)
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    out = evaluate(weights, "val", cfg, REPO / "runs", f"det_val_{run}")
    out.update({"run": run, "git_sha": sha, "git_dirty": dirty, "config_path": CONFIG,
                "overrides": over, "config": cfg,
                "weights": weights.relative_to(REPO).as_posix()})
    dest = REPO / out["save_dir"]
    (dest / "eval.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8",
                                    newline="\n")

    u, tm = out["ultralytics"], out["torchmetrics"]
    print(f"\n=== val ({out['images']} images, {out['boxes']} boxes), "
          f"{out['detections']} detections, max {out['max_detections_per_image']}/image ===")
    print(f"AP@0.5     Ultralytics {u['map50']:.4f} | from scratch all-point "
          f"{out['ap50_all_point']:.4f}, 101-point {out['ap50_101_point']:.4f} | "
          f"torchmetrics {tm['map_50']:.4f}")
    print(f"AP@0.5:0.95 Ultralytics {u['map50_95']:.4f} | from scratch 101-point "
          f"{out['ap50_95_101_point']:.4f} | torchmetrics {tm['map_50_95']:.4f}")
    print(f"from scratch minus torchmetrics, 101-point: AP@0.5 "
          f"{out['ap50_101_point'] - tm['map_50']:+.1e}, AP@0.5:0.95 "
          f"{out['ap50_95_101_point'] - tm['map_50_95']:+.1e}")
    print(f"Ultralytics P {u['precision']:.4f} R {u['recall']:.4f} (its own max-F1 confidence)")
    print(f"{'conf':>6}{'precision':>11}{'recall':>8}{'count MAE':>11}{'signed mean':>13}"
          f"{'signed range':>16}")
    for c, r in out["at_conf"].items():
        print(f"{c:>6}{r['precision']:>11.4f}{r['recall']:>8.4f}{r['count_mae']:>11.2f}"
              f"{r['count_signed_mean']:>13.2f}"
              f"{'[' + str(r['count_signed_min']) + ', ' + str(r['count_signed_max']) + ']':>16}")
    print(f"\nwrote {out['save_dir']}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
