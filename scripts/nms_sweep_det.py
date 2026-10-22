"""Panel-count error against the NMS IoU threshold, on val, at the operating confidence.

    python scripts/nms_sweep_det.py runs/det_oppoint_<run>

Takes the checkpoint and the operating confidence from the operating-point run
(`scripts/operating_point_det.py`), then runs one Ultralytics validation pass per NMS
IoU threshold 0.30, 0.35, ..., 0.80 through `solar_inspect.detection.evaluate`, each
with `conf=<operating>` and `iou=<t>` over configs/det_yolo.yaml. The detections a
pass returns are the panels it counts: per image, count error against ground truth
(MAE, signed mean, min, max) and the frames whose count differs from the 0.7 pass.

Declared in docs/EXPERIMENTS.md as a sweep that is reported, not a selection: the
config's `iou` stays 0.7 whatever the curve shows.

Reads val only (`splits()` refuses if val resolves to the test tree). Writes
runs/det_nms_sweep_<utc>/ (gitignored): one pass directory per threshold,
nms_sweep.json, and a figure of count MAE and signed mean against the threshold.
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

CONFIG = "configs/det_yolo.yaml"
THRESHOLDS = tuple(np.round(np.arange(30, 81, 5) / 100, 2))
REFERENCE = 0.7


def counts(pass_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per image: detections the pass returned, ground-truth boxes."""
    p = np.load(pass_dir / "predictions.npz")
    n = len(p["names"])
    return np.bincount(p["image"], minlength=n), np.bincount(p["gt_image"], minlength=n)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit(__doc__)
    op_dir = (REPO / argv[0]).resolve()
    op = json.loads((op_dir / "operating_point.json").read_text(encoding="utf-8"))
    weights, conf = REPO / op["weights"], op["conf_operating"]
    sha, dirty = git_state()
    cfg = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = REPO / "runs" / f"det_nms_sweep_{run}"

    per_t = {}
    for t in THRESHOLDS:
        out = evaluate(weights, "val", {**cfg, "conf": conf, "iou": float(t)}, dest,
                       f"iou_{t:.2f}")
        pred, gt = counts(REPO / out["save_dir"])
        per_t[float(t)] = {"pred": pred, "gt": gt,
                           "max_detections_per_image": out["max_detections_per_image"]}
    ref = per_t[REFERENCE]["pred"]
    rows = []
    for t, v in per_t.items():
        signed = v["pred"] - v["gt"]
        rows.append({"iou": t, "count_mae": float(np.abs(signed).mean()),
                     "count_signed_mean": float(signed.mean()),
                     "count_signed_min": int(signed.min()), "count_signed_max": int(signed.max()),
                     "detections": int(v["pred"].sum()),
                     "frames_changed_from_0.7": int((v["pred"] != ref).sum()),
                     "max_detections_per_image": v["max_detections_per_image"],
                     "per_image_count": v["pred"].tolist()})

    names = list(np.load(dest / f"iou_{REFERENCE:.2f}" / "predictions.npz")["names"])
    res = {"run": run, "git_sha": sha, "git_dirty": dirty, "operating_point_run":
           op_dir.relative_to(REPO).as_posix(), "weights": op["weights"], "split": "val",
           "images": len(names), "boxes": int(per_t[REFERENCE]["gt"].sum()), "conf": conf,
           "max_det": cfg["max_det"], "names": names,
           "gt_count": per_t[REFERENCE]["gt"].tolist(), "sweep": rows}
    (dest / "nms_sweep.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8",
                                         newline="\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["iou"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(x, [r["count_mae"] for r in rows], "o-", label="count MAE")
    ax.plot(x, [r["count_signed_mean"] for r in rows], "s--", label="signed mean (pred - GT)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(REFERENCE, color="grey", ls=":", label=f"config iou {REFERENCE}")
    ax.set_xlabel("NMS IoU threshold")
    ax.set_ylabel("panels per image")
    ax.set_title(f"val, {res['images']} images, {res['boxes']:,} boxes, conf {conf}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "count_error_vs_nms_iou.png", dpi=110)

    print(f"\nval ({res['images']} images, {res['boxes']} boxes), {op['weights']}, conf {conf}")
    print(f"{'iou':>6}{'count MAE':>11}{'signed mean':>13}{'min':>6}{'max':>6}"
          f"{'detections':>12}{'changed vs 0.7':>16}")
    for r in rows:
        print(f"{r['iou']:>6.2f}{r['count_mae']:>11.3f}{r['count_signed_mean']:>13.3f}"
              f"{r['count_signed_min']:>6}{r['count_signed_max']:>6}{r['detections']:>12}"
              f"{r['frames_changed_from_0.7']:>16}")
    print(f"\nwrote {dest.relative_to(REPO).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
