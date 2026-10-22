"""The operating confidence of the selected detector, chosen on val from saved predictions.

    python scripts/operating_point_det.py runs/det_val_<run>

Reads the `predictions.npz` and `eval.json` that `scripts/eval_val_det.py` wrote for
one val evaluation (conf 0.001, NMS IoU 0.7, max_det 3000) and, on the grid
0.05, 0.10, ..., 0.95, computes per-image panel-count error (MAE, signed mean, min,
max) and precision and recall at IoU 0.5. The operating confidence is the grid value
with the lowest count MAE; on a tie, the lower confidence, which keeps recall. The
rule was declared in docs/EXPERIMENTS.md before the tuning pass ran.

No new inference. Keeping the boxes with score >= t from a pass at conf 0.001 gives
exactly the boxes a pass at conf t would return. NMS visits boxes in descending
score order and a box is only ever suppressed by a box kept before it, i.e. one with
a higher (or equal) score. So a box scoring >= t is suppressed, or not, by boxes that
also score >= t, and the boxes below t, which a pass at conf t never sees, cannot
change its fate. The one way this breaks is `max_det` truncation, which cuts the low
end of a frame's list at 0.001 and not at t; the script refuses unless every frame's
count is below `max_det`.

Reads val only (refuses an eval.json whose split is not val). Writes
runs/det_oppoint_<utc>/ (gitignored): operating_point.json and a figure of count
MAE and signed mean against confidence.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.train import git_state   # noqa: E402
from solar_inspect.detection.ap import match                # noqa: E402
from solar_inspect.detection.count import count_error       # noqa: E402

GRID = tuple(np.round(np.arange(1, 20) * 0.05, 2))


def load(eval_dir: Path) -> tuple[dict, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """eval.json, and per image: detection boxes, scores, ground-truth boxes."""
    e = json.loads((eval_dir / "eval.json").read_text(encoding="utf-8"))
    if e["split"] != "val":
        raise SystemExit(f"{eval_dir.name} is a {e['split']} evaluation; this reads val only")
    p = np.load(eval_dir / "predictions.npz")
    n = len(p["names"])
    boxes = [p["boxes"][p["image"] == i] for i in range(n)]
    scores = [p["scores"][p["image"] == i] for i in range(n)]
    gts = [p["gt_boxes"][p["gt_image"] == i] for i in range(n)]
    return e, boxes, scores, gts


def table(boxes, scores, gts, grid=GRID) -> list[dict]:
    sc, tp, n_gt = match(boxes, scores, gts)
    gt_counts = [len(g) for g in gts]
    rows = []
    for c in grid:
        k = sc >= c
        signed, _, mae = count_error(scores, gt_counts, c)
        rows.append({"conf": float(c), "count_mae": mae,
                     "count_signed_mean": float(signed.mean()),
                     "count_signed_min": int(signed.min()), "count_signed_max": int(signed.max()),
                     "precision": float(tp[k].mean()) if k.any() else float("nan"),
                     "recall": float(tp[k].sum() / n_gt)})
    return rows


def operating(rows: list[dict]) -> float:
    """Lowest count MAE; on a tie the lower confidence (rows are in ascending conf)."""
    return min(rows, key=lambda r: (round(r["count_mae"], 9), r["conf"]))["conf"]


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit(__doc__)
    eval_dir = (REPO / argv[0]).resolve()
    sha, dirty = git_state()
    e, boxes, scores, gts = load(eval_dir)
    cfg = e["config"]
    if e["max_detections_per_image"] >= cfg["max_det"]:
        raise SystemExit(f"a frame reached max_det {cfg['max_det']}; filtering by score is "
                         "not equivalent to a pass at a higher confidence")
    rows = table(boxes, scores, gts)
    op = operating(rows)

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = REPO / "runs" / f"det_oppoint_{run}"
    dest.mkdir(parents=True)
    out = {"run": run, "git_sha": sha, "git_dirty": dirty,
           "eval_run": eval_dir.relative_to(REPO).as_posix(), "weights": e["weights"],
           "split": "val", "images": e["images"], "boxes": e["boxes"],
           "pass": {k: cfg[k] for k in ("conf", "iou", "max_det")},
           "max_detections_per_image": e["max_detections_per_image"],
           "rule": "lowest count MAE on the grid; on a tie, the lower confidence",
           "conf_operating": op, "grid": rows}
    (dest / "operating_point.json").write_text(json.dumps(out, indent=2) + "\n",
                                               encoding="utf-8", newline="\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = [r["conf"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(c, [r["count_mae"] for r in rows], "o-", label="count MAE")
    ax.plot(c, [r["count_signed_mean"] for r in rows], "s--", label="signed mean (pred - GT)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(op, color="grey", ls=":", label=f"operating conf {op:.2f}")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("panels per image")
    ax.set_title(f"val, {e['images']} images, {e['boxes']:,} boxes, NMS IoU {cfg['iou']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "count_error_vs_conf.png", dpi=110)

    print(f"val ({e['images']} images, {e['boxes']} boxes), {e['weights']}, "
          f"pass conf {cfg['conf']} iou {cfg['iou']} max_det {cfg['max_det']}")
    print(f"{'conf':>6}{'count MAE':>11}{'signed mean':>13}{'min':>6}{'max':>6}"
          f"{'precision':>11}{'recall':>8}")
    for r in rows:
        print(f"{r['conf']:>6.2f}{r['count_mae']:>11.3f}{r['count_signed_mean']:>13.3f}"
              f"{r['count_signed_min']:>6}{r['count_signed_max']:>6}"
              f"{r['precision']:>11.4f}{r['recall']:>8.4f}")
    print(f"\noperating confidence: {op:.2f}\nwrote {dest.relative_to(REPO).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
