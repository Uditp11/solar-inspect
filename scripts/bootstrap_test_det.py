"""Bootstrap intervals for the single test evaluation, from its saved predictions only.

    python scripts/bootstrap_test_det.py runs/det_test_<run>

No second pass over the test split: reads the `predictions.npz` and `eval.json` that
`scripts/eval_test_det.py` wrote. Resamples the test images with replacement (2,000
resamples, seed 0) and recomputes, on each resample, AP@0.5 (101-point), AP@0.5:0.95
(101-point, ten IoU thresholds), and precision, recall (IoU 0.5) and panel-count MAE at
the config's `conf_operating`;
the interval is the 2.5th to 97.5th percentile. Images are the unit because
detections and ground truths within a frame are not independent.

Matching is done once. `ap.match` processes detections in global score order, but a
detection only competes for ground truths of its own image, and within an image the
global order is the image's own score order; so each detection's TP flag at a
threshold depends on its image alone. Each resample concatenates its images'
(score, TP) arrays, a drawn image's arrays once per draw, and integrates the
precision envelope as `ap.ap_from_matches` does. On the original sample this
reproduces eval.json's AP exactly, which the script checks.

Also prints the count errors of the frames with more than 300 ground-truth boxes.
Writes runs/det_boot_<utc>/bootstrap.json (gitignored).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.train import git_state                    # noqa: E402
from solar_inspect.detection.ap import COCO_IOU_THRESHOLDS, ap_from_matches  # noqa: E402
from solar_inspect.detection.ap import match                                 # noqa: E402

RESAMPLES = 2000
SEED = 0
DENSE = 300


def per_image_matches(boxes, scores, gts, thresholds) -> list[list[tuple]]:
    """[threshold][image] -> (scores, TP flags) of that image's detections."""
    out = []
    for t in thresholds:
        out.append([match([b], [s], [g], t)[:2] for b, s, g in zip(boxes, scores, gts)])
    return out


def ap(per_image: list[tuple], idx, n_gt: int) -> float:
    sc = np.concatenate([per_image[i][0] for i in idx])
    tp = np.concatenate([per_image[i][1] for i in idx])
    return ap_from_matches(tp[np.argsort(-sc, kind="stable")], n_gt, "101")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit(__doc__)
    test_dir = (REPO / argv[0]).resolve()
    e = json.loads((test_dir / "eval.json").read_text(encoding="utf-8"))
    if e["split"] != "test":
        raise SystemExit(f"{test_dir.name} is a {e['split']} evaluation, not the test one")
    conf = e["conf_operating"]
    sha, dirty = git_state()
    p = np.load(test_dir / "predictions.npz")
    names = list(p["names"])
    n = len(names)
    boxes = [p["boxes"][p["image"] == i] for i in range(n)]
    scores = [p["scores"][p["image"] == i] for i in range(n)]
    gts = [p["gt_boxes"][p["gt_image"] == i] for i in range(n)]
    gt_n = np.array([len(g) for g in gts])
    abs_err = np.abs(np.array([(s >= conf).sum() for s in scores]) - gt_n)

    pm = per_image_matches(boxes, scores, gts, COCO_IOU_THRESHOLDS)   # pm[0] is IoU 0.5
    tp_at = np.array([t[s >= conf].sum() for s, t in pm[0]])          # per image, at conf
    det_at = np.array([(s >= conf).sum() for s, _ in pm[0]])

    def stats(idx) -> tuple[float, float, float]:
        k = int(gt_n[idx].sum())
        aps = [ap(m, idx, k) for m in pm]
        return (aps[0], float(np.mean(aps)), tp_at[idx].sum() / det_at[idx].sum(),
                tp_at[idx].sum() / k, float(abs_err[idx].mean()))

    point = stats(np.arange(n))
    at = e["at_conf"][str(conf)]
    ref = (e["ap50_101_point"], e["ap50_95_101_point"], at["precision"], at["recall"],
           at["count_mae"])
    if not np.allclose(point, ref, atol=1e-12, rtol=0):
        raise SystemExit(f"per-image matching gives {point}, eval.json {ref}")

    rng = np.random.default_rng(SEED)
    draws = np.array([stats(rng.integers(0, n, n)) for _ in range(RESAMPLES)])
    keys = ("ap50_101_point", "ap50_95_101_point", f"precision_at_{conf}",
            f"recall_at_{conf}", f"count_mae_at_{conf}")
    res = {k: {"point": point[j], "ci95": np.percentile(draws[:, j], [2.5, 97.5]).tolist(),
               "bootstrap_std": float(draws[:, j].std(ddof=1))} for j, k in enumerate(keys)}
    signed = np.array([(s >= conf).sum() for s in scores]) - gt_n
    dense = [{"image": names[i], "gt": int(gt_n[i]), "predicted": int(gt_n[i] + signed[i]),
              "signed_error": int(signed[i])} for i in np.argsort(-gt_n) if gt_n[i] > DENSE]

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = REPO / "runs" / f"det_boot_{run}"
    dest.mkdir(parents=True)
    out = {"run": run, "git_sha": sha, "git_dirty": dirty,
           "test_run": test_dir.relative_to(REPO).as_posix(), "images": n,
           "boxes": int(gt_n.sum()), "resamples": RESAMPLES, "seed": SEED,
           "interval": "percentile, 2.5 to 97.5", "conf_operating": conf, "metrics": res,
           f"frames_over_{DENSE}_boxes": dense}
    (dest / "bootstrap.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8",
                                         newline="\n")

    print(f"test ({n} images, {int(gt_n.sum())} boxes), {RESAMPLES} resamples of images, "
          f"seed {SEED}, percentile 95% intervals")
    for k, v in res.items():
        print(f"  {k:<22}{v['point']:.4f}  [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")
    print(f"frames with more than {DENSE} boxes, at conf {conf}:")
    for d in dense:
        print(f"  {d['image']}: {d['gt']} boxes, {d['predicted']} counted, {d['signed_error']:+d}")
    print(f"\nwrote {dest.relative_to(REPO).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
