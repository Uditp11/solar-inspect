"""Cross-check the from-scratch AP@0.5 against torchmetrics on seeded synthetic boxes.

    python scripts/crosscheck_ap.py

The synthetic frames are a test fixture for the metric, not data: they exist only
so the two implementations can be compared on inputs where everything is known.
Each frame gets 0 to 600 ground-truth boxes of D2's median size (56 x 37 px) placed
uniformly at random in a 640 x 512 frame, so unlike tiled panels they overlap one
another heavily; each ground truth is found with probability 0.85 as a jittered
copy, a tenth of the found ones get a lower-scored duplicate, and unmatched false
positives are added at a fifth of the ground-truth count. Scores are noisy and
clipped to [0.001, 1], so TP and FP interleave and some scores tie.

Two comparisons per seed:

- raw: the from-scratch all-point and 101-point AP against torchmetrics' AP@0.5.
- controlled: the same inputs with the three differences found during
  reconciliation removed from the from-scratch side's inputs, which should agree
  to float32 precision. (1) Score ties are broken into unique ranks, since the order
  of equal scores is implementation-defined. (2) Boxes are seen as torchmetrics
  passes them to pycocotools, float32 x, y, w, h, so IoUs near 0.5 and near-ties
  between two ground truths round the same way. (3) Ground truths are reversed per
  image, because on an exact IoU tie between two ground truths pycocotools' loop
  keeps the last and `ap.match` keeps the first.

torchmetrics' top max-detections is raised to 1000 (its default is 100); see
`solar_inspect.detection.reference`.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.detection.ap import average_precision     # noqa: E402
from solar_inspect.detection.reference import torchmetrics_map  # noqa: E402

W, H, BW, BH = 640, 512, 56.0, 37.0


def frames(seed: int, n_img: int = 40):
    rng = np.random.default_rng(seed)
    dets, scores, gts = [], [], []
    for _ in range(n_img):
        n = int(rng.integers(0, 601))
        xy = rng.uniform(0, [W - BW, H - BH], size=(n, 2))
        g = np.column_stack([xy, xy + [BW, BH]])
        found = g[rng.uniform(size=n) < 0.85]
        tp = found + rng.normal(0, 6, size=found.shape)
        dup = found[rng.uniform(size=len(found)) < 0.1] + rng.normal(0, 4, size=(1, 4))
        fxy = rng.uniform(0, [W - BW, H - BH], size=(n // 5, 2))
        fp = np.column_stack([fxy, fxy + [BW, BH]])
        d = np.concatenate([tp, dup, fp]).reshape(-1, 4)
        d[:, 2:] = np.maximum(d[:, 2:], d[:, :2] + 1)
        s = np.concatenate([rng.normal(0.7, 0.15, len(tp)), rng.normal(0.5, 0.15, len(dup)),
                            rng.normal(0.4, 0.15, len(fp))]).clip(0.001, 1)
        dets.append(d)
        scores.append(s)
        gts.append(g)
    return dets, scores, gts


def unique_ranks(scores, seed: int):
    """Same order wherever scores differ, ties broken at random, all values distinct."""
    flat = np.concatenate(scores)
    key = np.lexsort((np.random.default_rng(seed).uniform(size=len(flat)), flat))
    r = np.empty(len(flat))
    r[key] = (np.arange(len(flat)) + 1) / (len(flat) + 1)
    return np.split(r.astype(np.float32), np.cumsum([len(s) for s in scores])[:-1])


def as_pycocotools_sees(boxes):
    out = []
    for b in boxes:
        b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
        xy = b[:, :2].astype(float)
        out.append(np.column_stack([xy, xy + (b[:, 2:] - b[:, :2]).astype(float)]))
    return out


def main() -> int:
    warnings.filterwarnings("ignore", category=UserWarning)
    print(f"{'seed':>4} {'images':>6} {'gt':>6} {'dets':>6} {'max/img':>7} | raw: "
          f"{'all-point':>9} {'101-point':>9} {'torchmetrics':>12} {'101 - tm':>9} | "
          f"controlled: {'101 - tm':>9}")
    for seed in range(5):
        d, s, g = frames(seed)
        ap_all = average_precision(d, s, g, method="all")
        ap101 = average_precision(d, s, g, method="101")
        tm = torchmetrics_map(d, s, g, iou_thresholds=[0.5])["map_50"]

        su = unique_ranks(s, seed)
        ctl = average_precision(as_pycocotools_sees(d), [x.astype(float) for x in su],
                                [x[::-1] for x in as_pycocotools_sees(g)], method="101")
        tm_ctl = torchmetrics_map(d, su, g, iou_thresholds=[0.5])["map_50"]
        print(f"{seed:>4} {len(g):>6} {sum(map(len, g)):>6} {sum(map(len, d)):>6} "
              f"{max(map(len, d)):>7} |      {ap_all:>9.6f} {ap101:>9.6f} {tm:>12.6f} "
              f"{ap101 - tm:>+9.1e} |             {ctl - tm_ctl:>+9.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
