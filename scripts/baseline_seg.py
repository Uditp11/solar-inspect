"""The naive baseline for PV01: one colour threshold, chosen on train, scored on val.

    python scripts/baseline_seg.py

"What would a threshold on the blue channel get?" The family, declared here:

    feature   R, G, B (0..255) or the index 2B - R - G (-510..510; twice
              B minus mean(R, G), kept integer)
    direction panel where feature >= t, or where feature <= t
    t         every integer in the feature's range

The rule with the best **mean of the three per-subset train IoUs** is chosen (the
same criterion that picks the U-Net's epoch on val), ties to the first in the
order above. Guard tiles are not used. The chosen rule is then scored per subset
on **val**. Test is not scored here.

The search runs on per-value histograms: for each subset, how many panel and
background pixels take each feature value. A cumulative sum then gives tp and fp
for every threshold at once. The chosen rule is re-applied pixel by pixel and
counted with metrics.counts as a cross-check; the two must agree exactly.

Also recomputes the null all-background model per split: IoU 0, pixel accuracy
1 - foreground fraction. It predicts the same thing everywhere and selects
nothing, so computing it on test spends nothing.

Writes configs/seg_baseline.json (committed: the family, the chosen rule, its
train IoUs, its val scores, the null model per split).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.segmentation.data import SUBSETS, load_d3     # noqa: E402
from solar_inspect.segmentation.metrics import counts, scores    # noqa: E402

OUT = REPO / "configs" / "seg_baseline.json"
FEATURES = {"R": (0, 255), "G": (0, 255), "B": (0, 255), "2B-R-G": (-510, 510)}


def feature(rgb: np.ndarray, name: str) -> np.ndarray:
    """rgb (N, 3, H, W) uint8 -> int32 (N, H, W)."""
    c = rgb.astype(np.int32)
    if name == "2B-R-G":
        return 2 * c[:, 2] - c[:, 0] - c[:, 1]
    return c[:, "RGB".index(name)]


def iou_per_threshold(f: np.ndarray, m: np.ndarray, lo: int, hi: int
                      ) -> dict[str, np.ndarray]:
    """IoU of "f >= t" and of "f <= t" for every t in lo..hi."""
    n = hi - lo + 1
    pos = np.bincount(f[m] - lo, minlength=n)
    neg = np.bincount(f[~m] - lo, minlength=n)
    total_pos = pos.sum()
    out = {}
    # >= t: pixels at value t and above. Suffix sums.
    tp, fp = pos[::-1].cumsum()[::-1], neg[::-1].cumsum()[::-1]
    out[">="] = tp / np.maximum(tp + fp + (total_pos - tp), 1)
    # <= t: prefix sums.
    tp, fp = pos.cumsum(), neg.cumsum()
    out["<="] = tp / np.maximum(tp + fp + (total_pos - tp), 1)
    return out


def main() -> int:
    d = load_d3()
    img = d.images.cpu().numpy()
    msk = d.masks[:, 0].cpu().numpy().astype(bool)
    rows = {(sp, s): d.index[(sp, s)].cpu().numpy() for sp in ("train", "val", "test") for s in SUBSETS}

    best = None
    for name, (lo, hi) in FEATURES.items():
        per_sub = [iou_per_threshold(feature(img[rows[("train", s)]], name),
                                     msk[rows[("train", s)]], lo, hi) for s in SUBSETS]
        for direction in (">=", "<="):
            mean = np.mean([p[direction] for p in per_sub], axis=0)
            k = int(np.argmax(mean))
            print(f"{name:>7} {direction}  best t = {lo + k:>4}  mean train IoU {mean[k]:.4f}  "
                  + "  ".join(f"{s} {p[direction][k]:.4f}" for s, p in zip(SUBSETS, per_sub)))
            if best is None or mean[k] > best["mean_train_iou"]:
                best = {"feature": name, "direction": direction, "threshold": lo + k,
                        "mean_train_iou": float(mean[k]),
                        "train_iou": {s: float(p[direction][k]) for s, p in zip(SUBSETS, per_sub)}}

    def predict(r: np.ndarray) -> np.ndarray:
        f = feature(img[r], best["feature"])
        return f >= best["threshold"] if best["direction"] == ">=" else f <= best["threshold"]

    for s in SUBSETS:       # cross-check the histogram search against a direct count
        r = rows[("train", s)]
        direct = scores(*counts(predict(r), msk[r]))["iou"]
        assert abs(direct - best["train_iou"][s]) < 1e-12, (s, direct, best["train_iou"][s])

    val = {s: scores(*counts(predict(rows[("val", s)]), msk[rows[("val", s)]])) for s in SUBSETS}
    null = {sp: {s: {"foreground": float(msk[rows[(sp, s)]].mean()),
                     "pixel_accuracy": float(1 - msk[rows[(sp, s)]].mean()), "iou": 0.0}
                 for s in SUBSETS} for sp in ("train", "val", "test")}

    rule = f"panel where {best['feature']} {best['direction']} {best['threshold']}"
    print(f"\nchosen on train: {rule}  (mean train IoU {best['mean_train_iou']:.4f})")
    print(f"{'subset':<13}{'val IoU':>9}{'Dice':>8}{'prec':>8}{'recall':>8}")
    for s in SUBSETS:
        v = val[s]
        print(f"{s:<13}{v['iou']:>9.4f}{v['dice']:>8.4f}{v['precision']:>8.4f}{v['recall']:>8.4f}")
    print(f"mean val IoU {np.mean([val[s]['iou'] for s in SUBSETS]):.4f}")
    print("null all-background, pixel accuracy (IoU 0):")
    for sp in ("train", "val", "test"):
        print(f"  {sp:<6}" + "  ".join(f"{s} {null[sp][s]['pixel_accuracy']:.4f}" for s in SUBSETS))

    OUT.write_text(json.dumps({
        "_": "GENERATED by scripts/baseline_seg.py -- do not hand-edit.",
        "split_sha256": d.split_sha256,
        "family": {"features": {k: list(v) for k, v in FEATURES.items()},
                   "directions": [">=", "<="], "thresholds": "every integer in range",
                   "criterion": "mean of the three per-subset train IoUs"},
        "rule": rule, "chosen": best,
        "val": val, "mean_val_iou": float(np.mean([val[s]["iou"] for s in SUBSETS])),
        "null_all_background": null,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
