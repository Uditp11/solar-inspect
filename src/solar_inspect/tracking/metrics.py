"""Tracking metrics for D5: motmetrics, plus two counts written out by hand.

A sequence is a list of `Frame`s: the visible ground truth (ids and xyxy boxes), the
ignore boxes (labelled panels in the window that fail the visibility rule), and the
tracker's hypotheses (the detections it reports an id for). Definitions are declared
in docs/EXPERIMENTS.md, "D5 tracking metrics and sweep".

- `ignore_filter`: a hypothesis below IoU 0.5 with every visible box and at least 0.5
  with some ignore box is dropped before the frame is scored.
- motmetrics (library): MOTA, IDF1, IDSW, FP, FN, from `MOTAccumulator`'s CLEAR MOT
  matching at IoU >= 0.5. Distances are `motmetrics.distances.boxiou` with
  `iou_matrix`'s rule written out: `iou_matrix` calls `np.asfarray`, gone in NumPy 2.
- `clear_mot_matches` + `id_switches` (hand-written, scipy does the assignment): the
  same matching, then for each ground-truth id a switch each time its matched track id
  differs from the last one it was matched to. Must equal motmetrics' `num_switches`.
- `count_error` (hand-written): K distinct reported ids left after the ignore rule,
  G distinct ground-truth ids visible at least once.

On synthetic sequences MOTA and IDF1 check the implementation; they are not a
performance result.
"""
from __future__ import annotations

from typing import NamedTuple

import motmetrics as mm
import numpy as np
from scipy.optimize import linear_sum_assignment

from solar_inspect.detection.boxes import iou

MATCH_IOU = 0.5
MM_METRICS = ["mota", "idf1", "num_switches", "num_false_positives", "num_misses",
              "num_objects", "num_matches"]


class Frame(NamedTuple):
    gt_ids: np.ndarray       # (n,)
    gt_boxes: np.ndarray     # (n, 4) xyxy
    ign_boxes: np.ndarray    # (m, 4) xyxy
    hyp_ids: np.ndarray      # (k,)
    hyp_boxes: np.ndarray    # (k, 4) xyxy


def ignore_filter(gt_boxes, ign_boxes, hyp_boxes) -> np.ndarray:
    """(k,) bool: True for the hypotheses that are kept."""
    k = len(np.asarray(hyp_boxes).reshape(-1, 4))
    on_gt = (iou(hyp_boxes, gt_boxes) >= MATCH_IOU).any(axis=1) if len(gt_boxes) else np.zeros(k, bool)
    on_ign = (iou(hyp_boxes, ign_boxes) >= MATCH_IOU).any(axis=1) if len(ign_boxes) else np.zeros(k, bool)
    return on_gt | ~on_ign


def apply_ignore(f: Frame) -> tuple[Frame, int]:
    keep = ignore_filter(f.gt_boxes, f.ign_boxes, f.hyp_boxes)
    return f._replace(hyp_ids=f.hyp_ids[keep], hyp_boxes=f.hyp_boxes[keep]), int((~keep).sum())


def clear_mot_matches(frames: list[Frame]) -> list[list[tuple[int, int]]]:
    """Per frame, the (gt id, hyp id) pairs of CLEAR MOT matching: a ground-truth box
    keeps the hypothesis it was last matched to while their IoU is >= 0.5; the rest are
    paired by minimum total 1 - IoU among pairs with IoU >= 0.5."""
    last: dict[int, int] = {}
    out = []
    for f in frames:
        ok = iou(f.gt_boxes, f.hyp_boxes) >= MATCH_IOU
        pairs, g_used, h_used = [], set(), set()
        for i, g in enumerate(f.gt_ids.tolist()):
            if g in last:
                j = np.flatnonzero(f.hyp_ids == last[g])
                if len(j) and j[0] not in h_used and ok[i, j[0]]:
                    pairs.append((i, int(j[0])))
                    g_used.add(i)
                    h_used.add(int(j[0]))
        free = ok.copy()
        free[list(g_used), :] = False
        free[:, list(h_used)] = False
        if free.any():
            # Invalid pairs cost more than any full set of valid ones, so the assignment
            # first maximises the number of valid pairs, then minimises their cost.
            cost = np.where(free, 1.0 - iou(f.gt_boxes, f.hyp_boxes), 2.0 * min(ok.shape) + 1)
            rows, cols = linear_sum_assignment(cost)
            pairs += [(int(i), int(j)) for i, j in zip(rows, cols) if free[i, j]]
        matched = [(int(f.gt_ids[i]), int(f.hyp_ids[j])) for i, j in pairs]
        last.update(matched)
        out.append(matched)
    return out


def id_switches(matches: list[list[tuple[int, int]]]) -> int:
    """Each time a ground-truth id is matched to a track id other than its last one."""
    last: dict[int, int] = {}
    n = 0
    for frame in matches:
        for g, h in frame:
            n += g in last and last[g] != h
            last[g] = h
    return n


def count_error(frames: list[Frame]) -> dict:
    k = len(set(np.concatenate([f.hyp_ids for f in frames]).tolist())) if frames else 0
    g = len(set(np.concatenate([f.gt_ids for f in frames]).tolist())) if frames else 0
    return {"K": k, "G": g, "K_minus_G": k - g, "rel_count_error": (k - g) / g if g else float("nan")}


def _xywh(b: np.ndarray) -> np.ndarray:
    return np.c_[b[:, :2], b[:, 2:] - b[:, :2]]


def accumulate(frames: list[Frame]) -> mm.MOTAccumulator:
    acc = mm.MOTAccumulator(auto_id=True)
    for f in frames:
        if len(f.gt_ids) and len(f.hyp_ids):
            d = 1.0 - mm.distances.boxiou(_xywh(f.gt_boxes)[:, None], _xywh(f.hyp_boxes)[None, :])
            d = np.where(d > 1.0 - MATCH_IOU, np.nan, d)
        else:
            d = np.empty((len(f.gt_ids), len(f.hyp_ids)))
        acc.update(f.gt_ids, f.hyp_ids, d)
    return acc


def score(sequences: dict[str, list[Frame]]) -> dict[str, dict]:
    """Per sequence and "overall" (counts summed over sequences): MOTA, IDF1, IDSW
    (hand-written) and IDSW_mm (motmetrics), FP, FN, hypotheses dropped by the ignore
    rule, K, G, K - G, (K - G) / G."""
    names, accs, out = list(sequences), [], {}
    for name in names:
        kept = [apply_ignore(f) for f in sequences[name]]
        frames = [f for f, _ in kept]
        accs.append(accumulate(frames))
        out[name] = {"dropped": sum(n for _, n in kept),
                     "idsw": id_switches(clear_mot_matches(frames)), **count_error(frames)}
    summary = mm.metrics.create().compute_many(accs, metrics=MM_METRICS, names=names,
                                               generate_overall=True)
    total = {k: sum(out[n][k] for n in names) for k in ("dropped", "idsw", "K", "G")}
    total["K_minus_G"] = total["K"] - total["G"]
    total["rel_count_error"] = total["K_minus_G"] / total["G"] if total["G"] else float("nan")
    out["overall"] = total
    for name, row in summary.iterrows():
        r = out["overall" if name == "OVERALL" else name]
        r.update({"mota": float(row["mota"]), "idf1": float(row["idf1"]),
                  "idsw_mm": int(row["num_switches"]), "fp": int(row["num_false_positives"]),
                  "fn": int(row["num_misses"]), "gt_boxes": int(row["num_objects"]),
                  "matches": int(row["num_matches"])})
    return out
