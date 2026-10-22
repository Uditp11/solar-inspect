"""False-positive classes of the post-hoc detection error analysis, on hand-made frames.

Rule, in this order, for a detection that `ap.match` called a false positive:
duplicate (IoU > 0.3 with a true positive of the same frame), localisation (best IoU
with any ground truth in [0.1, 0.5)), background (best IoU < 0.1). Boxes are 10x10
unless stated; IoUs below are worked by hand.
"""
from __future__ import annotations

import numpy as np

from solar_inspect.detection.errors import classify


def test_one_of_each_class():
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=float)
    det = np.array([
        [0, 0, 10, 10],    # TP on gt 0
        [1, 0, 11, 10],    # IoU with det 0 = 90 / 110 = 0.818 > 0.3   -> duplicate
        [20, 0, 24, 10],   # best gt IoU = 40 / 100 = 0.4 in [0.1, 0.5) -> localisation
        [60, 0, 70, 10],   # no overlap with anything                   -> background
    ], dtype=float)
    tp = np.array([True, False, False, False])
    assert classify(det, tp, gt).tolist() == ["tp", "duplicate", "localisation", "background"]


def test_duplicate_is_tested_before_localisation():
    # A box spanning two found panels: IoU with each gt and each TP is 50 / 150 = 0.333,
    # so it is in the localisation band too; the rule order makes it a duplicate.
    gt = np.array([[0, 0, 10, 10], [10, 0, 20, 10]], dtype=float)
    det = np.array([[0, 0, 10, 10], [10, 0, 20, 10], [5, 0, 15, 10]], dtype=float)
    tp = np.array([True, True, False])
    assert classify(det, tp, gt)[2] == "duplicate"


def test_band_edges():
    gt = np.array([[0, 0, 10, 10]], dtype=float)
    det = np.array([
        [0, 0, 1, 10],     # IoU 10 / 100 = 0.1 exactly: localisation (band is closed at 0.1)
        [0, 0, 0.9, 10],   # IoU 0.09: background
        [40, 0, 50, 10],   # TP somewhere else (a gt it matched is not needed for the rule)
        [43, 0, 53, 10],   # IoU with that TP = 70 / 130 = 0.538 -> duplicate
        [47, 0, 57, 10],   # IoU with the TP = 30 / 170 = 0.176, not > 0.3, gt IoU 0 -> background
    ], dtype=float)
    tp = np.array([False, False, True, False, False])
    assert classify(det, tp, gt).tolist() == [
        "localisation", "background", "tp", "duplicate", "background"]


def test_false_positive_on_a_taken_ground_truth_with_no_true_positive_overlap():
    # Both halves of the gt have IoU 0.5 with it and IoU 0 with each other. The higher
    # scorer took the gt; the other is a false positive the three classes do not cover
    # (best gt IoU >= 0.5, no TP overlap > 0.3). It is reported, not forced into a class.
    gt = np.array([[0, 0, 10, 10]], dtype=float)
    det = np.array([[0, 0, 10, 5], [0, 5, 10, 10]], dtype=float)
    tp = np.array([True, False])
    assert classify(det, tp, gt).tolist() == ["tp", "unclassified"]


def test_frame_without_ground_truth_is_all_background():
    det = np.array([[0, 0, 10, 10]], dtype=float)
    assert classify(det, np.array([False]), np.zeros((0, 4))).tolist() == ["background"]
