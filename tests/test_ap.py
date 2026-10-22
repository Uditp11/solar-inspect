"""AP@0.5, tested before it exists.

Rules: detections from every image are pooled and sorted by score globally; each
one, in that order, is matched to the highest-IoU still-unmatched ground truth in
its own image with IoU >= 0.5, else it is a false positive. So a second detection
on an already matched ground truth is a false positive. Precision is replaced by
its envelope (running max from the right), then integrated either at every recall
step ("all", the primary) or at the 101 recall points 0, 0.01, ..., 1 ("101", COCO).

The two-image toy, worked by hand. Ground truth: image A has gA1 = [0,0,10,10] and
gA2 = [20,0,30,10]; image B has gB1 = [0,0,10,10]. So n_gt = 3, and gA2 is never
found, which caps recall at 2/3.

    det  image  box             score  IoU             outcome
    d1   A      [0,0,10,10]     0.9    1 with gA1      TP
    d2   B      [50,50,60,60]   0.8    0               FP
    d3   B      [1,0,11,10]     0.7    90/110 with gB1 TP
    d4   A      [40,40,50,50]   0.6    0               FP

    cumulative TP 1, 1, 2, 2
    precision     1, 1/2, 2/3, 1/2
    recall        1/3, 1/3, 2/3, 2/3
    envelope      1, 2/3, 2/3, 1/2

    all-point: (1/3 - 0)(1) + (2/3 - 1/3)(2/3) = 1/3 + 2/9 = 5/9
    101-point: r = 0.00 .. 0.33 (34 points) -> first index with recall >= r is 0, envelope 1
               r = 0.34 .. 0.66 (33 points) -> index 2, envelope 2/3
               r = 0.67 .. 1.00 (34 points) -> recall never reaches r, 0
               (34 + 33 x 2/3) / 101 = 56/101
    5/9 = 0.5556 and 56/101 = 0.5545, so the two interpolations differ on the same toy.
"""
from __future__ import annotations

import numpy as np
import pytest

from solar_inspect.detection.ap import average_precision, match

A_GT = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=float)
B_GT = np.array([[0, 0, 10, 10]], dtype=float)
A_DET = np.array([[0, 0, 10, 10], [40, 40, 50, 50]], dtype=float)
A_SC = np.array([0.9, 0.6])
B_DET = np.array([[50, 50, 60, 60], [1, 0, 11, 10]], dtype=float)
B_SC = np.array([0.8, 0.7])

TOY = ([A_DET, B_DET], [A_SC, B_SC], [A_GT, B_GT])


def test_toy_tp_fp_sequence_is_global_by_score():
    scores, tp, n_gt = match(*TOY)
    assert scores.tolist() == [0.9, 0.8, 0.7, 0.6]
    assert tp.tolist() == [True, False, True, False]
    assert n_gt == 3


def test_toy_all_point_ap():
    assert average_precision(*TOY, method="all") == pytest.approx(5 / 9)


def test_toy_101_point_ap_differs_from_all_point():
    ap101 = average_precision(*TOY, method="101")
    assert ap101 == pytest.approx(56 / 101)
    assert ap101 != pytest.approx(5 / 9)


def test_duplicate_detection_is_a_false_positive():
    # one image, g1 = [0,0,10,10], g2 = [20,0,30,10]
    # a [0,0,10,10] 0.9 -> g1 TP; b [1,0,11,10] 0.8 -> g1 already taken, IoU 0 with g2, FP;
    # c [20,0,30,10] 0.7 -> g2 TP.
    # precision 1, 1/2, 2/3; recall 1/2, 1/2, 1; envelope 1, 2/3, 2/3
    # all-point: (1/2)(1) + (1/2)(2/3) = 5/6
    # 101-point: r = 0 .. 0.50 (51 points) at 1, r = 0.51 .. 1 (50 points) at 2/3
    #            (51 + 100/3) / 101 = 253/303
    gt = [np.array([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=float)]
    det = [np.array([[0, 0, 10, 10], [1, 0, 11, 10], [20, 0, 30, 10]], dtype=float)]
    sc = [np.array([0.9, 0.8, 0.7])]
    assert match(det, sc, gt)[1].tolist() == [True, False, True]
    assert average_precision(det, sc, gt, method="all") == pytest.approx(5 / 6)
    assert average_precision(det, sc, gt, method="101") == pytest.approx(253 / 303)


def test_match_takes_the_best_unmatched_ground_truth_not_the_best_overall():
    # g1 = [0,0,10,10], g2 = [2,0,12,10]
    # d1 [0,0,10,10] 0.9: IoU 1 with g1, 80/120 with g2 -> g1
    # d2 [0.5,0,10.5,10] 0.8: g1 95/105 = 0.905 (taken), g2 85/115 = 0.739 >= 0.5 -> g2, TP.
    # (VOC's rule, "best overall, FP if already taken", would call d2 a FP.)
    gt = [np.array([[0, 0, 10, 10], [2, 0, 12, 10]], dtype=float)]
    det = [np.array([[0, 0, 10, 10], [0.5, 0, 10.5, 10]], dtype=float)]
    assert match(det, [np.array([0.9, 0.8])], gt)[1].tolist() == [True, True]


def test_iou_exactly_one_half_is_a_match():
    # [0,0,10,10] vs [0,0,10,5]: inter 50, union 100 -> exactly 0.5, matched under >=
    gt = [np.array([[0, 0, 10, 10]], dtype=float)]
    det = [np.array([[0, 0, 10, 5]], dtype=float)]
    assert match(det, [np.array([0.9])], gt)[1].tolist() == [True]


def test_image_without_ground_truth_contributes_only_false_positives():
    # The toy plus image C, no ground truth, one detection at 0.95, now first by score.
    # sequence FP, TP, FP, TP, FP; cumulative TP 0, 1, 1, 2, 2
    # precision 0, 1/2, 1/3, 1/2, 2/5; recall 0, 1/3, 1/3, 2/3, 2/3
    # envelope 1/2, 1/2, 1/2, 1/2, 2/5
    # all-point: (1/3)(1/2) + (1/3)(1/2) = 1/3
    # 101-point: r = 0 -> index 0; r = 0.01 .. 0.33 -> index 1; r = 0.34 .. 0.66 -> index 3;
    #            the envelope is 1/2 at all three, so 67 points at 1/2, the other 34 at 0
    #            -> 33.5/101
    det, sc, gt = (list(x) for x in TOY)
    det.append(np.array([[100, 100, 110, 110]], dtype=float))
    sc.append(np.array([0.95]))
    gt.append(np.zeros((0, 4)))
    assert match(det, sc, gt)[1].tolist() == [False, True, False, True, False]
    assert average_precision(det, sc, gt, method="all") == pytest.approx(1 / 3)
    assert average_precision(det, sc, gt, method="101") == pytest.approx(33.5 / 101)


def test_no_detections_at_all_is_zero():
    empty = [np.zeros((0, 4)), np.zeros((0, 4))]
    none = [np.zeros(0), np.zeros(0)]
    for method in ("all", "101"):
        assert average_precision(empty, none, [A_GT, B_GT], method=method) == 0.0


def test_unknown_method_is_refused():
    with pytest.raises(ValueError):
        average_precision(*TOY, method="11")
