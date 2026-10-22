"""Greedy NMS, tested before it exists.

Convention: a box is suppressed when its IoU with an already kept box is strictly
greater than the threshold (torchvision's). Kept indices come back in descending
score order; equal scores keep the lower index first.

The five-box case, threshold 0.5, worked by hand:

    i  box            score
    0  [0,0,10,10]    0.90
    1  [1,0,11,10]    0.80   vs 0: inter 9 x 10 = 90, union 100 + 100 - 90 = 110 -> 0.818
    2  [20,0,30,10]   0.70   vs 3: inter 5 x 10 = 50, union 100 + 100 - 50 = 150 -> 0.333
    3  [25,0,35,10]   0.85   vs 0: disjoint -> 0
    4  [0,0,10,6]     0.60   vs 0: inter 10 x 6 = 60, union 100 + 60 - 60 = 100 -> 0.6

    score order 0, 3, 1, 2, 4
    0 kept; 3 kept (IoU 0 with 0); 1 suppressed (0.818 > 0.5 with 0);
    2 kept (0 with 0, 0.333 with 3); 4 suppressed (0.6 > 0.5 with 0)  -> [0, 3, 2]
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision

from solar_inspect.detection.nms import nms

FIVE = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [20, 0, 30, 10],
                 [25, 0, 35, 10], [0, 0, 10, 6]], dtype=float)
FIVE_SCORES = np.array([0.90, 0.80, 0.70, 0.85, 0.60])


def test_hand_built_five_box_case():
    assert nms(FIVE, FIVE_SCORES, 0.5).tolist() == [0, 3, 2]


def test_iou_exactly_at_the_threshold_is_kept():
    # [0,0,10,10] vs [0,0,10,5]: inter 50, union 100 -> exactly 0.5, not > 0.5
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 5]], dtype=float)
    assert nms(boxes, np.array([0.9, 0.8]), 0.5).tolist() == [0, 1]


def test_a_suppressed_box_suppresses_nothing():
    # threshold 0.3: a-b and b-c are both 50 / 150 = 1/3 > 0.3, a-c is 0.
    # b is suppressed by a, so c survives even though it overlaps b.
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [10, 0, 20, 10]], dtype=float)
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.3).tolist() == [0, 2]


def test_equal_scores_keep_the_lower_index():
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10]], dtype=float)
    assert nms(boxes, np.array([0.5, 0.5]), 0.5).tolist() == [0]
    assert nms(boxes[::-1].copy(), np.array([0.5, 0.5]), 0.5).tolist() == [0]


def test_empty_input():
    out = nms(np.zeros((0, 4)), np.zeros(0), 0.5)
    assert out.shape == (0,) and out.dtype.kind == "i"


def test_matches_torchvision_on_seeded_random_boxes():
    # torchvision is the cross-check here, not the implementation.
    rng = np.random.default_rng(0)
    for thr in (0.3, 0.5, 0.7):
        xy = rng.uniform(0, 600, size=(400, 2))
        wh = rng.uniform(10, 80, size=(400, 2))
        boxes = np.column_stack([xy, xy + wh])
        scores = rng.uniform(size=400)
        ref = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), thr)
        assert nms(boxes, scores, thr).tolist() == ref.tolist()
