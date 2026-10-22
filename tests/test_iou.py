"""Pairwise IoU for xyxy boxes, tested before it exists.

Continuous coordinates throughout: a box [x1, y1, x2, y2] has area (x2 - x1)(y2 - y1),
with no "+1" pixel convention. Every expected value is worked by hand:

    identical   [0,0,2,2] vs [0,0,2,2]   inter 4, union 4 + 4 - 4 = 4      -> 1
    disjoint    [0,0,1,1] vs [2,2,3,3]   inter 0                           -> 0
    shared edge [0,0,1,1] vs [1,0,2,1]   inter width min(1,2) - max(0,1) = 0 -> 0
    partial     [0,0,2,2] vs [1,0,3,2]   inter 1 x 2 = 2, union 4 + 4 - 2 = 6 -> 2/6
    contained   [0,0,4,4] vs [1,1,3,3]   inter 4, union 16 + 4 - 4 = 16    -> 1/4
    zero area   [1,1,1,1] vs [0,0,2,2]   inter 0, union 0 + 4 - 0 = 4      -> 0
    both zero   [1,1,1,1] vs [1,1,1,1]   inter 0, union 0, defined as 0    -> 0

The shared-edge case is the one the NMS argument for D2 rests on: panels tiled in a
row share an edge, so their axis-aligned boxes have IoU 0 and NMS cannot merge them.
"""
from __future__ import annotations

import numpy as np
import pytest

from solar_inspect.detection.boxes import iou


def one(a, b) -> float:
    m = iou(np.array([a], dtype=float), np.array([b], dtype=float))
    assert m.shape == (1, 1)
    return float(m[0, 0])


def test_identical_boxes_are_one():
    assert one([0, 0, 2, 2], [0, 0, 2, 2]) == pytest.approx(1.0)


def test_disjoint_boxes_are_zero():
    assert one([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0


def test_boxes_sharing_an_edge_are_zero():
    assert one([0, 0, 1, 1], [1, 0, 2, 1]) == 0.0
    assert one([0, 0, 1, 1], [0, 1, 1, 2]) == 0.0          # stacked vertically
    assert one([0, 0, 1, 1], [1, 1, 2, 2]) == 0.0          # corner only


def test_hand_computed_partial_overlap():
    assert one([0, 0, 2, 2], [1, 0, 3, 2]) == pytest.approx(2 / 6)


def test_full_containment_is_the_area_ratio():
    assert one([0, 0, 4, 4], [1, 1, 3, 3]) == pytest.approx(1 / 4)
    assert one([1, 1, 3, 3], [0, 0, 4, 4]) == pytest.approx(1 / 4)


def test_zero_area_boxes_do_not_divide_by_zero():
    with np.errstate(all="raise"):
        assert one([1, 1, 1, 1], [0, 0, 2, 2]) == 0.0
        assert one([1, 1, 1, 1], [1, 1, 1, 1]) == 0.0


def test_matrix_shape_and_entries():
    a = np.array([[0, 0, 2, 2], [0, 0, 4, 4]], dtype=float)
    b = np.array([[0, 0, 2, 2], [1, 0, 3, 2], [1, 1, 3, 3]], dtype=float)
    # rows of a against columns of b, each entry from the table in the docstring
    # except a[1] vs b[0]: [0,0,4,4] contains [0,0,2,2], inter 4, union 16 -> 1/4,
    # and a[1] vs b[1]: [0,0,4,4] contains [1,0,3,2], inter 4, union 16 -> 1/4.
    expected = np.array([[1, 2 / 6, 1 / 7],
                         [1 / 4, 1 / 4, 1 / 4]])
    # a[0] vs b[2]: [0,0,2,2] vs [1,1,3,3], inter 1 x 1 = 1, union 4 + 4 - 1 = 7 -> 1/7
    np.testing.assert_allclose(iou(a, b), expected)


def test_empty_inputs_give_the_right_empty_shape():
    some = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [1, 1, 2, 2]], dtype=float)
    none = np.zeros((0, 4))
    assert iou(none, some).shape == (0, 3)
    assert iou(some[:2], none).shape == (2, 0)
    assert iou(none, none).shape == (0, 0)
