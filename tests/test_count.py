"""Per-image panel-count error, tested before it exists.

A detection counts when its score is >= the confidence threshold. Signed error is
predicted minus ground-truth count, so negative means panels missed. Worked by hand:

    image  scores               gt   conf 0.5: pred, signed, abs   conf 0.3: pred, signed, abs
    0      0.9, 0.8, 0.3, 0.1   3    2, -1, 1                      3,  0, 0
    1      0.6                  0    1, +1, 1                      1, +1, 1
    2      (none)               2    0, -2, 2                      0, -2, 2
    MAE                              4/3                           3/3 = 1

At 0.3 the score of exactly 0.3 counts, which is the >= boundary.
"""
from __future__ import annotations

import numpy as np
import pytest

from solar_inspect.detection.count import count_error

SCORES = [np.array([0.9, 0.8, 0.3, 0.1]), np.array([0.6]), np.zeros(0)]
GT = [3, 0, 2]


def test_hand_made_case_at_one_half():
    signed, absolute, mae = count_error(SCORES, GT, conf=0.5)
    assert signed.tolist() == [-1, 1, -2]
    assert absolute.tolist() == [1, 1, 2]
    assert mae == pytest.approx(4 / 3)


def test_threshold_is_inclusive():
    signed, _, mae = count_error(SCORES, GT, conf=0.3)
    assert signed.tolist() == [0, 1, -2]
    assert mae == pytest.approx(1.0)
