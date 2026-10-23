"""Per-tile counts, the paired tile bootstrap and the near-boundary mask, tested first.

Hand-worked cases:

    tile counts    two 2x2 tiles, p = [[1,1],[0,0]] and [[0,0],[0,0]],
                   t = [[1,0],[0,0]] and [[0,0],[1,0]]
                   tile 0: tp 1, fp 1, fn 0      tile 1: tp 0, fp 0, fn 1
                   summed: tp 1, fp 1, fn 1, IoU 1/3, the subset number

    paired         a = tiles (tp, fp, fn) (1,0,0) and (0,0,1)
    bootstrap      b = tiles            (1,1,0) and (0,1,1)
                   A resample of two tiles is {0,0}, {1,1} or one of each:
                     {0,0}: a 2/2 = 1,   b 2/4 = 1/2,  a - b = 1/2
                     {1,1}: a 0/2 = 0,   b 0/4 = 0,    a - b = 0
                     mixed: a 1/2,       b 1/4,        a - b = 1/4
                   Paired, a - b is never below 0. Unpaired (a and b drawn
                   separately) it could be 0 - 1/2 = -1/2.

    near boundary  a 1 x 10 strip, columns 0..4 panel, 5..9 not. Euclidean
                   distance from each pixel to the nearest pixel of the other
                   class: 5 4 3 2 1 | 1 2 3 4 5. Within 1 px: columns 4, 5.
                   Within 3 px: columns 2..7.

    luminance      Y = 0.299 R + 0.587 G + 0.114 B, rounded:
                   (255, 0, 0) -> 76.245 -> 76     (0, 255, 0) -> 149.685 -> 150
                   (0, 0, 255) -> 29.07 -> 29      (255, 255, 255) -> 255
                   (10, 20, 30) -> 2.99 + 11.74 + 3.42 = 18.15 -> 18
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from solar_inspect.segmentation.evaluate import (
    luminance, near_boundary, paired_bootstrap, scores_np, tile_counts,
)
from solar_inspect.segmentation.metrics import scores


def test_tile_counts_by_hand():
    p = torch.tensor([[[[1, 1], [0, 0]]], [[[0, 0], [0, 0]]]], dtype=torch.bool)
    t = torch.tensor([[[[1, 0], [0, 0]]], [[[0, 0], [1, 0]]]], dtype=torch.bool)
    c = tile_counts(p, t)
    assert c.tolist() == [[1, 1, 0], [0, 0, 1]]
    assert scores(*c.sum(0))["iou"] == pytest.approx(1 / 3)


def test_scores_np_equals_scores_including_the_zero_over_zero_rules():
    rng = np.random.default_rng(0)
    c = rng.integers(0, 4, (500, 3))        # small counts, so 0 / 0 cases occur
    got = scores_np(c)
    for i, row in enumerate(c):
        ref = scores(*row)
        for k, v in ref.items():
            if np.isnan(v):
                assert np.isnan(got[k][i]), (row, k)
            else:
                assert got[k][i] == pytest.approx(v, abs=1e-15), (row, k)


def test_bootstrap_of_one_tile_collapses_to_the_point_value():
    a = np.array([[3, 1, 2]])
    r = paired_bootstrap(a, a, draws=200, seed=0)
    assert r["a"]["iou"]["point"] == pytest.approx(0.5)
    assert r["a"]["iou"]["ci95"] == pytest.approx([0.5, 0.5])
    assert r["a_minus_b"]["iou"]["ci95"] == [0.0, 0.0]


def test_bootstrap_is_paired_by_hand():
    a = np.array([[1, 0, 0], [0, 0, 1]])
    b = np.array([[1, 1, 0], [0, 1, 1]])
    r = paired_bootstrap(a, b, draws=10_000, seed=0)
    assert r["a"]["iou"]["point"] == pytest.approx(0.5)
    assert r["b"]["iou"]["point"] == pytest.approx(0.25)
    assert r["a_minus_b"]["iou"]["point"] == pytest.approx(0.25)
    # each of {0,0} and {1,1} has probability 1/4, so both tails are reached
    assert r["a"]["iou"]["ci95"] == pytest.approx([0.0, 1.0])
    assert r["a_minus_b"]["iou"]["ci95"] == pytest.approx([0.0, 0.5])


def test_bootstrap_is_reproducible_and_counts_undefined_draws():
    # tile 1 has nothing predicted: a draw of only tile 1 has no precision
    a = np.array([[2, 1, 0], [0, 0, 3]])
    r1 = paired_bootstrap(a, a, draws=1000, seed=7)
    r2 = paired_bootstrap(a, a, draws=1000, seed=7)
    assert r1 == r2
    assert 150 < r1["a"]["precision"]["undefined_draws"] < 350     # about 1/4 of 1000
    assert r1["a"]["iou"]["undefined_draws"] == 0


def test_near_boundary_by_hand():
    m = np.zeros((1, 10), bool)
    m[0, :5] = True
    assert np.flatnonzero(near_boundary(m, 1)[0]).tolist() == [4, 5]
    assert np.flatnonzero(near_boundary(m, 3)[0]).tolist() == [2, 3, 4, 5, 6, 7]


@pytest.mark.parametrize("fill", [False, True])
def test_a_tile_of_one_class_has_no_boundary(fill):
    # The tile edge is not a mask boundary.
    assert not near_boundary(np.full((8, 8), fill), 3).any()


def test_luminance_by_hand_replicated_to_three_channels():
    px = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (10, 20, 30)]
    rgb = torch.tensor(px, dtype=torch.uint8).T.reshape(1, 3, 1, 5)
    y = luminance(rgb)
    assert y.dtype == torch.uint8 and y.shape == (1, 3, 1, 5)
    assert y[0, :, 0].tolist() == [[76, 150, 29, 255, 18]] * 3
