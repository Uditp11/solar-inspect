"""Dice, IoU, precision and recall on binary masks, tested before they exist.

Every expected value is worked by hand. Masks are flattened, so a stack of tiles
is scored over all of its pixels at once (one subset, one number):

    half overlap   p = [1,1,0,0], t = [1,0,0,0]   tp 1, fp 1, fn 0
                   Dice 2*1 / (2 + 1) = 2/3,  IoU 1 / (1 + 1 + 0) = 1/2
                   precision 1/2, recall 1/1
    identical      p = t = [1,0,1,0]              tp 2               -> 1, 1
    one empty      p = [0,0,0,0], t = [1,1,0,0]   tp 0, fn 2         -> 0, 0
    both empty     p = t = [0,0,0,0]              0 / 0, defined as 1.0: there was
                   nothing to find and nothing was predicted, so the mask is right.

Precision with no predicted positives, and recall with no true positives, are
0 / 0 and are returned as NaN rather than given a value: a null model has no
precision, and a number in that cell would read as a measurement.

Dice and IoU are two views of the same counts: Dice = 2 IoU / (1 + IoU).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from solar_inspect.segmentation.metrics import (
    counts, dice, dice_torch, iou, iou_torch, scores, soft_dice_loss,
)

P = np.array([[1, 1, 0, 0]], dtype=bool)
T = np.array([[1, 0, 0, 0]], dtype=bool)
EMPTY = np.zeros((1, 4), dtype=bool)


def test_half_overlap_dice_is_two_thirds():
    assert dice(P, T) == pytest.approx(2 / 3)


def test_half_overlap_iou_is_one_half():
    assert iou(P, T) == pytest.approx(1 / 2)


def test_identical_masks_score_one():
    m = np.array([[1, 0, 1, 0]], dtype=bool)
    assert dice(m, m) == 1.0 and iou(m, m) == 1.0


def test_one_empty_scores_zero():
    t = np.array([[1, 1, 0, 0]], dtype=bool)
    assert dice(EMPTY, t) == 0.0 and iou(EMPTY, t) == 0.0
    assert dice(t, EMPTY) == 0.0 and iou(t, EMPTY) == 0.0


def test_both_empty_is_defined_as_one():
    assert dice(EMPTY, EMPTY) == 1.0 and iou(EMPTY, EMPTY) == 1.0


def test_counts_and_positive_class_precision_recall():
    assert counts(P, T) == (1, 1, 0)
    s = scores(*counts(P, T))
    assert s["precision"] == pytest.approx(1 / 2)
    assert s["recall"] == pytest.approx(1.0)
    assert s["iou"] == pytest.approx(1 / 2) and s["dice"] == pytest.approx(2 / 3)


def test_precision_with_nothing_predicted_is_nan_not_a_number():
    s = scores(*counts(EMPTY, T))
    assert math.isnan(s["precision"]) and s["recall"] == 0.0 and s["iou"] == 0.0


def test_recall_with_nothing_to_find_is_nan():
    s = scores(*counts(P, EMPTY))
    assert math.isnan(s["recall"]) and s["precision"] == 0.0


def test_stack_is_scored_over_all_pixels_not_averaged_per_tile():
    # Tile 1: tp 1, fp 1. Tile 2: both empty (per-tile IoU 1.0).
    # Pooled over pixels: tp 1, fp 1, fn 0 -> IoU 1/2. A mean of per-tile IoUs
    # would give (1/2 + 1) / 2 = 3/4: empty tiles would inflate the score.
    p = np.concatenate([P, EMPTY])
    t = np.concatenate([T, EMPTY])
    assert iou(p, t) == pytest.approx(1 / 2)


def test_torch_matches_numpy_on_the_hand_cases():
    for p, t in [(P, T), (EMPTY, T), (EMPTY, EMPTY), (T, T)]:
        tp_, tt_ = torch.from_numpy(p), torch.from_numpy(t)
        assert float(iou_torch(tp_, tt_)) == pytest.approx(iou(p, t))
        assert float(dice_torch(tp_, tt_)) == pytest.approx(dice(p, t))


def test_dice_iou_identity_on_random_masks():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.random((3, 16, 16)) < rng.random()
        t = rng.random((3, 16, 16)) < rng.random()
        j = iou(p, t)
        assert dice(p, t) == pytest.approx(2 * j / (1 + j), abs=1e-12)


# --- soft Dice loss --------------------------------------------------------

@pytest.mark.parametrize("logit", [-20.0, -200.0])
def test_soft_dice_loss_is_zero_and_finite_on_empty_pred_and_target_in_float16(logit):
    # Float16 logits, as autocast hands them over. sigmoid(-200) is 0 exactly
    # even in float32 (below its smallest subnormal), so sum p + sum t is 0.
    logits = torch.full((2, 1, 8, 8), logit, dtype=torch.float16)
    target = torch.zeros((2, 1, 8, 8), dtype=torch.float16)
    loss = soft_dice_loss(logits, target)
    assert torch.isfinite(loss)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_soft_dice_loss_without_epsilon_fails_silently_on_empty_tiles():
    target = torch.zeros((2, 1, 8, 8), dtype=torch.float16)
    # Prediction exactly 0: 0 / 0, NaN, no exception.
    assert torch.isnan(soft_dice_loss(torch.full_like(target, -200.0), target, eps=0.0))
    # Prediction 2e-9 per pixel: 0 / tiny = 0, so a correct empty mask scores a
    # loss of 1, the same as a completely wrong one.
    assert float(soft_dice_loss(torch.full_like(target, -20.0), target, eps=0.0)) == 1.0


def test_soft_dice_loss_on_perfect_and_opposite_predictions():
    target = torch.zeros((1, 1, 4, 4))
    target[..., :2, :] = 1.0                                  # 8 positives
    perfect = (target * 2 - 1) * 30.0                         # sigmoid -> {0, 1}
    assert float(soft_dice_loss(perfect, target)) == pytest.approx(0.0, abs=1e-6)
    # Opposite: overlap 0, sum p 8, sum t 8 -> 1 - eps / (16 + eps) with eps = 1
    assert float(soft_dice_loss(-perfect, target)) == pytest.approx(1 - 1 / 17, abs=1e-6)
