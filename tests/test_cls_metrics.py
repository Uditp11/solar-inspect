"""Hand-computed tests for the classification maths written by hand.

Every expected value below was worked out on paper first. That is the point of
the file: a test whose expectation came from running the code proves the code is
deterministic, not that it is right.

Covered: the confusion matrix and everything derived from it, the class weights,
focal loss, the KD loss, and ECE. The KD tests are the ones that earn their keep
-- two of the three ways that loss goes wrong fail by making distillation look
useless rather than by raising, so they are invisible without a test that pins
the scale.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_inspect.classification.calibration import ece, nll, softmax
from solar_inspect.classification.train import (class_weights, confusion,
                                                focal_loss, kd_loss, metrics)


# --------------------------------------------------------------------------
# confusion matrix and the metrics derived from it
# --------------------------------------------------------------------------

def test_confusion_is_true_by_predicted():
    # 3 classes. true=0 predicted 0,0,1 | true=1 predicted 1 | true=2 predicted 0.
    true = np.array([0, 0, 0, 1, 2])
    pred = np.array([0, 0, 1, 1, 0])
    cm = confusion(true, pred, 3)
    assert cm.tolist() == [[2, 1, 0],
                           [0, 1, 0],
                           [1, 0, 0]]
    assert cm.sum() == len(true)


def test_metrics_on_a_hand_worked_matrix():
    #        pred0 pred1 pred2
    # true0    2     1     0     support 3
    # true1    0     1     0     support 1
    # true2    1     0     0     support 1
    cm = np.array([[2, 1, 0], [0, 1, 0], [1, 0, 0]])
    m = metrics(cm)

    # recall = diag / row sum
    assert m["recall"] == pytest.approx([2 / 3, 1.0, 0.0])
    # precision = diag / column sum; columns are 3, 2, 0
    assert m["precision"] == pytest.approx([2 / 3, 1 / 2, 0.0])
    # F1 per class: 2PR/(P+R) = 2/3, 2/3, 0
    assert m["f1"] == pytest.approx([2 / 3, 2 / 3, 0.0])
    assert m["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0.0) / 3)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["support"].tolist() == [3, 1, 1]


def test_empty_class_scores_zero_rather_than_nan():
    """A class with no support and no predictions must not produce nan.

    macro-F1 averages over classes, so one nan makes the whole run's headline
    number nan -- and at 57:1 a rare class missing from a small split is not
    hypothetical.
    """
    cm = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0]])
    m = metrics(cm)
    assert np.isfinite(m["f1"]).all()
    assert m["f1"][2] == 0.0
    assert m["macro_f1"] == pytest.approx(2 / 3)


def test_macro_f1_agrees_with_sklearn():
    rng = np.random.default_rng(0)
    true, pred = rng.integers(0, 5, 400), rng.integers(0, 5, 400)
    from sklearn.metrics import f1_score
    mine = metrics(confusion(true, pred, 5))["macro_f1"]
    assert mine == pytest.approx(
        f1_score(true, pred, average="macro", zero_division=0), abs=1e-12)


# --------------------------------------------------------------------------
# class weights
# --------------------------------------------------------------------------

def test_class_weights_are_inverse_frequency_averaging_to_one():
    # 6 labels over 3 classes: counts 3, 2, 1. w_c = n / (k * n_c) = 6/(3*n_c).
    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    w = class_weights(labels, 3)
    assert w.tolist() == pytest.approx([2 / 3, 1.0, 2.0])
    # The average weight *per example* is 1, which is what keeps the weighted
    # loss on the same scale as unweighted CE.
    assert float((w[labels]).mean()) == pytest.approx(1.0)


def test_absent_class_does_not_divide_by_zero():
    w = class_weights(torch.tensor([0, 0, 1, 1]), 3)
    assert torch.isfinite(w).all()


# --------------------------------------------------------------------------
# focal loss
# --------------------------------------------------------------------------

def test_focal_at_gamma_zero_is_cross_entropy():
    torch.manual_seed(0)
    logits, y = torch.randn(8, 4), torch.randint(0, 4, (8,))
    assert float(focal_loss(logits, y, 0.0)) == pytest.approx(
        float(torch.nn.functional.cross_entropy(logits, y)), abs=1e-6)


def test_focal_downweights_the_easy_example_by_one_minus_p_squared():
    # Two examples, one easy and one hard, hand-computed at gamma=2.
    logits = torch.tensor([[10.0, 0.0], [0.0, 0.0]])
    y = torch.tensor([0, 0])
    p = torch.softmax(logits, 1)[torch.arange(2), y]        # ~1.0 and exactly 0.5
    expected = float((-((1 - p) ** 2) * torch.log(p)).mean())
    assert float(focal_loss(logits, y, 2.0)) == pytest.approx(expected, abs=1e-7)
    # The easy example's contribution is ~0; the hard one's is 0.25 * ln 2, and
    # the mean over two examples is half of that.
    assert float(focal_loss(logits, y, 2.0)) == pytest.approx(
        0.5 * 0.25 * np.log(2.0), abs=1e-4)


# --------------------------------------------------------------------------
# KD loss -- the three silent failure modes
# --------------------------------------------------------------------------

def test_kd_at_alpha_one_is_exactly_cross_entropy():
    torch.manual_seed(0)
    s, t, y = torch.randn(6, 5), torch.randn(6, 5), torch.randint(0, 5, (6,))
    assert float(kd_loss(s, t, y, alpha=1.0, T=4.0)) == pytest.approx(
        float(torch.nn.functional.cross_entropy(s, y)), abs=1e-6)


def test_kd_soft_term_is_zero_when_the_student_already_matches_the_teacher():
    torch.manual_seed(0)
    t, y = torch.randn(6, 5), torch.randint(0, 5, (6,))
    both = kd_loss(t, t, y, alpha=0.0, T=3.0)
    assert float(both) == pytest.approx(0.0, abs=1e-6)


def test_batchmean_is_num_classes_times_mean():
    """The bug that eats afternoons, pinned to a number.

    reduction='mean' divides by every element of the (batch, classes) tensor
    instead of by the batch, so at 12 classes it scales the KD term by exactly
    1/12. Nothing raises; distillation simply stops working.
    """
    import torch.nn.functional as F
    torch.manual_seed(0)
    s, t, T, k = torch.randn(6, 12), torch.randn(6, 12), 4.0, 12
    ls, pt = F.log_softmax(s / T, 1), F.softmax(t / T, 1)
    batchmean = F.kl_div(ls, pt, reduction="batchmean")
    wrong = F.kl_div(ls, pt, reduction="mean")
    assert float(batchmean) == pytest.approx(float(wrong) * k, abs=1e-6)
    # And kd_loss uses the right one.
    y = torch.randint(0, k, (6,))
    soft_part = float(kd_loss(s, t, y, alpha=0.0, T=T))
    assert soft_part == pytest.approx((T ** 2) * float(batchmean), abs=1e-5)


def test_t_squared_keeps_the_soft_term_commensurate_across_temperatures():
    """Soft-target gradients scale as 1/T^2, so T**2 is what cancels it.

    Without the factor the KD term shrinks as T rises and alpha stops meaning the
    same thing at T=1 and T=4. Checked in the limit: for small logit differences
    the T^2-scaled KL tends to a constant in T, while the unscaled one falls as
    1/T^2.
    """
    import torch.nn.functional as F
    s, t = torch.randn(64, 12) * 0.01, torch.randn(64, 12) * 0.01
    y = torch.randint(0, 12, (64,))
    scaled = [float(kd_loss(s, t, y, alpha=0.0, T=T)) for T in (2.0, 8.0)]
    raw = [float(F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1),
                          reduction="batchmean")) for T in (2.0, 8.0)]
    assert scaled[1] == pytest.approx(scaled[0], rel=0.05)      # flat in T
    assert raw[1] == pytest.approx(raw[0] / 16.0, rel=0.05)     # falls as 1/T^2


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def test_softmax_is_temperature_monotone_and_normalised():
    z = np.array([[2.0, 1.0, 0.0]])
    assert softmax(z).sum() == pytest.approx(1.0)
    # Higher T flattens: the top probability falls, the bottom rises.
    assert softmax(z, 4.0)[0, 0] < softmax(z, 1.0)[0, 0]
    assert softmax(z, 4.0)[0, 2] > softmax(z, 1.0)[0, 2]
    # T -> large tends to uniform.
    assert softmax(z, 1e6)[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-4)


def test_nll_of_a_confident_correct_prediction_is_near_zero():
    z = np.array([[20.0, 0.0]])
    assert nll(z, np.array([0]), 1.0) == pytest.approx(0.0, abs=1e-6)
    assert nll(z, np.array([1]), 1.0) > 15.0


def test_ece_is_zero_for_a_perfectly_calibrated_model():
    # Confidence 1.0 and always right; confidence 0.5 (bin centre) right half the
    # time. Each bin's |accuracy - mean confidence| is 0.
    conf = np.array([1.0, 1.0, 0.5, 0.5])
    correct = np.array([1.0, 1.0, 1.0, 0.0])
    assert ece(conf, correct, bins=2) == pytest.approx(0.0)


def test_ece_hand_computed_on_two_bins():
    # bins=2 -> edges 0, 0.5, 1. digitize(right=True) on the single interior edge
    # 0.5 puts <= 0.5 in bin 0 and > 0.5 in bin 1.
    # bin 0: conf 0.4, 0.4 (mean 0.4), accuracy 0.5  -> |0.5-0.4| = 0.1, share 0.5
    # bin 1: conf 0.9, 0.9 (mean 0.9), accuracy 1.0  -> |1.0-0.9| = 0.1, share 0.5
    conf = np.array([0.4, 0.4, 0.9, 0.9])
    correct = np.array([1.0, 0.0, 1.0, 1.0])
    assert ece(conf, correct, bins=2) == pytest.approx(0.5 * 0.1 + 0.5 * 0.1)


def test_ece_of_an_overconfident_model_is_the_gap():
    # One bin, confidence 0.9 throughout, right 60% of the time.
    conf = np.full(10, 0.9)
    correct = np.array([1.0] * 6 + [0.0] * 4)
    assert ece(conf, correct, bins=1) == pytest.approx(0.3)
