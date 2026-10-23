"""Binary-mask metrics and the soft Dice loss, written from the definitions.

Every metric here is a function of three counts over the positive (panel) class:
tp, fp, fn. Masks are flattened before counting, so a stack of tiles is scored
over all of its pixels at once. That is the per-subset number: one IoU for every
pixel of a subset, not a mean of per-tile IoUs, which would let empty tiles (IoU
1.0 by convention when nothing is predicted) inflate the score.

    IoU       = tp / (tp + fp + fn)
    Dice      = 2 tp / (2 tp + fp + fn)          = 2 IoU / (1 + IoU)
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)

0 / 0 conventions, stated once:
- IoU and Dice with no positives in either mask are 1.0: nothing to find,
  nothing predicted, the mask is right.
- precision with no predicted positives, and recall with no true positives, are
  NaN. A null model has no precision; a number there would read as a measurement.
"""
from __future__ import annotations

import numpy as np
import torch


def counts(pred: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    """(tp, fp, fn) of two binary masks of the same shape, over all pixels."""
    p, t = np.asarray(pred, dtype=bool), np.asarray(target, dtype=bool)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {t.shape}")
    return int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())


def counts_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(tp, fp, fn) as an int64 tensor on the input's device. No host sync."""
    p, t = pred.bool(), target.bool()
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: {tuple(p.shape)} vs {tuple(t.shape)}")
    return torch.stack([(p & t).sum(), (p & ~t).sum(), (~p & t).sum()])


def scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    """IoU, Dice, precision and recall from counts, with the 0 / 0 rules above."""
    tp, fp, fn = int(tp), int(fp), int(fn)
    union = tp + fp + fn
    return {
        "iou": tp / union if union else 1.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if union else 1.0,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
    }


def iou(pred: np.ndarray, target: np.ndarray) -> float:
    return scores(*counts(pred, target))["iou"]


def dice(pred: np.ndarray, target: np.ndarray) -> float:
    return scores(*counts(pred, target))["dice"]


def iou_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tp, fp, fn = counts_torch(pred, target).double()
    union = tp + fp + fn
    return torch.where(union > 0, tp / union.clamp_min(1), torch.ones_like(union))


def dice_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tp, fp, fn = counts_torch(pred, target).double()
    denom = 2 * tp + fp + fn
    return torch.where(denom > 0, 2 * tp / denom.clamp_min(1), torch.ones_like(denom))


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - (2 sum(p t) + eps) / (sum p + sum t + eps), with p = sigmoid(logits).

    Summed over the whole batch, not per tile: a per-tile Dice on a tile with no
    panels is 0 / 0 in the limit and its gradient is unstable, and PV01 Brick
    tiles are 5% positive. Computed in float32 whatever autocast hands in. eps
    makes an empty prediction on an empty target a loss of 0 instead of NaN;
    under float16, sigmoid(-20) underflows to exactly 0, so that case is real.
    """
    p = torch.sigmoid(logits.float())
    t = target.float()
    inter = (p * t).sum()
    return 1.0 - (2.0 * inter + eps) / (p.sum() + t.sum() + eps)
