"""Per-tile scoring of PV01, the paired tile bootstrap and the boundary-error mask.

The per-subset number is unchanged from metrics.py: tp, fp and fn summed over
every pixel of the subset. Keeping them per tile is what lets a bootstrap
resample tiles and sum again; on the original sample that sum is the subset's
counts, so the bootstrap's point value is the reported number exactly.

**Paired bootstrap.** Each draw resamples the tiles of one subset with
replacement, and the same indices are applied to both models' per-tile counts,
so the interval of the difference reflects only how the two models disagree on
the same tiles, not how hard each draw's tiles happen to be. The interval is the
2.5th to 97.5th percentile of the draws. Tiles of one scene are correlated, so
treating tiles as the independent unit makes these intervals narrower than the
truth.

**Near a boundary.** A pixel is within r px of the true mask boundary when the
Euclidean distance between its centre and the nearest pixel of the other class
is at most r (scipy's exact distance transform). The tile edge is not a
boundary, so a tile of one class has none.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

METRICS = ("iou", "dice", "precision", "recall")


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor, batch: int, amp: bool) -> torch.Tensor:
    """Standardised (N, 3, H, W) -> bool (N, 1, H, W): sigmoid > 0.5, i.e. logit > 0.

    Batched and autocast as train.evaluate is, so val reproduces training's numbers.
    """
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=amp):
            out.append(model(x[i:i + batch]) > 0)
    return torch.cat(out)


def colour_rule(rgb: torch.Tensor, feature: str, direction: str, threshold: int) -> torch.Tensor:
    """uint8 (N, 3, H, W) -> bool (N, 1, H, W), the rule scripts/baseline_seg.py chose."""
    c = rgb.int()
    f = 2 * c[:, 2] - c[:, 0] - c[:, 1] if feature == "2B-R-G" else c[:, "RGB".index(feature)]
    return (f >= threshold if direction == ">=" else f <= threshold).unsqueeze(1)


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    """uint8 (N, 3, H, W) -> uint8 (N, 3, H, W): BT.601 luma, rounded, in all three channels.

    Y = 0.299 R + 0.587 G + 0.114 B, computed in float64 and rounded to the nearest
    integer (ties to even). The colour control for the thermal check: colour is
    removed, everything else about the tile is kept.
    """
    w = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float64, device=rgb.device)
    y = torch.einsum("nchw,c->nhw", rgb.double(), w).round().clamp(0, 255).to(torch.uint8)
    return y[:, None].expand(-1, 3, -1, -1).contiguous()


def tile_counts(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """(N, ...) masks -> int64 (N, 3): tp, fp, fn of each tile."""
    p, t = pred.bool().flatten(1), target.bool().flatten(1)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    return torch.stack([(p & t).sum(1), (p & ~t).sum(1), (~p & t).sum(1)], 1).cpu().numpy()


def scores_np(c: np.ndarray) -> dict[str, np.ndarray]:
    """metrics.scores over the last axis of (..., 3) counts, with the same 0 / 0 rules."""
    tp, fp, fn = (np.asarray(c)[..., k].astype(np.float64) for k in range(3))
    union = tp + fp + fn
    return {
        "iou": np.where(union > 0, tp / np.maximum(union, 1), 1.0),
        "dice": np.where(union > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 1.0),
        "precision": np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan),
        "recall": np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), np.nan),
    }


def paired_bootstrap(a: np.ndarray, b: np.ndarray, draws: int, seed: int) -> dict:
    """Point values and percentile 95% intervals of a, b and a - b, per metric.

    a and b are (n_tiles, 3) per-tile counts of two models on the same tiles.
    `undefined_draws` counts draws where the metric is 0 / 0 (NaN); the interval
    is taken over the others.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"a and b cover different tiles: {a.shape} vs {b.shape}")
    idx = np.random.default_rng(seed).integers(0, len(a), (draws, len(a)))
    pa, pb = scores_np(a.sum(0)), scores_np(b.sum(0))
    da, db = scores_np(a[idx].sum(1)), scores_np(b[idx].sum(1))

    def summary(point: float, d: np.ndarray) -> dict:
        ok = ~np.isnan(d)
        return {"point": float(point),
                "ci95": np.percentile(d[ok], [2.5, 97.5]).tolist() if ok.any() else [None, None],
                "undefined_draws": int((~ok).sum())}

    return {
        "a": {m: summary(pa[m], da[m]) for m in METRICS},
        "b": {m: summary(pb[m], db[m]) for m in METRICS},
        "a_minus_b": {m: summary(pa[m] - pb[m], da[m] - db[m]) for m in METRICS},
        "draws": draws, "seed": seed, "tiles": len(a),
    }


def near_boundary(mask: np.ndarray, r: float) -> np.ndarray:
    """bool (H, W): pixels within r px of a pixel of the other class."""
    m = np.asarray(mask, dtype=bool)
    if m.all() or not m.any():
        return np.zeros_like(m)
    return np.where(m, distance_transform_edt(m), distance_transform_edt(~m)) <= r
