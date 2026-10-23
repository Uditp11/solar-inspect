"""PV01 tiles and masks.

Mask values in PV01 are exactly {0, 211, 212, 213} (docs/DATA.md); the three
non-zero codes are panel classes the published code list distinguishes and this
module does not. A value outside that set is refused, not counted as panel.

All 645 tiles fit on the GPU as one uint8 tensor (645 x 3 x 256 x 256, 127 MB),
so they are read once and an epoch is a permutation and some slicing, as for D1.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "data" / "d3_pv01"
FULL_SPLIT = REPO / "data" / "d3_split.json"
PINNED_SPLIT = REPO / "configs" / "d3_split.json"

PV01_CODES = frozenset({0, 211, 212, 213})
SUBSETS = ("Brick", "FlatConcrete", "SteelTile")
SPLITS = ("train", "val", "test", "guard")
HW = (256, 256)


def binarise(mask: np.ndarray) -> np.ndarray:
    """Panel = mask != 0, after checking every value is a PV01 code."""
    seen = {int(v) for v in np.unique(mask)}
    if not seen <= PV01_CODES:
        raise ValueError(f"mask values {sorted(seen - PV01_CODES)} are not PV01 codes "
                         f"{sorted(PV01_CODES)}")
    return np.asarray(mask) != 0


@dataclass
class D3Data:
    """Every tile and mask, already on the device, with split and subset indices."""

    images: torch.Tensor                    # uint8 (645, 3, 256, 256)
    masks: torch.Tensor                     # uint8 (645, 1, 256, 256), 0 or 1
    keys: list[str]                         # row -> "subset/name", the hash order
    index: dict[tuple[str, str], torch.Tensor]  # (split, subset) -> int64 rows
    mean: torch.Tensor                      # (1, 3, 1, 1), over TRAIN tiles only
    std: torch.Tensor
    split_sha256: str

    def rows(self, split: str) -> torch.Tensor:
        return torch.cat([self.index[(split, s)] for s in SUBSETS])

    def batch(self, rows: torch.Tensor) -> torch.Tensor:
        """uint8 rows -> standardised float. The only place normalisation happens."""
        return (self.images[rows].float().div_(255.0) - self.mean) / self.std


def load_d3(device: str | torch.device | None = None) -> D3Data:
    """Load PV01 onto `device`, refusing any split but the committed one.

    Guard tiles (scripts/split_d3.py) are loaded and sit in their own index; no
    caller trains on them, scores them or computes statistics from them.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    full = json.loads(FULL_SPLIT.read_text(encoding="utf-8"))
    pinned = json.loads(PINNED_SPLIT.read_text(encoding="utf-8"))
    assignment: dict[str, str] = full["assignment"]
    canonical = "\n".join(f"{k},{s}" for k, s in sorted(assignment.items()))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if digest != pinned["sha256"]:
        raise RuntimeError(
            f"split hash mismatch: data/d3_split.json is {digest[:12]}, "
            f"configs/d3_split.json pins {pinned['sha256'][:12]}. Re-run "
            "scripts/split_d3.py; if it still differs, the split rule changed."
        )

    keys = sorted(assignment)
    imgs = np.empty((len(keys), *HW, 3), dtype=np.uint8)
    masks = np.empty((len(keys), *HW), dtype=np.uint8)
    for i, k in enumerate(keys):
        sub, name = k.split("/")
        folder = ROOT / f"PV01_Rooftop_{sub}"
        a = np.asarray(Image.open(folder / f"{name}.bmp").convert("RGB"))
        if a.shape != (*HW, 3):
            raise ValueError(f"{k}: expected {(*HW, 3)}, got {a.shape}")
        imgs[i] = a
        masks[i] = binarise(np.asarray(Image.open(folder / f"{name}_label.bmp")))

    images = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous().to(device)
    mask_t = torch.from_numpy(masks).unsqueeze(1).to(device)
    index = {
        (sp, sub): torch.tensor([i for i, k in enumerate(keys)
                                 if assignment[k] == sp and k.startswith(sub + "/")],
                                dtype=torch.int64, device=device)
        for sp in SPLITS for sub in SUBSETS
    }
    assert sum(len(v) for v in index.values()) == len(keys), "split does not cover the tiles"

    # Per-channel statistics over the train split only, guard tiles excluded.
    train = images[torch.cat([index[("train", s)] for s in SUBSETS])].float().div_(255.0)
    mean = train.mean((0, 2, 3), keepdim=True)
    std = train.std((0, 2, 3), keepdim=True)
    del train

    print(f"D3 on {device}: {tuple(images.shape)} uint8  "
          + "  ".join(f"{sp}={sum(len(index[(sp, s)]) for s in SUBSETS)}" for sp in SPLITS)
          + f"  train mean={[round(v, 4) for v in mean.flatten().tolist()]}"
          + f" std={[round(v, 4) for v in std.flatten().tolist()]}")
    return D3Data(images=images, masks=mask_t, keys=keys, index=index,
                  mean=mean, std=std, split_sha256=digest)
