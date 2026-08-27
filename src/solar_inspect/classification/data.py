"""D1 as one tensor on the GPU. No DataLoader anywhere in this module.

20,000 crops x 40 x 24 x 1 byte is 19.2 MB. That fits in VRAM with room to spare,
so the whole dataset is uploaded once and an epoch is a permutation and some
slicing. A DataLoader here would read 20,000 files per epoch to feed a model that
trains in minutes, and on Windows `num_workers > 0` brings the spawn trap with
it. Not writing one sidesteps the problem rather than guarding against it.

Orientation matters and is asserted, not assumed. D1 is **portrait**: PIL reports
`im.size == (24, 40)`, which is (W, H), so every tensor here is (N, 1, 40, 24).
A conv stack written against (24, 40) is transposed, still trains, and still
produces a plausible macro-F1. See docs/DATA.md.
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
D1 = REPO / "data" / "d1"
CACHE = REPO / "data" / "d1_cache.npy"
FULL_SPLIT = REPO / "data" / "d1_split.json"
PINNED_SPLIT = REPO / "configs" / "d1_split.json"

HW = (40, 24)                       # (H, W). Portrait. Not negotiable, see above.
SPLITS = ("train", "val", "test")


@dataclass
class D1Data:
    """Everything a training step needs, already on the device."""

    images: torch.Tensor                    # uint8 (N, 1, 40, 24)
    labels: torch.Tensor                    # int64 (N,)
    index: dict[str, torch.Tensor]          # split -> int64 row indices
    classes: list[str]                      # label id -> class name
    paths: list[str]                        # row -> image_filepath, the row order
    mean: float                             # over the TRAIN split only
    std: float

    def batch(self, rows: torch.Tensor) -> torch.Tensor:
        """uint8 rows -> standardised float. The only place normalisation happens."""
        return (self.images[rows].float().div_(255.0) - self.mean) / self.std


def _build_cache(paths: list[str]) -> np.ndarray:
    out = np.empty((len(paths), *HW), dtype=np.uint8)
    for i, rel in enumerate(paths):
        a = np.asarray(Image.open(D1 / rel))
        if a.shape != HW or a.dtype != np.uint8:
            raise ValueError(f"{rel}: expected uint8 {HW}, got {a.dtype} {a.shape}")
        out[i] = a
        if (i + 1) % 5000 == 0:
            print(f"  decoded {i + 1}/{len(paths)}")
    return out


def load_d1(device: str | torch.device | None = None) -> D1Data:
    """Load D1 onto `device`, verifying the split hash and the tensor shape."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    meta = json.loads((D1 / "module_metadata.json").read_text(encoding="utf-8"))
    full = json.loads(FULL_SPLIT.read_text(encoding="utf-8"))
    pinned = json.loads(PINNED_SPLIT.read_text(encoding="utf-8"))
    assignment: dict[str, str] = full["assignment"]

    # Recompute the hash rather than trusting the one written beside the data.
    # This is what stops a re-run of split_d1.py with a different seed from
    # silently redefining "val" between now and the single test evaluation.
    canonical = "\n".join(f"{p},{s}" for p, s in sorted(assignment.items()))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if digest != pinned["sha256"]:
        raise RuntimeError(
            f"split hash mismatch: data/d1_split.json is {digest[:12]}, "
            f"configs/d1_split.json pins {pinned['sha256'][:12]}. The split on disk "
            "is not the committed one -- re-run scripts/split_d1.py, and if it still "
            "differs the seed or the ratios changed and every number measured "
            "against the old split is now on a different dataset."
        )

    paths = sorted(e["image_filepath"] for e in meta.values())
    n_on_disk = sum(1 for _ in (D1 / "images").glob("*.jpg"))
    if n_on_disk != len(paths):
        raise RuntimeError(f"metadata lists {len(paths)} images, {n_on_disk} on disk")

    # Cache keyed on the file count: a re-download with a different number of
    # images invalidates it. Row order is sorted image_filepath, same order the
    # split hash is computed over.
    if CACHE.exists() and (arr := np.load(CACHE)).shape[0] == n_on_disk:
        print(f"cache hit: {CACHE.relative_to(REPO).as_posix()} {arr.shape}")
    else:
        print(f"decoding {len(paths)} JPEGs once -> {CACHE.relative_to(REPO).as_posix()}")
        arr = _build_cache(paths)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(CACHE, arr)

    if arr.shape != (len(paths), *HW):
        raise ValueError(f"expected {(len(paths), *HW)}, got {arr.shape}")

    classes = sorted({e["anomaly_class"] for e in meta.values()})
    label_of = {e["image_filepath"]: classes.index(e["anomaly_class"]) for e in meta.values()}

    images = torch.from_numpy(arr).unsqueeze(1).to(device)          # (N, 1, 40, 24)
    labels = torch.tensor([label_of[p] for p in paths], dtype=torch.int64, device=device)
    assert images.shape == (len(paths), 1, *HW), f"tensor is {tuple(images.shape)}, want (N, 1, 40, 24)"

    index = {
        s: torch.tensor([i for i, p in enumerate(paths) if assignment[p] == s],
                        dtype=torch.int64, device=device)
        for s in SPLITS
    }
    assert sum(len(v) for v in index.values()) == len(paths), "split does not cover the data"

    # Statistics over the train split only. Computing them over all 20,000 leaks
    # val and test pixel intensities into the normalisation -- small, but it is
    # the kind of thing that gets asked about.
    train_px = images[index["train"]].float().div_(255.0)
    mean, std = float(train_px.mean()), float(train_px.std())
    del train_px

    print(f"D1 on {device}: {tuple(images.shape)} uint8, "
          + "  ".join(f"{s}={len(index[s])}" for s in SPLITS)
          + f"  train mean={mean:.4f} std={std:.4f}")
    return D1Data(images=images, labels=labels, index=index, classes=classes,
                  paths=paths, mean=mean, std=std)


if __name__ == "__main__":
    d = load_d1()
    print(d.classes)
    print("batch:", d.batch(d.index["val"][:8]).shape, d.batch(d.index["val"][:8]).dtype)
