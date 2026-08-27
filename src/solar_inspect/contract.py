"""The one record every module speaks, plus the RLE codec that carries its masks.

Imports are stdlib, numpy and pycocotools only. **No torch.** Every module in the
project imports this file, so a heavy import here is a heavy import everywhere --
including in the analytics module, which never touches a GPU.

Masks are pycocotools *compressed* RLE: column-major (Fortran order), alternating
runs, starting with a zero-run. `counts` is stored as a UTF-8 `str` so a Finding
is JSON-serialisable without a bytes shim.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pycocotools import mask as coco_mask


def encode_rle(mask: np.ndarray) -> tuple[str, tuple[int, int]]:
    """Encode a 2-D binary mask to (counts, (H, W))."""
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    if mask.dtype.kind not in "biu":
        # A float probability map would astype() to all zeros below and encode as
        # an empty mask without complaint. Threshold before calling this.
        raise ValueError(f"expected a binary mask, got dtype {mask.dtype}")
    # asfortranarray is load-bearing: pycocotools rejects C-contiguous input with
    # "ndarray is not Fortran contiguous". The dangerous input is a *transposed
    # view* -- mask.T of a C array is already Fortran contiguous, so it encodes
    # happily with H and W swapped, which on a square mask is undetectable.
    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("utf-8"), (mask.shape[0], mask.shape[1])


def decode_rle(counts: str, shape: tuple[int, int]) -> np.ndarray:
    """Decode (counts, (H, W)) back to a uint8 mask of that shape."""
    h, w = shape
    return coco_mask.decode({"counts": counts.encode("utf-8"), "size": [h, w]})


@dataclass
class Finding:
    """One detected thing, in one frame, at whatever stage of the pipeline.

    Modules fill this in progressively -- detection sets bbox and det_score, the
    tracker sets track_id, segmentation sets the mask, classification sets the
    class. Nothing imports another module's internals; this record and files on
    disk are the whole interface.

    `est_power_loss_pct` is deliberately absent. Severity lives in a separate
    Triage record so the architecture itself carries the caveat that this data
    cannot support a power-loss number.
    """

    finding_id: str
    frame_id: str
    frame_index: int
    source_frame_path: str
    bbox: tuple[float, float, float, float]     # xyxy, absolute px
    det_score: float                            # detector confidence
    track_id: int | None = None                 # Module 3
    mask_rle: str | None = None                 # pycocotools compressed RLE
    mask_shape: tuple[int, int] | None = None   # REQUIRED to decode RLE
    defect_class: str | None = None             # Module 4
    cls_confidence: float | None = None         # calibrated
    row: int | None = None
    delta_dn_uncalibrated: float | None = None  # NOT temperature -- see DATA.md

    def __post_init__(self) -> None:
        if self.mask_rle is not None and self.mask_shape is None:
            raise ValueError(
                "mask_shape is mandatory alongside mask_rle: compressed RLE is "
                "undecodable without (H, W)."
            )
