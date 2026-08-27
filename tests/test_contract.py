"""RLE round-trip and Finding validation.

Written before contract.py exists. The masks here are small enough that every
expected value is hand-computable, which is the point: these tests are the
evidence that the RLE convention is understood rather than trusted.
"""
from __future__ import annotations

import numpy as np
import pytest

from solar_inspect.contract import Finding, decode_rle, encode_rle


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two binary masks. Hand-checkable."""
    a, b = a.astype(bool), b.astype(bool)
    return float((a & b).sum()) / float((a | b).sum())


def test_rle_roundtrip_preserves_mask():
    """The plan's round-trip: 8x10, rectangle at [2:5, 3:7]."""
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    counts, shape = encode_rle(mask)
    assert shape == (8, 10)
    assert isinstance(counts, str)
    assert np.array_equal(decode_rle(counts, shape), mask)


def test_rle_roundtrip_non_square_asymmetric():
    """The test that pins column-major order.

    COCO RLE is Fortran order. Measured, not assumed: pycocotools rejects a
    C-contiguous array outright ("ndarray is not Fortran contiguous"), so the
    missing asfortranarray fails loudly. What does NOT fail loudly is a
    transposed *view* -- mask.T of a C array is already Fortran contiguous and
    encodes happily with H and W swapped, which a square fixture cannot detect.
    A non-square fixture catches it on the shape, and `wrong` below -- what a
    C-order misread of the buffer produces -- catches a same-shape scramble.
    """
    mask = np.array(
        [
            [1, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert mask.shape == (4, 6)
    wrong = mask.ravel(order="C").reshape(mask.shape, order="F")
    assert not np.array_equal(mask, wrong), "fixture is order-blind, pick another"

    counts, shape = encode_rle(mask)
    decoded = decode_rle(counts, shape)
    assert shape == (4, 6)
    assert np.array_equal(decoded, mask)
    assert not np.array_equal(decoded, wrong)


@pytest.mark.parametrize("fill", [0, 1])
def test_rle_roundtrip_degenerate_masks(fill):
    """All-zero and all-ones.

    Compressed RLE alternates runs starting with a zero-run, so an all-ones mask
    is encoded as a zero-run of length 0 followed by one run of H*W. That leading
    empty run is where hand-rolled conventions break.
    """
    mask = np.full((6, 7), fill, dtype=np.uint8)
    counts, shape = encode_rle(mask)
    decoded = decode_rle(counts, shape)
    assert shape == (6, 7)
    assert np.array_equal(decoded, mask)
    assert int(decoded.sum()) == fill * 42


def test_rle_roundtrip_iou():
    """The encode -> decode -> IoU round-trip required by spec 7.

    Identical masks give exactly 1.0. Against a mask shifted by (+1, +1) the
    value is hand-computed: both rectangles are 3x4 = 12 px, they overlap on
    rows 3:5 and cols 4:7 = 2*3 = 6 px, so IoU = 6 / (12 + 12 - 6) = 1/3.
    """
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    shifted = np.zeros((8, 10), dtype=np.uint8)
    shifted[3:6, 4:8] = 1

    decoded = decode_rle(*encode_rle(mask))
    assert mask_iou(decoded, mask) == 1.0
    assert mask_iou(decoded, shifted) == pytest.approx(1.0 / 3.0)


def test_finding_requires_mask_shape_with_rle():
    """mask_shape is mandatory alongside mask_rle -- RLE alone is undecodable."""
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    counts, shape = encode_rle(mask)
    common = dict(
        finding_id="f0",
        frame_id="frame_0001",
        frame_index=1,
        source_frame_path="data/d2/train/images/frame_0001.jpg",
        bbox=(10.0, 20.0, 30.0, 40.0),
        det_score=0.9,
    )

    with pytest.raises(ValueError, match="mask_shape"):
        Finding(**common, mask_rle=counts)

    ok = Finding(**common, mask_rle=counts, mask_shape=shape)
    assert np.array_equal(decode_rle(ok.mask_rle, ok.mask_shape), mask)

    bare = Finding(**common)          # no mask at all is legal -- detection only
    assert bare.mask_rle is None and bare.track_id is None
