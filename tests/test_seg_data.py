"""PV01 mask binarisation, tested before it exists.

PV01 masks hold exactly {0, 211, 212, 213} (docs/DATA.md). The published code list
is wider (11/12, 111/121-125 belong to PV03 and PV08), so a value outside the PV01
set means the wrong subset, or a resampled mask, and is refused rather than
silently counted as panel.
"""
from __future__ import annotations

import numpy as np
import pytest

from solar_inspect.segmentation.data import binarise


def test_binarise_is_mask_not_zero():
    m = np.array([[0, 211], [212, 213]], dtype=np.uint8)
    assert binarise(m).tolist() == [[False, True], [True, True]]
    assert binarise(m).dtype == bool


def test_binarise_accepts_all_background():
    assert not binarise(np.zeros((4, 4), dtype=np.uint8)).any()


@pytest.mark.parametrize("bad", [11, 12, 111, 125, 1, 255])
def test_binarise_refuses_codes_outside_pv01(bad):
    m = np.array([[0, 211], [bad, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match=str(bad)):
        binarise(m)
