"""PV01 tiles and masks.

Mask values in PV01 are exactly {0, 211, 212, 213} (docs/DATA.md); the three
non-zero codes are panel classes the published code list distinguishes and this
module does not. A value outside that set is refused, not counted as panel.
"""
from __future__ import annotations

import numpy as np

PV01_CODES = frozenset({0, 211, 212, 213})


def binarise(mask: np.ndarray) -> np.ndarray:
    """Panel = mask != 0, after checking every value is a PV01 code."""
    seen = {int(v) for v in np.unique(mask)}
    if not seen <= PV01_CODES:
        raise ValueError(f"mask values {sorted(seen - PV01_CODES)} are not PV01 codes "
                         f"{sorted(PV01_CODES)}")
    return np.asarray(mask) != 0
