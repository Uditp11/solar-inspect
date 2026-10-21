"""Availability and the sister ratio, on toy frames, tested before either exists.

Availability, inverters A and B, counting threshold 0.1:

    t1  irr 0.05  A 0   B 0     not counted at 0.1
    t2  irr 0.10  A 10  B 0     counted (the threshold is inclusive)
    t3  irr 0.80  A 12  B 9
    t4  irr 0.60  A 11  B --    B has no row: a data gap, not an outage

    at 0.1:  A 3/3 = 1,        B 1/2 = 0.5,  B missing 1 of 3 counted timestamps
    at 0.01: A 3/4 = 0.75,     B 1/3,        B missing 1 of 4

Sister ratio, inverters A, B, C, threshold 0.1:

    t1  irr 0.5   A 10  B 20  C 0    producing median 15 -> 2/3, 4/3, 0
    t2  irr 0.5   A 0   B 0   C 0    nobody producing: no reference, no ratio
    t3  irr 0.5   A 9   B 12  C 6    producing median 9  -> 1, 4/3, 2/3
    t4  irr 0.05  A 100 B 1   C 1    not counted

    medians: A (2/3, 1) -> 5/6, B 4/3, C (0, 2/3) -> 1/3, each from 2 timestamps
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from solar_inspect.analytics.inverters import availability, sister_ratio


def frame(rows):
    return pd.DataFrame(rows, columns=["DATE_TIME", "SOURCE_KEY", "IRRADIATION", "AC_POWER"])


AVAIL = frame([
    (1, "A", 0.05, 0.0), (1, "B", 0.05, 0.0),
    (2, "A", 0.10, 10.0), (2, "B", 0.10, 0.0),
    (3, "A", 0.80, 12.0), (3, "B", 0.80, 9.0),
    (4, "A", 0.60, 11.0),
])

SISTER = frame([
    (1, "A", 0.5, 10.0), (1, "B", 0.5, 20.0), (1, "C", 0.5, 0.0),
    (2, "A", 0.5, 0.0), (2, "B", 0.5, 0.0), (2, "C", 0.5, 0.0),
    (3, "A", 0.5, 9.0), (3, "B", 0.5, 12.0), (3, "C", 0.5, 6.0),
    (4, "A", 0.05, 100.0), (4, "B", 0.05, 1.0), (4, "C", 0.05, 1.0),
])


def test_availability_at_declared_threshold():
    out = availability(AVAIL, 0.1)
    assert out.loc["A", ["counted", "producing", "missing"]].tolist() == [3, 3, 0]
    assert out.loc["B", ["counted", "producing", "missing"]].tolist() == [2, 1, 1]
    assert out.loc["A", "availability"] == 1.0
    assert out.loc["B", "availability"] == 0.5


def test_availability_lower_threshold_counts_dawn():
    out = availability(AVAIL, 0.01)
    assert out.loc["A", "availability"] == pytest.approx(3 / 4)
    assert out.loc["B", "availability"] == pytest.approx(1 / 3)
    assert out.loc["B", "missing"] == 1


def test_sister_ratio_hand_example():
    out = sister_ratio(SISTER, 0.1)
    np.testing.assert_allclose(out.loc[["A", "B", "C"], "sister_ratio"], [5 / 6, 4 / 3, 1 / 3])
    assert out["n"].tolist() == [2, 2, 2]


def test_sister_ratio_all_zero_timestamp_gives_no_ratio_and_no_division_by_zero():
    only_zero = SISTER[SISTER["DATE_TIME"] == 2]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sister_ratio(only_zero, 0.1)
        full = sister_ratio(SISTER, 0.1)
    assert out["n"].tolist() == [0, 0, 0]
    assert out["sister_ratio"].isna().all()
    assert np.isfinite(full["sister_ratio"]).all()
