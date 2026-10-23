"""The triage dispatch rule and the CoA table, tested first on hand-built cases.

Classes are D1's twelve, in the classifier's (alphabetical) order. Probabilities
are written out per case so every expected order can be checked by eye.

    flag      P(any anomaly) = 1 - P(No-Anomaly) >= t
    class     argmax over the eleven anomaly classes, never No-Anomaly
    order     CoA descending, then extent (module, substring, cell, external),
              then P(class) descending, then image name ascending

Cases, t = 0.5:
    a  Diode 0.9                      -> CoA 3, substring
    b  Offline-Module 0.6             -> CoA 2, module
    c  Hot-Spot 0.7                   -> CoA 3, cell
    d  Soiling 0.8                    -> CoA 2, external
    e  Diode 0.9 (same as a)          -> CoA 3, substring, ties a on P: by name
    f  No-Anomaly 0.45, Cell 0.30, Cracking 0.25
                                      -> flagged (0.55 >= 0.5), its top class is
                                         No-Anomaly; class Cell (0.30), CoA 2, cell
    g  No-Anomaly 0.9, Diode 0.1      -> 0.1 < 0.5, not dispatched
    Expected dispatch order: a, e, c, b, f, d
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from solar_inspect.analytics.triage import (
    COA_TABLE, EXTENT_ORDER, delta_dn_uncalibrated, dispatch,
)

CLASSES = ["Cell", "Cell-Multi", "Cracking", "Diode", "Diode-Multi", "Hot-Spot",
           "Hot-Spot-Multi", "No-Anomaly", "Offline-Module", "Shadowing", "Soiling",
           "Vegetation"]
ADR = Path(__file__).resolve().parents[1] / "docs" / "adr" / "0006-coa-by-failure-mode.md"


def row(**p: float) -> np.ndarray:
    v = np.zeros(len(CLASSES))
    for k, x in p.items():
        v[CLASSES.index(k.replace("_", "-"))] = x
    assert abs(v.sum() - 1) < 1e-12
    return v


CASES = {
    "a.jpg": row(Diode=0.9, No_Anomaly=0.1),
    "b.jpg": row(Offline_Module=0.6, No_Anomaly=0.4),
    "c.jpg": row(Hot_Spot=0.7, No_Anomaly=0.3),
    "d.jpg": row(Soiling=0.8, No_Anomaly=0.2),
    "e.jpg": row(Diode=0.9, No_Anomaly=0.1),
    "f.jpg": row(No_Anomaly=0.45, Cell=0.30, Cracking=0.25),
    "g.jpg": row(No_Anomaly=0.9, Diode=0.1),
}


def run(t: float = 0.5):
    names = list(CASES)
    return dispatch(names, np.stack([CASES[n] for n in names]), CLASSES, t)


def test_dispatch_order_and_ranks():
    out = run()
    sent = [r for r in out if r["triage"].dispatch_rank is not None]
    assert [r["image"] for r in sent] == ["a.jpg", "e.jpg", "c.jpg", "b.jpg", "f.jpg", "d.jpg"]
    assert [r["triage"].dispatch_rank for r in sent] == [1, 2, 3, 4, 5, 6]
    assert [(r["triage"].coa, r["triage"].extent) for r in sent] == [
        (3, "substring"), (3, "substring"), (3, "cell"), (2, "module"), (2, "cell"),
        (2, "external")]


def test_flagged_module_whose_top_class_is_no_anomaly_gets_an_anomaly_class():
    f = next(r for r in run() if r["image"] == "f.jpg")
    assert f["top_is_no_anomaly"] is True
    assert f["triage_class"] == "Cell"
    assert f["p_class"] == pytest.approx(0.30)
    assert f["p_any"] == pytest.approx(0.55)


def test_no_anomaly_module_is_never_dispatched():
    g = next(r for r in run() if r["image"] == "g.jpg")
    assert g["triage"].dispatch_rank is None
    assert (g["triage"].coa, g["triage"].extent, g["triage"].action) == (1, "none", "none")
    # and whatever the threshold, no dispatched row is CoA 1 or class No-Anomaly
    for t in (0.0, 0.05, 0.5, 0.99):
        for r in run(t):
            if r["triage"].dispatch_rank is not None:
                assert r["triage"].coa > 1 and r["triage_class"] != "No-Anomaly"


def test_threshold_is_inclusive():
    # b has P(any) = 1 - 0.4 = 0.6 (equal to 0.6 in float64); at t = 0.6 it is
    # flagged, just above it is not
    assert 1 - 0.4 == 0.6
    by = {r["image"]: r for r in run(0.6)}
    assert by["b.jpg"]["triage"].dispatch_rank is not None
    by = {r["image"]: r for r in run(0.6000001)}
    assert by["b.jpg"]["triage"].dispatch_rank is None


def test_extent_breaks_a_coa_tie_before_probability():
    # Offline-Module (module) at 0.51 goes before Cell (cell) at 0.95
    names = ["x.jpg", "y.jpg"]
    p = np.stack([row(Cell=0.95, No_Anomaly=0.05), row(Offline_Module=0.51, No_Anomaly=0.49)])
    out = dispatch(names, p, CLASSES, 0.5)
    assert [r["image"] for r in sorted(out, key=lambda r: r["triage"].dispatch_rank)] == \
        ["y.jpg", "x.jpg"]


def test_coa_table_covers_every_class_and_matches_adr_0006():
    assert set(COA_TABLE) == set(CLASSES)
    assert set(EXTENT_ORDER) == {"module", "substring", "cell", "external"}
    text = ADR.read_text(encoding="utf-8")
    rows = re.findall(r"^\| ([A-Za-z-]+) \| [^|]+ \| (\d) \| (\w+) \| ([^|]+) \|", text, re.M)
    adr = {c: (int(coa), ext, act.strip()) for c, coa, ext, act in rows if c in CLASSES}
    assert adr == COA_TABLE


def test_delta_dn_uncalibrated_by_hand():
    # 100 pixels: 0..99. Median 49.5; numpy's linear 99th percentile is 98.01.
    crop = np.arange(100, dtype=np.uint8).reshape(10, 10)
    assert delta_dn_uncalibrated(crop) == pytest.approx(98.01 - 49.5)
    assert delta_dn_uncalibrated(np.full((40, 24), 7, np.uint8)) == 0.0
