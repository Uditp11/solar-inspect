"""Triage: a Class of Abnormality per finding, by failure mode, and the dispatch order.

COA_TABLE is ADR 0006's table (tests/test_triage.py checks the two agree). The
CoA comes from the predicted class alone; no pixel value moves it, because D1's
pixels are not radiometric.

The dispatch rule (configs/triage.yaml):
- flag a module when calibrated P(any anomaly) = 1 - P(No-Anomaly) >= t;
- its class is the argmax over the eleven anomaly classes, so a flagged module
  whose single largest probability is No-Anomaly still gets an anomaly class;
- order: CoA descending, then extent in EXTENT_ORDER, then P(class) descending,
  then image name ascending.

`delta_dn_uncalibrated` is computed per crop and carried, never ranked on.
"""
from __future__ import annotations

import numpy as np

from solar_inspect.contract import EXTENTS, Triage

NULL_CLASS = "No-Anomaly"
EXTENT_ORDER = EXTENTS              # module, substring, cell, external

_DIODE = "inspect bypass diode and junction box; replace module if faulty"
_HOT = "inspect on site with a radiometric camera; replace if confirmed"
_CELL = "inspect; replace if damage confirmed"
COA_TABLE: dict[str, tuple[int, str, str]] = {   # class -> (CoA, extent, action)
    "No-Anomaly": (1, "none", "none"),
    "Diode": (3, "substring", _DIODE),
    "Diode-Multi": (3, "substring", _DIODE),
    "Hot-Spot": (3, "cell", _HOT),
    "Hot-Spot-Multi": (3, "cell", _HOT),
    "Offline-Module": (2, "module", "check wiring and connectors"),
    "Cell": (2, "cell", _CELL),
    "Cell-Multi": (2, "cell", _CELL),
    "Cracking": (2, "cell", _CELL),
    "Soiling": (2, "external", "clean"),
    "Vegetation": (2, "external", "cut vegetation"),
    "Shadowing": (2, "external", "remove shade source"),
}


def delta_dn_uncalibrated(crop: np.ndarray) -> float:
    """99th-percentile DN minus median DN of one raw 8-bit crop (numpy's linear percentile).

    An intra-crop contrast, not a temperature, and not comparable between crops.
    """
    a = np.asarray(crop, dtype=np.float64)
    return float(np.percentile(a, 99) - np.median(a))


def dispatch(images: list[str], probs: np.ndarray, classes: list[str], t: float
             ) -> list[dict]:
    """One row per image: its Triage and the triage class. Dispatched rows first, in order.

    probs are calibrated (n, 12) probabilities in `classes` order. Undispatched
    rows follow in input order with CoA 1, extent "none" and no rank.
    """
    na = classes.index(NULL_CLASS)
    anomaly = [i for i in range(len(classes)) if i != na]
    sent, rest = [], []
    for img, p in zip(images, np.asarray(probs, dtype=np.float64)):
        k = anomaly[int(np.argmax(p[anomaly]))]
        r = {"image": img, "triage_class": classes[k], "p_class": float(p[k]),
             "p_any": float(1.0 - p[na]), "top_is_no_anomaly": int(np.argmax(p)) == na}
        (sent if r["p_any"] >= t else rest).append(r)
    sent.sort(key=lambda r: (-COA_TABLE[r["triage_class"]][0],
                             EXTENT_ORDER.index(COA_TABLE[r["triage_class"]][1]),
                             -r["p_class"], r["image"]))
    for rank, r in enumerate(sent, 1):
        coa, extent, action = COA_TABLE[r["triage_class"]]
        r["triage"] = Triage(r["image"], coa, extent, action, dispatch_rank=rank)
    for r in rest:
        r["triage"] = Triage(r["image"], *COA_TABLE[NULL_CLASS])
    return sent + rest
