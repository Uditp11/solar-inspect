# PV Inspection Pipeline — Thermal Defect Detection, Tracking and Analytics

> **Status: in progress.** Every `TODO` below is a number that has not been measured yet.
> Nothing in this file is a claim until it points at a committed run.

Four perception modules trained and benchmarked independently on public solar imagery,
sharing one `Finding` data contract, plus an analytics layer on real plant generation data.
Metric primitives are written from scratch and unit-tested; production models are fine-tuned
from libraries. The seam between the two is stated below and nowhere contradicted.

---

## Scope — what this is and what it is not

TODO — written last, describing only what actually ran.

**It is not:** TODO

---

## Results

One row per module. Every number re-measured against its output file before shipping,
and reported with the size of the split it was measured on.

| Module | Data | Metric | Value | Split size |
|---|---|---|---|---|
| 1 · Segmentation (site footprint) | D3 PV01 | IoU / Dice | TODO | TODO |
| 1 · Naive colour-threshold baseline | D3 PV01 | IoU / Dice | TODO | TODO |
| 2 · Detection | D2 | mAP@0.5 (mine) | TODO | TODO |
| 2 · Detection | D2 | mAP@0.5 (torchmetrics) | TODO | TODO |
| 2 · Detection | D2 | mAP@0.5:0.95 (torchmetrics) | TODO | TODO |
| 2 · Detection | D2 | per-image panel-count error | TODO | TODO |
| 3 · Tracking | D5 (synthetic) | MOTA / IDF1 / IDSW | TODO | TODO |
| 3 · Deduplication | D5 (synthetic) | N frames → M detections → K unique | TODO | TODO |
| 4 · Classification | D1 | macro-F1 | TODO | TODO |
| 4 · Classification | D1 | macro-F1 at 2.2% base rate | TODO | TODO |
| 4 · Distillation | D1 | teacher vs student macro-F1 / params | TODO | TODO |
| 5 · Analytics | D4 | per-inverter slope ranking | TODO | TODO |

---

## The from-scratch seam

**From scratch** = I wrote the algorithm and the module composition, using `torch.nn` layers
and NumPy primitives. I did not implement autograd, cuDNN kernels, or LAPACK routines.

| Component | Written by me | Library | Cross-checked against |
|---|---|---|---|
| IoU, NMS, mAP@0.5 | ✓ | — | torchmetrics |
| mAP@0.5:0.95 | — | torchmetrics | — |
| U-Net | ✓ (`nn.Conv2d` etc.) | — | — |
| Detector | — | Ultralytics YOLOv8 | — |
| Kalman filter | ✓ | — | analytic 2-step test |
| Assignment | — | `scipy.optimize.linear_sum_assignment` | — |
| MOTA/IDF1 | — | motmetrics | — |
| ID switches, unique-module error | ✓ | — | hand-computed toy |
| OLS (normal equations, GD) | ✓ | — | `np.linalg.lstsq`, scikit-learn |

---

## There is no temperature in this project

Neither D1 nor D2 carries radiometric data — both are 8-bit rendered imagery.
The field is `delta_dn_uncalibrated` (hotspot digital number minus panel median digital
number), computed **within a single frame only**, never across frames.

TODO — the full explanation, once the field is actually computed.

---

## Data

See [`docs/DATA.md`](docs/DATA.md) for sources, licences, checksums, formats and the
oddities found during EDA. No dataset is committed; `scripts/download_data.py` reproduces
all four from scratch.

---

## Modules

### 1 · Segmentation — site footprint
TODO

### 2 · Detection
TODO

### 3 · Tracking
TODO

### 4 · Classification and distillation
TODO

### 5 · Analytics and triage
TODO

---

## Tests

TODO — paste the local `pytest` output.

---

## What I didn't do

TODO

---

## Licence

AGPL-3.0. Ultralytics YOLOv8 is AGPL-3.0, and their licence terms require publicly
releasing the complete corresponding source of the entire derivative work — so this repo
is AGPL-3.0 too. Shipping something like this commercially, you would want an enterprise
licence or a differently-licensed detector, which is one reason the metrics here are
written by hand rather than pulled in from more of the stack.
