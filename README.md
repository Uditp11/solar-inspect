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
| 1 · Segmentation (site footprint) | D3 PV01 | IoU / Dice, **per subset** | TODO | TODO |
| 1 · Naive colour-threshold baseline | D3 PV01 | IoU / Dice, **per subset** | TODO | TODO |
| 2 · Detection | D2 | mAP@0.5 (mine) | TODO | 63 imgs / 3,984 boxes |
| 2 · Detection | D2 | mAP@0.5 (torchmetrics) | TODO | 63 imgs / 3,984 boxes |
| 2 · Detection | D2 | mAP@0.5:0.95 (torchmetrics) | TODO | 63 imgs / 3,984 boxes |
| 2 · Detection | D2 | per-image panel-count error | TODO | 63 imgs / 3,984 boxes |
| 3 · Tracking | D5 (synthetic) | MOTA / IDF1 / IDSW | TODO | TODO |
| 3 · Deduplication | D5 (synthetic) | N frames → M detections → K unique | TODO | TODO |
| 4 · Classification | D1 | macro-F1, ResNet-18 fine-tuned | **0.6956** | 3,007 imgs (test, once) |
| 4 · Classification | D1 | accuracy vs null model | 0.8251 vs 0.4988 | 3,007 imgs (test, once) |
| 4 · Classification | D1 | best from-scratch CNN, macro-F1 | 0.5980 ± 0.0085 | 2,988 imgs (val, 3 seeds) |
| 4 · Classification | D1 | macro-F1 at 2.2% base rate | TODO | TODO |
| 4 · Distillation | D1 | teacher vs student macro-F1 / params | TODO | TODO |

The test number comes from one evaluation, of a config
([`configs/cls_final.yaml`](configs/cls_final.yaml)) committed and pushed in
[`c1df507`](../../commit/c1df507) **before** the split was read;
[`scripts/eval_test_d1.py`](scripts/eval_test_d1.py) refuses to run otherwise. Per-class
recall with supports, the confusion matrix and the full ablation are in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). **D1 ships no official split, so none of these
numbers is comparable to a published one.**

**The classification rows carry a measured leakage bound, and it belongs here rather than
in a footnote.** Val and test each contain ~4% of images with a ≥ 0.98 cosine neighbour in
train (129 of 2,988 and 133 of 3,007). A controlled experiment — remove the 232 train
neighbours, retrain, 3 seeds, against a control removing the *same number of random
same-class images* — shows the effect is causal: leaky-subset accuracy falls **4.9 points**
under neighbour removal and **0.0000** under the random control. Diluted across the full
split the effect is **0.005 macro-F1**, inside a seed spread of 0.012–0.019, so it is not
measurable on the headline number at this budget. On the test split, for the model above,
the class-matched leaky-minus-clean difference is **+0.020 ± 0.026 (z = 0.79)** — smaller
than on the small CNN, because a better model has less headroom left to gain. Exact and
near-exact duplicates were removed; statistical near-neighbours were measured and bounded
rather than removed. See
[ADR 0004](docs/adr/0004-d1-near-duplicate-leakage-is-bounded-not-removed.md).
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
| D1 duplicate search + components | ✓ | — | SHA-256 byte equality (a subset of it) |
| Macro-F1, per-class recall | ✓ | — | `sklearn.metrics.f1_score`, every run |

---

## There is no temperature in this project

Neither D1 nor D2 carries radiometric data — both are 8-bit rendered imagery.
The field is `delta_dn_uncalibrated` (hotspot digital number minus panel median digital
number), computed **within a single frame only**, never across frames.

TODO — the full explanation, once the field is actually computed.

---

## Data

See [`docs/DATA.md`](docs/DATA.md) for sources, licences, checksums, formats and the
oddities found during EDA. No dataset is committed. Reproducing the data:

```bash
pip install -r requirements.txt     # 0. torch comes from PyTorch's index, not PyPI
python scripts/download_data.py     # 1. fetch, checksum and extract all four datasets
python scripts/split_d2.py          # 2. rebuild D2's split and regenerate configs/d2.yaml
python scripts/dedup_d1.py          # 3. exclude D1's near-identical crops
python scripts/split_d1.py          # 4. rebuild D1's split
```

**None of them is optional, and 3 comes before 4.** D2's published train/val/test splits
are contaminated and are not used here; `configs/d2.yaml` points at the rebuilt split,
which only exists once step 2 has run — see
[ADR 0002](docs/adr/0002-d2-is-resplit-by-sortie.md). D1 ships no split at all, and
splitting it before deduplicating leaves a near-identical pair straddling train and test
— see [ADR 0003](docs/adr/0003-d1-is-deduplicated-at-4-dn-by-component.md). Step 0 is not
optional either: without the `--extra-index-url` line in `requirements.txt`, pip resolves
`torch` to the CPU-only wheel and every GPU run here silently runs on the CPU.

---

## Modules

### 1 · Segmentation — site footprint

PV01 is 0.1 m RGB rooftop imagery from a single geography. It is a **site-footprint /
array-extent** primitive; it does not compose with the thermal stages and it cannot
normalise thermal statistics.

**Results are reported per rooftop subset, never pooled**, because the class balance
across the three subsets differs by a factor of ten:

| Subset | Masks | Positive pixels | A null all-background model scores |
|---|---:|---:|---:|
| Rooftop_Brick | 138 | 5.20% | 94.8% pixel accuracy, 0 IoU |
| Rooftop_FlatConcrete | 413 | 33.65% | 66.4% pixel accuracy, 0 IoU |
| Rooftop_SteelTile | 94 | 50.41% | 49.6% pixel accuracy, 0 IoU |
| **Pooled** | 645 | 30.00% | 70.0% pixel accuracy, 0 IoU |

That spread is the argument for per-subset reporting. Pooling hides it, and a pooled
figure would make the null model look uniformly mediocre when on Brick it is deceptively
strong and on SteelTile it is worse than a coin flip — the positive class is the
majority there.

TODO — measured IoU/Dice, the naive baseline, overlay figures, and the greyscaled-D2
negative result.

### 2 · Detection

**D2's published splits are not used.** 353 published files are only 252 unique images,
and 34 of the 35 published test images are byte-identical to a train or val image. The
split here is deduplicated by content hash and grouped by acquisition sortie —
**138 / 51 / 63 images, 10,612 / 4,929 / 3,984 boxes**. See
[ADR 0002](docs/adr/0002-d2-is-resplit-by-sortie.md).

Because this is not the published split, **no number below is comparable to a published
number on D2.**

`max_det` is set to **1000**, not Ultralytics' default of 300. Eight of the 252 unique
frames carry more than 300 boxes (max 584) and three of those land in the test split, so
the default would have silently truncated 3.2% of frames — and per-image panel-count
error, the domain metric, would have been wrong with no error message anywhere.

TODO — measured mAP, the NMS threshold sweep figure, panel-count error.

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
