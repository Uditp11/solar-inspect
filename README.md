# PV Inspection Pipeline: thermal defect detection, tracking and analytics

> **Status (22 Oct 2026): in progress.** Module 4 (classification) and module 2 (detection) are
> finished. Module 5's regression and per-inverter analysis on plant generation data is
> finished; its triage and SQL layer is not built yet. Segmentation and tracking are not built
> yet.

The plan is four perception modules trained and benchmarked independently on public solar imagery,
sharing one `Finding` data contract, plus an analytics layer on real plant generation data.
Metric primitives are written from scratch and unit-tested; the main models are fine-tuned
from libraries. The seam between the two is stated below.

---

## Results: module 4, classification

Every number below is logged in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) with the
commit, config and split it came from, and is reported with the size of the split it was
measured on.

| Module | Data | Metric | Value | Split size |
|---|---|---|---|---|
| 4 · Classification | D1 | macro-F1, ResNet-18 fine-tuned | **0.6956** | 3,007 imgs (test, once) |
| 4 · Classification | D1 | accuracy vs null model, ResNet-18 fine-tuned | 0.8251 vs 0.4988 | 3,007 imgs (test, once) |
| 4 · Classification | D1 | best from-scratch CNN, macro-F1 | 0.5980 ± 0.0085 | 2,988 imgs (val, 3 seeds) |
| 4 · Classification | D1 | triage precision @ 2.2% prevalence, ResNet-18 fine-tuned | 0.502 at recall 0.896 | 3,007 imgs (test, once) |
| 4 · Classification | D1 | ECE, global / class-balanced (T-scaled), ResNet-18 fine-tuned | 0.021 / 0.184 | 3,007 imgs (test, once) |
| 4 · Distillation | D1 | teacher ResNet-18 / student from-scratch CNN / distilled macro-F1 | 0.6735 / 0.5980 ± 0.0085 / 0.5987 ± 0.0075 | 2,988 imgs (val) |
| 4 · Distillation | D1 | params, teacher vs student | 11,176,396 vs 115,948 (96×) | n/a |
| 4 · Distillation | D1 | CPU latency, batch 64, 1 thread | 706.1 ms vs 228.7 ms (3.1×) | n/a |

The test number comes from one evaluation, of a config
([`configs/cls_final.yaml`](configs/cls_final.yaml)) committed and pushed in
[`1a0fdfa`](../../commit/1a0fdfa) **before** the split was read;
[`scripts/eval_test_d1.py`](scripts/eval_test_d1.py) refuses to run otherwise. Per-class
recall with supports, the confusion matrix and the full ablation are in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). **D1 ships no official split, so none of these
numbers is comparable to a published one.**

**The classification rows carry a measured leakage bound, and it belongs here rather than
in a footnote.** Val and test each contain ~4% of images with a ≥ 0.98 cosine neighbour in
train (129 of 2,988 and 133 of 3,007). A controlled experiment shows the effect is causal:
remove the 232 train neighbours, retrain over 3 seeds, and compare against a control that
removes the *same number of random same-class images*. Leaky-subset accuracy falls
**4.9 points** under neighbour removal and **0.0000** under the random control. Diluted
across the full split the effect is **0.005 macro-F1**, inside a seed spread of 0.012 to
0.019, so it is not measurable on the headline number at this budget. On the test split,
for the model above, the class-matched leaky-minus-clean difference is
**+0.020 ± 0.026 (z = 0.79)**. That is smaller than on the small CNN, because a better
model has less headroom left to gain. Exact and near-exact duplicates were removed;
statistical near-neighbours were measured and bounded rather than removed. See
[ADR 0004](docs/adr/0004-d1-near-duplicate-leakage-is-bounded-not-removed.md).

## Results: module 5, analytics on plant generation data (regression half)

Every number below is logged in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) (the D4
sections) with the commit and script that produced it, and is given as **Plant 1 / Plant 2**
beside the rows, inverters and timestamps it was measured on. D4 is 34 days of 15-minute
inverter data from two plants, one weather sensor each. **Everything here is in-sample**:
nothing is held out, so these numbers describe those 34 days and are not a predictive score.

| Module | Data | Metric | Value | Split size |
|---|---|---|---|---|
| 5 · Regression | D4 | R², AC power on irradiation, module and ambient temperature, daylight rows | 0.9602 / 0.4354 | 38,376 / 38,722 rows · 22 / 22 inverters · 1,754 / 1,862 timestamps |
| 5 · Regression | D4 | R² with night rows left in (inflation) | 0.9790 / 0.6136 (+0.019 / +0.178) | 68,774 / 67,698 rows · 22 / 22 inverters |
| 5 · Regression | D4 | standard error ÷ classical, Newey-West within inverter | 1.31 to 1.90 / 1.62 to 3.09 | 38,376 / 38,722 rows · 1,754 / 1,862 timestamps |
| 5 · Regression | D4 | standard error ÷ classical, Driscoll-Kraay (shared sensor) | 3.53 to 5.33 / 3.50 to 5.17 | 38,376 / 38,722 rows · 1,754 / 1,862 timestamps |
| 5 · Per inverter | D4 | availability, median (min) | 1.0000 (0.9857) / 0.8714 (0.7716) | 30,718 / 28,818 counted rows · 22 / 22 inverters · 1,397 / 1,392 counted timestamps |
| 5 · Per inverter | D4 | slope intervals excluding the plant median, 95% (Bonferroni) | 9 (4) / 9 (5) of 22 | 30,655 / 25,044 producing counted rows · 22 / 22 inverters |
| 5 · Per inverter | D4 | bottom 3 by slope also bottom 3 by sister ratio · Spearman of the two rankings | 2 / 2 of 3 · 0.740 / 0.907 | 30,655 / 25,044 producing counted rows · 22 / 22 inverters |
| 5 · Per inverter | D4 | Spearman, declared availability vs common window | 1.000 / 0.914 | 3,057 of 3,157 / 2,355 of 3,259 timestamps · 22 / 22 inverters |

The standard-error ratios are ranges over the four coefficients. A counted row is an
inverter's 15-minute interval with irradiation ≥ 0.1 kW/m²; availability is the share of
counted rows with any AC output, and the slope is a per-inverter fit of AC power on
irradiation over its producing counted rows, with a 95% Newey-West interval. All of these
rules were committed in [`configs/d4_analysis.yaml`](configs/d4_analysis.yaml) before any
per-inverter number existed.

**What it shows.** The two lowest-slope inverters of each plant are clear of the rest under
every check: the robustness model, the regression-free sister ratio, Bonferroni, and a pooled
fit whose errors allow for the shared sensor. Their AC/DC conversion matches their sisters'
(a DC-slope ranking agrees with the AC one at Spearman 0.999 / 1.000), so their shortfall is
already in the DC they receive, on the array side, not in the inverter. Which array-side cause
it is, D4 cannot say. Plant 2's low R² is mostly zero-output intervals under sun: 3,774
counted intervals at zero, of which 423 have more than half the plant at zero, 3,264 a
smaller group and 87 a single inverter, concentrated at midday. The cleaning findings, the
diagnostics, the confounders D4 can and cannot measure, and the IEC 61724 terms it lacks the
inputs for are in [`docs/explainers/05-analytics.md`](docs/explainers/05-analytics.md).

## Results: module 2, detection

Every number below is logged in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) (the D2 sections)
with the commit and run it came from. Test is 63 images, 3,984 boxes, read once; each interval
is a 95% percentile bootstrap over the 63 images (2,000 resamples) from that one pass's saved
predictions. The operating confidence is 0.25, chosen on val.

| Module | Data | Metric | Value | 95% interval | Split size |
|---|---|---|---|---|---|
| 2 · Detection | D2 | AP@0.5, 101-point, from scratch (= torchmetrics), YOLOv8n fine-tuned | **0.7552** | 0.6418 to 0.8438 | 63 imgs, 3,984 boxes (test, once) |
| 2 · Detection | D2 | AP@0.5:0.95, 101-point (= torchmetrics) | 0.5578 | 0.4675 to 0.6374 | 63 imgs, 3,984 boxes (test, once) |
| 2 · Detection | D2 | precision at conf 0.25, IoU 0.5 | 0.6025 | 0.4925 to 0.6872 | 63 imgs, 3,984 boxes (test, once) |
| 2 · Detection | D2 | recall at conf 0.25, IoU 0.5 | 0.7899 | 0.6969 to 0.8637 | 63 imgs, 3,984 boxes (test, once) |
| 2 · Detection | D2 | panel-count MAE at conf 0.25, panels per frame | 25.48 | 19.65 to 31.94 | 63 imgs, 3,984 boxes (test, once) |

Beside the headline, on the same test pass: AP@0.5 **all-point 0.7571** (the area under the
whole precision envelope, VOC style, rather than COCO's 101 samples of it), and **Ultralytics'
own 0.7562** (it scores boxes before clipping them to the frame and integrates its curve with the
trapezoid rule, both measured on val). The config ([`configs/det_yolo.yaml`](configs/det_yolo.yaml))
was committed and pushed in [`83e51cb`](../../commit/83e51cb) before the test split was read, with
the SHA-256 of the one checkpoint allowed to be evaluated;
[`scripts/eval_test_det.py`](scripts/eval_test_det.py) refuses otherwise.

- **Test is far below val.** The same model scores AP@0.5 0.9745, AP@0.5:0.95 0.8249 and a count
  MAE of 1.88 on val (51 images, 4,929 boxes); the test intervals do not reach those values. Test
  is a later sortie of the same flight, held out to measure this kind of shift (ADR 0002), and
  val chose the epoch, the model and the operating point, so its numbers were expected to be
  optimistic; how much of the gap is each is not separated. On test the count error is mostly
  over-counting (+19.7 panels per frame on average). The cause is not established.
- **Tuning pass:** YOLOv8n against YOLOv8s, with and without brightness jitter, three seeds each,
  under a rule declared before the first run. No arm beat the baseline by more than the noise floor
  (0.010 in val AP@0.5:0.95), so the baseline YOLOv8n was kept.
- **Operating point:** 0.25 has the lowest panel-count MAE on val (1.88). It did not transfer: on
  test 0.5 has the lower count error (14.10), which is reported and not adopted.
- **NMS sweep:** on val, count MAE runs from 3.49 at NMS IoU 0.3 to 1.88 at 0.7 and 3.41 at 0.8, but
  without its one 584-panel frame it stays between 1.78 and 1.98 over the whole range. On that
  frame's small panels the predicted boxes of neighbours overlap where the labels do not.

**These numbers are not comparable to a published D2 number**: the published splits are
contaminated and are not used (see [ADR 0002](docs/adr/0002-d2-is-resplit-by-sortie.md)).

## Not built yet

- **1 · Segmentation (D3 PV01):** site-footprint IoU and Dice per rooftop subset, against a naive colour-threshold baseline.
- **3 · Tracking (D5, synthetic):** MOTA, IDF1 and ID switches, and how many unique modules remain after deduplicating detections across frames.
- **5 · Triage and SQL:** ordinal triage of findings and an SQL layer over the analytics. The regression and per-inverter half of module 5 is built (above).

---

## Scope: what this is and what it is not

What has finished is module 4, module 2 and the regression half of module 5. A small CNN written from scratch and an ImageNet ResNet-18 were
trained on D1 thermal crops, on a split rebuilt after deduplication, and the ResNet-18
config was evaluated once on the held-out test split. Calibration, a base-rate correction
and a distillation experiment followed, on val or on the saved outputs of that one test
run. D2 and D3 were downloaded and prepared (D2 deduplicated and re-split, D3 counted per
subset). On D2 a YOLOv8n fine-tune was selected on val and evaluated once on the held-out test
sortie; no model has been trained on D3.

Module 5's regression half ran on D4, real generation data from two plants: a plant-level
fit with its diagnostics, and per-inverter availability and performance rankings.

**It is not** a finished pipeline: segmentation, tracking, triage and SQL have no code yet, so
nothing runs end to end. Its numbers are not comparable to a published benchmark, because D1
ships no official split and D2's published one is not used. It is not a temperature or power-loss
measurement, because the imagery carries no radiometric data.

---

## The from-scratch seam

**From scratch** = I wrote the algorithm and the module composition, using `torch.nn` layers
and NumPy primitives. I did not implement autograd, cuDNN kernels, or LAPACK routines.

Rows marked **planned** belong to modules that are not built yet; for those, the last
column is the planned cross-check.

| Component | Status | Written by me | Library | Cross-checked against |
|---|---|---|---|---|
| IoU, NMS, mAP@0.5 | built | ✓ | none | torchmetrics (mAP@0.5, equal at 101 points on val and test); `torchvision.ops.nms` (NMS), and Ultralytics' NMS on the detector's own boxes (equal counts on all 51 val images); hand-computed (IoU) |
| mAP@0.5:0.95 | built | | torchmetrics | the from-scratch AP averaged over ten IoU thresholds |
| U-Net | planned | ✓ (`nn.Conv2d` etc.) | none | |
| Detector | built | | Ultralytics YOLOv8 | Ultralytics' own mAP (gap measured: unclipped boxes, trapezoid rule) |
| Kalman filter | planned | ✓ | none | analytic 2-step test |
| Assignment | planned | | `scipy.optimize.linear_sum_assignment` | |
| MOTA/IDF1 | planned | | motmetrics | |
| ID switches, unique-module error | planned | ✓ | none | hand-computed toy |
| OLS, normal equations | built | ✓ | linear solve: `scipy.linalg.cho_solve` | `np.linalg.lstsq`, scikit-learn |
| OLS, gradient descent | built | ✓ | none | `np.linalg.lstsq`, scikit-learn |
| Durbin-Watson, VIF | built | ✓ | none | hand-computed, `tests/test_diagnostics.py` |
| Classical and Newey-West covariance | built | ✓ | `scipy.linalg.cho_solve` for (AᵀA)⁻¹ | hand-computed; statsmodels `HAC` and `hac-panel`, to 1e-10 |
| Driscoll-Kraay covariance, ACF, Breusch-Pagan | built | | statsmodels | |
| Availability, sister ratio, zero-output stretches | built | ✓ | pandas grouping | hand-computed toy frames, `tests/test_inverters.py` |
| Spearman rank correlation | built | | `scipy.stats.spearmanr` | |
| D1 duplicate search + components | built | ✓ | none | SHA-256 byte equality (a subset of it) |
| Macro-F1, per-class recall | built | ✓ | none | `sklearn.metrics.f1_score`, every run |
| Focal loss, KD loss, class weights | built | ✓ | `F.kl_div`, `F.cross_entropy` | hand-computed, `tests/test_cls_metrics.py` |
| ECE, temperature scaling | built | ✓ | `scipy.optimize.minimize_scalar` fits T | hand-computed, `tests/test_cls_metrics.py` |
| ResNet-18 backbone | built | | torchvision, ImageNet weights | |

---

## There is no temperature in the imagery

Neither D1 nor D2 carries radiometric data; both are 8-bit rendered imagery.
The field is `delta_dn_uncalibrated` (hotspot digital number minus panel median digital
number), computed **within a single frame only**, never across frames.

The field exists in the `Finding` contract
([`src/solar_inspect/contract.py`](src/solar_inspect/contract.py)), but no module computes
it yet. [`docs/DATA.md`](docs/DATA.md) explains why the imagery cannot give a temperature. D4's
module and ambient temperatures are a different thing: readings from each plant's weather
sensor, used as such in module 5, and they say nothing about the imagery.

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
which only exists once step 2 has run. See
[ADR 0002](docs/adr/0002-d2-is-resplit-by-sortie.md). D1 ships no split at all, and
splitting it before deduplicating leaves a near-identical pair straddling train and test.
See [ADR 0003](docs/adr/0003-d1-is-deduplicated-at-4-dn-by-component.md). Step 0 is not
optional either: without the `--extra-index-url` line in `requirements.txt`, pip resolves
`torch` to the CPU-only wheel and every GPU run here silently runs on the CPU.

---

## Modules

### 4 · Classification and distillation (finished)

Twelve-class defect classification on D1: No-Anomaly plus eleven defect types. A
116k-parameter CNN written from scratch
([`model.py`](src/solar_inspect/classification/model.py)) is compared with an ImageNet
ResNet-18 fine-tuned on single-channel crops
([`resnet.py`](src/solar_inspect/classification/resnet.py)). Three imbalance treatments
(class weights, focal loss, resampling) were ablated over three seeds against a noise floor
of 0.02 declared before any arm ran; none beat plain cross-entropy. The ResNet-18 config was
committed before the test split was read and evaluated once. Temperature scaling, a
base-rate correction to 2.2% field prevalence and a distillation run came after. Every run
is in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), and the walkthrough is in
[`docs/explainers/04-classification.md`](docs/explainers/04-classification.md).

### 2 · Detection (finished)

Panel detection on D2, 640 x 512 thermal frames from one drone flight. YOLOv8n, fine-tuned with
Ultralytics on the training sorties and selected on the val sortie; IoU, greedy NMS, AP at one IoU
threshold (all-point and 101-point) and AP@0.5:0.95 are written from scratch, unit-tested against
hand-computed values and cross-checked against torchvision and torchmetrics. A tuning pass (two
model sizes, brightness jitter on or off, three seeds each) ran under a selection rule declared
before it started; the operating confidence and an NMS IoU sweep were measured on val; the config,
with the hash of the one checkpoint allowed, was pushed before the test sortie was read once.

D2's published splits are not used. 353 published files are only 252 unique images, and 34 of
the 35 published test images are byte-identical to a train or val image. The split here is
deduplicated by content hash and grouped by acquisition sortie: **138 / 51 / 63 images, 10,612 /
4,929 / 3,984 boxes**. See [ADR 0002](docs/adr/0002-d2-is-resplit-by-sortie.md).

`max_det` is set to **3000** in [`configs/det_yolo.yaml`](configs/det_yolo.yaml), not
Ultralytics' default of 300. Eight of the 252 unique frames carry more than 300 boxes
(max 584) and three of those land in the test split, so the default would have silently
truncated 3.2% of frames. Per-image panel-count error, the domain metric, would then
have been wrong with no error message anywhere. 1000 was not enough either: the densest
val frame (584 boxes) returned exactly 1,000 detections with the cap at 1000.

Every run is in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), and the walkthrough is in
[`docs/explainers/02-detection.md`](docs/explainers/02-detection.md).

### 5 · Analytics on plant generation data (regression half finished)

AC power regressed on the weather columns per plant, solved two ways from scratch (normal
equations with a Cholesky solve, and gradient descent) and checked against `lstsq` and
scikit-learn; diagnosed for the assumption this data breaks (autocorrelation), with
Newey-West and Driscoll-Kraay standard errors; then per-inverter availability and
performance rankings under rules committed before any per-inverter number existed, with a
regression-free sister-ratio comparison, robustness, sensitivity and multiple-comparison
checks, and a measured look at the confounders. Every run is in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), and the walkthrough is in
[`docs/explainers/05-analytics.md`](docs/explainers/05-analytics.md).

### Data work already done for the unbuilt modules

**1 · Segmentation, site footprint.** PV01 is 0.1 m RGB rooftop imagery from a single
geography. It is a **site-footprint / array-extent** primitive; it does not compose with
the thermal stages and it cannot normalise thermal statistics.

Results will be reported per rooftop subset, never pooled, because the class balance
across the three subsets differs by a factor of ten:

| Subset | Masks | Positive pixels | A null all-background model scores |
|---|---:|---:|---:|
| Rooftop_Brick | 138 | 5.20% | 94.8% pixel accuracy, 0 IoU |
| Rooftop_FlatConcrete | 413 | 33.65% | 66.4% pixel accuracy, 0 IoU |
| Rooftop_SteelTile | 94 | 50.41% | 49.6% pixel accuracy, 0 IoU |
| **Pooled** | 645 | 30.00% | 70.0% pixel accuracy, 0 IoU |

That spread is the argument for per-subset reporting. Pooling hides it, and a pooled
figure would make the null model look uniformly mediocre when on Brick it is deceptively
strong and on SteelTile it is worse than a coin flip, because the positive class is the
majority there.

---

## Tests

```bash
python -m pytest
```

Run on 22 Oct 2026 (Python 3.11.6, torch 2.11.0+cu128): **118 passed**, 7 deprecation
warnings from torch and pycocotools, in 7 s. No test needs a GPU. Three D4 tests read the
real CSVs and are skipped when `data/d4/` has not been downloaded; everything else needs no
dataset. They cover the classification maths, the detection metrics (IoU, NMS, AP at one and
at ten IoU thresholds, panel-count error), the tuning pass's selection rule and the
checkpoint-hash guard on the test evaluation, the analytics
primitives (OLS, covariance, Durbin-Watson, VIF, availability, sister ratio, zero-output
stretches), checked against hand-computed values and against torchvision and statsmodels
where they compute the same definition, and the `Finding` contract's mask encoding. The
torchmetrics cross-check of AP runs in `scripts/crosscheck_ap.py` and in every evaluation.

---

## What I didn't do

- Sweep the distillation settings. KD ran at one declared α = 0.5 and T = 4; whether it helps elsewhere is not ruled out.
- Run a capacity sweep on the student, which is the experiment that would test why soft targets did not separate Cell from Cell-Multi.
- Ablate the two ResNet-18 adaptations (the 96×96 resize and the summed conv1 weights) or sweep its learning rate.
- Measure latency in a setting where FLOPs dominate (INT8, ONNX Runtime, TensorRT, batch 1 on an embedded target).
- Compute a performance ratio or specific yield (kWh/kWp) for D4. It needs nameplate DC capacity per inverter and plane-of-array irradiance, and D4 has neither, so the per-inverter slope cannot tell a weak inverter from a smaller array, and the DC/AC ratio is not measurable.
- Identify a cause for any underperforming inverter. The confounders are measured as patterns or stated as not measurable; 34 days cannot separate a fault from a season.
- Separate partial output from the main power curve in the slope fits. Plant 2's producing intervals include some well below the line, and they pull its slopes down.
- Detect oriented boxes. Panels are labelled and detected with axis-aligned boxes, so rows that cross the frame diagonally give neighbouring boxes that overlap; the oriented-box formulation (x, y, w, h, θ) is not tried.
- Train a second detector family. Only YOLOv8 (n and s) was fine-tuned; no two-stage or NMS-free (DETR-style) detector was compared.
- Test on more than one flight. D2 is one drone flight on one day with one 640 x 512 thermal camera; the test sortie is a later part of that same flight.
- Use RGB. The detector sees only the rendered thermal frames; no visible-light imagery is fused.
- Write anchor assignment. YOLOv8 is anchor-free (its task-aligned assigner replaced anchor matching), so the historical anchor-matching step was not implemented.
- Build modules 1 and 3, or module 5's triage and SQL layer. See "Not built yet" above.

---

## Licence

AGPL-3.0. Ultralytics YOLOv8 is AGPL-3.0, and their licence terms require publicly
releasing the complete corresponding source of the entire derivative work, so this repo
is AGPL-3.0 too. Shipping something like this commercially, you would want an enterprise
licence or a differently-licensed detector, which is one reason the metrics here are
written by hand rather than pulled in from more of the stack.
