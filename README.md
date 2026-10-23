# PV Inspection Pipeline: thermal defect detection, tracking and analytics

> **Status (23 Oct 2026):** all five modules are finished: module 4 (classification), module 2
> (detection), module 3 (tracking), module 1 (segmentation) and module 5 (analytics on plant
> generation data, triage and SQL).

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
| 4 · Classification | D1 | triage precision @ 2.2% prevalence, ResNet-18 fine-tuned, threshold chosen on val | 0.429 at recall 0.913, 46.8 flags per 1,000 | 3,007 imgs (test, once) |
| 4 · Classification | D1 | triage precision @ 2.2% prevalence, threshold chosen on test (optimistic), superseded | 0.502 at recall 0.896, 39.3 flags per 1,000 | 3,007 imgs (test) |
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

## Results: module 5, analytics on plant generation data, triage and SQL

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

### Triage and SQL

| Module | Data | Metric | Value | Split size |
|---|---|---|---|---|
| 5 · Triage | D1 | dispatches per 1,000 modules at 2.2% prevalence, threshold chosen on val | **46.8** (CoA 3: 8.87, CoA 2: 37.95) | 3,007 imgs (test, saved logits) |
| 5 · Triage | D1 | work orders, CoA 3 / CoA 2 | 302 / 1,115 of 1,417 | 3,007 imgs (test, saved logits) |
| 5 · Triage | D1 | precision @ 2.2% of the CoA 3 classes: Diode / Diode-Multi / Hot-Spot / Hot-Spot-Multi | 0.598 / 0.840 / 0.194 / 0.258 | 3,007 imgs (test, once) |
| 5 · SQL | D4, D1 findings | queries equal to their pandas equivalents | 5 of 5 | 136,472 generation rows · 44 inverters · 3,007 findings |
| 5 · SQL | D4 | sister ratio on daily energy (SQL) vs per timestamp, Spearman | 0.983 / 0.932 | 22 / 22 inverters |

**Triage** assigns each finding a Class of Abnormality (CoA 1 to 3) **by failure mode**, not
from any pixel value, because the imagery carries no temperature; the table cites the public
IEA-PVPS failure fact sheets per class ([ADR 0006](docs/adr/0006-coa-by-failure-mode.md)). CoA 3
goes to the modes that can be safety-relevant (bypass diode, hot spot), so it is an upper bound
that sends a crew with a radiometric camera to confirm it. Modules are flagged at the triage
threshold chosen on val (module 4's row above), named by their most likely anomaly class, and
ordered by CoA, extent and probability. No energy, power-loss or currency figure is attached
to anything. The findings are rebuilt from the saved logits of module 4's single test pass,
after proving their order against what that pass saved. Two of the four CoA 3 classes have
precision below 0.3 at field prevalence: whether a module is faulty survives the prevalence
shift, which fault it has does not.

**SQL** puts D4, the inverter rankings, the CoA table and the findings into SQLite through
Python's standard library, and runs five hand-written queries from [`sql/`](sql/): daily
aggregates with a median built from window functions and a `HAVING` floor, `RANK()` within
plant, gaps by `LAG()`, a sister comparison on daily energy, and the work-order list by
`ROW_NUMBER()`. Each is checked against its pandas equivalent on a hand-worked fixture and on the
real data; the ranks equal the ranking code's and the work orders equal the triage run's, row
for row. The daily sister comparison is a different statistic from the per-timestamp one, and
the Spearman above is that measured difference. Spatial queries (PostGIS `ST_DWithin`, a survey
self-join) are written in the explainer and **not run**: no dataset here carries coordinates.

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
  over-counting (+19.7 panels per frame on average). Post-hoc, from the saved outputs and the
  labels (no number above changes): the over-count is boxes beside labelled panels, duplicate or
  misplaced (2,045 of 2,076 false positives), not boxes on background (22). The worst frames are
  close views of rows crossing the frame steeply, where the axis-aligned boxes of neighbouring
  panels overlap, and per-frame AP@0.5 follows box aspect ratio (Spearman +0.57). Frames whose
  median box has aspect ratio w/h below 1.1: none of 136 in train, 22 of 63 in test. On those 22,
  test AP@0.5 is 0.2253; on the other 41 it is 0.9223 (val 0.9745). This is consistent with a change
  in row orientation relative to the camera between sorties, not separable from neighbour overlap;
  oriented boxes are the standard remedy and were not tried.
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

## Results: module 3, tracking

Every number below is logged in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) (the D5 sections) with
the commit and run it came from. **D5 is synthetic.** D2's frames do not overlap, so a real flyover
cannot be rebuilt from them: each D5 sequence is a 320 x 256 window moving for 60 frames over one D2
val frame, with random-walk acceleration, yaw, an altitude ramp and a drifting occluder, and the
boxes the trackers see are the module 2 detector's real detections on every rendered window. The
metrics, the settings rule and the sweep were committed before any metric number existed.

Deduplication, held-out sequences (seeds 100 to 119), defaults (pan 4 px per frame, occlusion 10
frames, detector conf 0.25):

- no tracker: **1,200 frames, 27,518 detections, 21,261 ids, vs 705 true modules**
- greedy IoU: **1,200 frames, 27,518 detections, 1,035 ids, vs 705 true modules**
- SORT: **1,200 frames, 27,518 detections, 1,103 ids, vs 705 true modules**

**Implementation correctness check on synthetic sequences, not a performance result:**

| Tracker | MOTA | IDF1 | ID switches | FP | FN | K ids | G true modules | (K − G)/G, median [range] over 20 sequences |
|---|---|---|---|---|---|---|---|---|
| no tracker | -0.110 | 0.035 | 17,801 | 2,756 | 156 | 21,261 | 705 | 30.256 [22.652, 48.417] |
| greedy IoU | 0.894 | 0.732 | 355 | 1,187 | 442 | 1,035 | 705 | 0.493 [0.222, 1.750] |
| SORT | 0.890 | 0.725 | 383 | 1,176 | 499 | 1,103 | 705 | 0.552 [0.231, 1.917] |

- **Settings, chosen on the 10 development sequences only:** both trackers at `max_age` 15,
  `min_hits` 3, IoU gate 0.1, the lowest summed |K − G| in a declared 30-setting grid, committed in
  [`configs/tracking.yaml`](configs/tracking.yaml) before any held-out sequence was scored.
- **Sweep, one factor at a time on the held-out sequences:** greedy IoU, which has no motion model,
  ends with fewer surplus ids than SORT in 11 of 13 cells; SORT is ahead only at pan speed 2 and at
  `max_age` 1, by 2 and 3 ids. The gap grows with pan speed (SORT minus greedy: -2, 68, 122, 167 ids
  at 2, 4, 6, 8 px per frame), and so do the window's velocity reversals at the frame bounds, so the
  two are not separated.
- **Post-hoc motion diagnostic, on ground-truth boxes:** one frame ahead, the last box always
  overlaps the next at the 0.1 gate (median motion 3.8 px against a 62 x 37 px box), and SORT's
  Kalman prediction is closer than the last box in median IoU at every speed, one frame ahead and
  coasting 5 and 10 frames. So it does not explain why greedy wins; the effect of detector noise and
  of the assignment was not measured.

The hand-written ID-switch count equals motmetrics' on every sequence. Hypotheses that sit only on
labelled panels failing the visibility rule (at the window edge or under the occluder) are dropped
before scoring rather than counted as false positives. The walkthrough is in
[`docs/explainers/03-tracking.md`](docs/explainers/03-tracking.md).

## Results: module 1, segmentation

Every number below is logged in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) (the D3 sections) with
the commit and run it came from. PV01 is 0.1 m RGB rooftop imagery from one region: this module is a
**site-footprint / array-extent** primitive, and it does not compose with the thermal stages. Every
number is per rooftop subset, never pooled. Test is 93 tiles, read once. Each interval is a 95%
percentile bootstrap over the subset's test tiles (10,000 resamples, seed 0); the difference is
**paired**, U-Net and colour rule scored on the same resampled tiles. Tiles within a scene are
correlated, so these intervals are narrower than the truth; the number of spatially separate tile
groups is given beside each.

| Subset | U-Net IoU (test, once) | colour rule IoU (test) | U-Net minus colour rule (test, paired) | null model (test): IoU, pixel accuracy | Split size | U-Net IoU (val) |
|---|---|---|---|---|---|---|
| Brick | **0.9101**, 0.8593 to 0.9385 | 0.3068, 0.1876 to 0.4177 | +0.6033, 0.5073 to 0.6977 | 0, 0.9422 | 19 tiles, 8 groups (test, once) | 0.9074 (21 tiles) |
| FlatConcrete | **0.7910**, 0.7350 to 0.8430 | 0.4350, 0.3790 to 0.4895 | +0.3560, 0.3232 to 0.3944 | 0, 0.5609 | 60 tiles, 3 groups (test, once) | 0.9272 (60 tiles) |
| SteelTile | **0.9778**, 0.9620 to 0.9851 | 0.3912, 0.2232 to 0.5849 | +0.5867, 0.3980 to 0.7537 | 0, 0.7453 | 14 tiles, 2 groups (test, once) | 0.9455 (14 tiles) |

U-Net, test, Brick / FlatConcrete / SteelTile: Dice 0.9529 / 0.8833 / 0.9888, precision 0.9601 /
0.9703 / 0.9879, recall 0.9459 / 0.8106 / 0.9897. The U-Net is written from `torch.nn` layers and
trained from scratch (8,633,441 parameters, no pretrained weights); the colour rule is **panel
where 2B - R - G >= 24**, chosen on train; the null model predicts background everywhere. The
evaluation config ([`configs/seg_eval.yaml`](configs/seg_eval.yaml)), with the SHA-256 of the one
checkpoint allowed, was committed in [`49c44d9`](../../commit/49c44d9) before the test split was read;
[`scripts/eval_seg.py`](scripts/eval_seg.py) refuses otherwise.

- **Val**, the selection split (epoch 87, by the mean of the three per-subset val IoUs): the colour
  rule scores 0.3967 / 0.6611 / 0.7313 there, and the paired U-Net lead is +0.5107 / +0.2660 /
  +0.2143, every interval clear of 0. Selection flatters Brick's val IoU by 0.007 against the mean of
  the last ten epochs, and the other two by less than 0.001.
- **FlatConcrete test is below its val (0.7910 against 0.9272), and the drop is recall.** Post-hoc,
  from the saved predictions: one 45-tile roof holds 99.4% of the subset's missed pixels (IoU 0.7808
  there; the other two groups 0.9150 and 0.9393), where large stretches of panel render bright, close
  to white, and are missed. The cause of that appearance is not established. On Brick and SteelTile
  the remaining error is at array edges: 75% and 88% of the error pixels lie within 3 px of a true
  boundary.
- **Pixel accuracy misleads on Brick only.** The null model scores 0.9422 there by finding nothing;
  the colour rule, at IoU 0.3068, scores 0.8950, below it.
- **The panel share shifts between splits** (SteelTile test 25% against 53% in train; FlatConcrete
  val and test 41% and 44% against 30%), so val, train and test numbers are not directly comparable.
- **It does not transfer to thermal.** Run unchanged on D2's 51 thermal val frames, it marks one
  pixel in 16.7 million as panel where ground-truth boxes cover 46%. A control that removes only
  colour, PV01 val tiles converted to greyscale, collapses it as well (val IoU 0.0407 / 0.0466 /
  0.0013 against 0.9074 / 0.9272 / 0.9455 in colour): the model needs colour, which is enough to
  explain the thermal result; resolution, viewpoint and content are not tested separately.

The split is by scene, linked across all three subsets, with SteelTile cut into blocks and a guard
band ([ADR 0005](docs/adr/0005-d3-is-split-by-scene.md)). The walkthrough is in
[`docs/explainers/01-segmentation.md`](docs/explainers/01-segmentation.md).

## Scope: what this is and what it is not

What has finished is module 4, module 2, module 3, module 1 and module 5. A small CNN written from scratch and an ImageNet ResNet-18 were
trained on D1 thermal crops, on a split rebuilt after deduplication, and the ResNet-18
config was evaluated once on the held-out test split. Calibration, a base-rate correction
and a distillation experiment followed, on val or on the saved outputs of that one test
run. D2 was deduplicated and re-split by sortie, and D3 split by scene. On D2 a YOLOv8n fine-tune
was selected on val and evaluated once on the held-out test sortie. On D3 a U-Net written from
scratch was selected on val and evaluated once on the held-out test tiles, against a colour-threshold
baseline and a null model. Module 3's trackers ran on D5, synthetic sequences built
by moving a window over D2 val frames, fed the module 2 detector's real detections.

Module 5 ran on D4, real generation data from two plants: a plant-level fit with its
diagnostics, and per-inverter availability and performance rankings; its triage ran on the
saved outputs of module 4's test pass, and its SQL layer over both.

**It is not** an end-to-end pipeline: each module runs on its own dataset, joined by the `Finding`
contract rather than by one run from a flight's images to a work order, and the segmentation module works on RGB rooftop tiles that no thermal stage shares. The tracking numbers come from synthetic sequences and check the implementation; they are
not a tracking benchmark. Its numbers are not comparable to a published benchmark, because D1
ships no official split and D2's published one is not used. It is not a temperature or power-loss
measurement, because the imagery carries no radiometric data.

---

## The from-scratch seam

**From scratch** = I wrote the algorithm and the module composition, using `torch.nn` layers
and NumPy primitives. I did not implement autograd, cuDNN kernels, or LAPACK routines.

| Component | Status | Written by me | Library | Cross-checked against |
|---|---|---|---|---|
| IoU, NMS, mAP@0.5 | built | ✓ | none | torchmetrics (mAP@0.5, equal at 101 points on val and test); `torchvision.ops.nms` (NMS), and Ultralytics' NMS on the detector's own boxes (equal counts on all 51 val images); hand-computed (IoU) |
| mAP@0.5:0.95 | built | | torchmetrics | the from-scratch AP averaged over ten IoU thresholds |
| U-Net | built | ✓ (`torch.nn` layers, no pretrained weights) | none | hand-counted parameter count of the smallest net, output shapes, each skip used (`tests/test_unet.py`) |
| IoU, Dice, soft Dice loss | built | ✓ | none | hand-computed values; Dice = 2 IoU / (1 + IoU) on 50 random mask pairs (`tests/test_seg_metrics.py`) |
| Colour-threshold baseline | built | ✓ | none | the histogram search against a direct pixel count, equal (asserted in `scripts/baseline_seg.py`) |
| Tile bootstrap, boundary share | built | ✓ | `scipy.ndimage.distance_transform_edt` for the boundary distance | hand-worked cases, `tests/test_seg_evaluate.py` |
| Detector | built | | Ultralytics YOLOv8 | Ultralytics' own mAP (gap measured: unclipped boxes, trapezoid rule) |
| Kalman filter | built | ✓ | none | analytic 2-step test |
| Assignment | built | | `scipy.optimize.linear_sum_assignment` | hand-computed matrices, `tests/test_sort.py` |
| MOTA/IDF1 | built | | motmetrics (`boxiou`, because `iou_matrix` fails on NumPy 2) | hand-computed sequences, `tests/test_track_metrics.py` |
| ID switches, unique-module error | built | ✓ | none | motmetrics' switch count, equal on every sequence; hand-computed sequences |
| Flyover warp (D5) | built | | `cv2.warpPerspective` | analytic visible fractions against a supersampled pixel mask |
| OLS, normal equations | built | ✓ | linear solve: `scipy.linalg.cho_solve` | `np.linalg.lstsq`, scikit-learn |
| OLS, gradient descent | built | ✓ | none | `np.linalg.lstsq`, scikit-learn |
| Durbin-Watson, VIF | built | ✓ | none | hand-computed, `tests/test_diagnostics.py` |
| Classical and Newey-West covariance | built | ✓ | `scipy.linalg.cho_solve` for (AᵀA)⁻¹ | hand-computed; statsmodels `HAC` and `hac-panel`, to 1e-10 |
| Driscoll-Kraay covariance, ACF, Breusch-Pagan | built | | statsmodels | |
| Availability, sister ratio, zero-output stretches | built | ✓ | pandas grouping | hand-computed toy frames, `tests/test_inverters.py` |
| Spearman rank correlation | built | | `scipy.stats.spearmanr` | |
| Triage, CoA by failure mode from public sources (ADR 0006) | built | ✓ | none | hand-built dispatch cases and the ADR's table, `tests/test_triage.py` |
| Triage threshold at field prevalence | built | ✓ | none | hand-computed, `tests/test_cls_metrics.py` |
| SQL: hand-written queries; spatial queries written, not run | built | ✓ | SQLite from Python's standard library (`sqlite3`) | each query against its pandas equivalent, `tests/test_sql.py` and `scripts/sql_d4.py` |
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
([`src/solar_inspect/contract.py`](src/solar_inspect/contract.py)). Module 5's triage fills it
on each D1 crop as the 99th-percentile minus the median digital number of the raw 8-bit crop,
and never ranks on it: D1's crops come from mixed sensors, so two crops are two frames. [`docs/DATA.md`](docs/DATA.md) explains why the imagery cannot give a temperature. D4's
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
python scripts/split_d3.py          # 5. rebuild D3's split; the loader checks its hash
```

**None of them is optional, and 3 comes before 4.** D2's published train/val/test splits
are contaminated and are not used here; `configs/d2.yaml` points at the rebuilt split,
which only exists once step 2 has run. See
[ADR 0002](docs/adr/0002-d2-is-resplit-by-sortie.md). D1 ships no split at all, and
splitting it before deduplicating leaves a near-identical pair straddling train and test.
See [ADR 0003](docs/adr/0003-d1-is-deduplicated-at-4-dn-by-component.md). D3 ships no split
either; step 5 rebuilds the scene split, and the segmentation loader refuses any assignment whose
hash differs from the one pinned in `configs/d3_split.json` (see
[ADR 0005](docs/adr/0005-d3-is-split-by-scene.md)). Step 0 is not
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

### 3 · Tracking (finished)

Multi-object tracking of panels across frames, so that each module is counted once. SORT's
constant-velocity box Kalman filter is written from scratch, starting from an analytic test, with
SORT's noise constants and area clamp; SORT itself (assignment with scipy, an IoU gate after
assignment, SORT's track management), a greedy-IoU tracker with no motion model and a no-tracker
floor are written beside it. They run on D5: synthetic sequences built from D2 val frames by a seeded
generator (moving, rotating and scaling window, drifting occluder), with the module 2 detector's real
detections and exact per-panel identities. MOTA and IDF1 come from motmetrics; the ID-switch count and
the unique-module count error are written by hand and checked against it. Settings were chosen on
development sequences under a rule declared first, and a held-out sweep varies one factor at a time.
Every run is in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), and the walkthrough is in
[`docs/explainers/03-tracking.md`](docs/explainers/03-tracking.md).

### 5 · Analytics on plant generation data, triage and SQL (finished)

AC power regressed on the weather columns per plant, solved two ways from scratch (normal
equations with a Cholesky solve, and gradient descent) and checked against `lstsq` and
scikit-learn; diagnosed for the assumption this data breaks (autocorrelation), with
Newey-West and Driscoll-Kraay standard errors; then per-inverter availability and
performance rankings under rules committed before any per-inverter number existed, with a
regression-free sister-ratio comparison, robustness, sensitivity and multiple-comparison
checks, and a measured look at the confounders. Then triage: a `Triage` record beside
`Finding`, a CoA by failure mode from public sources, and a dispatch rule declared before it
ran; and an SQL layer of five queries on SQLite, each cross-checked against pandas. Every run
is in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), and the walkthrough is in
[`docs/explainers/05-analytics.md`](docs/explainers/05-analytics.md).

### 1 · Segmentation, site footprint (finished)

Binary panel masks on D3 (PV01), 256 x 256 RGB rooftop tiles at 0.1 m from a single geography. It is
a **site-footprint / array-extent** primitive; it does not compose with the thermal stages and it
cannot normalise thermal statistics. A U-Net written from `torch.nn` layers
([`unet.py`](src/solar_inspect/segmentation/unet.py)) is trained from scratch with BCE plus soft
Dice, on a split built by scene and linked across the three rooftop subsets, and selected on val;
IoU, Dice and the soft Dice loss are written from scratch and unit-tested against hand-computed
values. A one-threshold colour rule chosen on train and an all-background null model are scored on
the same tiles. The evaluation config was committed before the test tiles were read once, and a
transfer check on D2 thermal frames was declared before it ran. Every run is in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), and the walkthrough is in
[`docs/explainers/01-segmentation.md`](docs/explainers/01-segmentation.md).

Results are reported per rooftop subset, never pooled, because the class balance across the
three subsets differs by a factor of ten (over all 645 tiles):

| Subset | Masks | Positive pixels | A null all-background model scores |
|---|---:|---:|---:|
| Rooftop_Brick | 138 | 5.20% | 94.8% pixel accuracy, 0 IoU |
| Rooftop_FlatConcrete | 413 | 33.65% | 66.4% pixel accuracy, 0 IoU |
| Rooftop_SteelTile | 94 | 50.41% | 49.6% pixel accuracy, 0 IoU |
| **Pooled** | 645 | 30.00% | 70.0% pixel accuracy, 0 IoU |

That spread is the argument for per-subset reporting. Pooling hides it, and a pooled
figure would make the null model look uniformly mediocre when on Brick it is deceptively
strong and on SteelTile it is worse than a coin flip, because the positive class is the
majority there. Per split the shares move: SteelTile's test tiles are 25% panel, and the null model
scores 0.7453 on them.

---

## Tests

```bash
python -m pytest
```

Run on 23 Oct 2026 (Python 3.11.6, torch 2.11.0+cu128): **231 passed**, 7 deprecation
warnings from torch and pycocotools, in 9 s. No test needs a GPU. Three D4 tests read the
real CSVs and are skipped when `data/d4/` has not been downloaded; everything else needs no
dataset. They cover the classification maths, the detection metrics (IoU, NMS, AP at one and
at ten IoU thresholds, panel-count error), the tuning pass's selection rule and the
checkpoint-hash guard on the test evaluation, the tracking code (the Kalman filter against an
analytic two-step solution and its two limits, SORT and greedy association and track management, the
D5 generator, the ignore rule, CLEAR MOT matching, ID switches and the count error), the analytics
primitives (OLS, covariance, Durbin-Watson, VIF, availability, sister ratio, zero-output
stretches), the triage (the `Triage` record, every tie-break of the dispatch rule, the CoA table
against ADR 0006, the threshold at field prevalence), the five SQL queries against hand-worked rows
and their pandas equivalents, the segmentation code (IoU, Dice and the soft Dice loss including its two silent
failures without eps, mask binarisation, the augmentation moving tile and mask together, the U-Net's
hand-counted parameters, its skips and its receptive field, per-tile counts, the paired tile bootstrap, the
near-boundary mask and the greyscale conversion), checked against hand-computed values and against torchvision and statsmodels
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
- Compute HOTA. Tracking is scored with MOTA, IDF1, ID switches and the unique-module count; HOTA, which separates detection from association, is described in the explainer and not computed.
- Use appearance features. No tracker here re-identifies a panel by how it looks after a long gap.
- Run ByteTrack or OC-SORT. SORT is the only Kalman tracker built, compared with greedy IoU and no tracker.
- Compensate camera motion. The trackers predict in image space with no estimate of the platform's ego-motion, from the images or from odometry.
- Try `max_age` beyond 15. Both trackers chose the largest value of the declared grid, and it was not extended.
- Track on a real flyover. D2's frames do not overlap, so D5's sequences are synthetic; no real multi-frame thermal sequence was used.
- Compare an ImageNet-pretrained encoder for the U-Net. Module 1 trains from scratch only, so what pretraining would gain on 445 training tiles is not measured.
- Run the spatial queries. No dataset here carries coordinates, so the PostGIS queries in explainer 05 are written and not run, and the triage ranking has no string position or affected-module count.
- Settle whether D1's Cracking class is glass breakage. ADR 0006 keeps it at CoA 2 as the project's assignment; if it is, the fact sheets would put it at CoA 3.

---

## Licence

AGPL-3.0. Ultralytics YOLOv8 is AGPL-3.0, and their licence terms require publicly
releasing the complete corresponding source of the entire derivative work, so this repo
is AGPL-3.0 too. Shipping something like this commercially, you would want an enterprise
licence or a differently-licensed detector, which is one reason the metrics here are
written by hand rather than pulled in from more of the stack.
