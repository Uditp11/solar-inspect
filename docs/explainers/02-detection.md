# Explainer 02: Detection

Module 2 of the pipeline. Panel detection on D2, with the metrics written from scratch and the
detector fine-tuned from a library.

Every number here traces to `docs/EXPERIMENTS.md` and a commit. The test split was read once,
with the config committed and pushed in `83e51cb`, and is not read again.

---

## What it is

A detector that takes one 640 x 512 thermal frame from a drone and returns an axis-aligned box
around every PV module in it, with a confidence score. The model is YOLOv8n, pretrained on COCO
and fine-tuned with Ultralytics on D2: one class, `panel`.

The two halves of the seam:

- **Written from scratch, unit-tested against hand-computed values:** IoU, greedy NMS, AP at one
  IoU threshold (all-point and 101-point interpolation), AP@0.5:0.95 as the mean of ten of those,
  and per-image panel-count error.
- **Library:** Ultralytics YOLOv8 (model, training loop, assigner, loss, inference and its own
  NMS); torchmetrics (pycocotools backend) for mAP@0.5:0.95 and as the cross-check of the
  from-scratch AP; `torchvision.ops.nms` as the cross-check of the from-scratch NMS.

The reported model scores **test AP@0.5 0.7552** (101-point, equal to torchmetrics', 95% interval
0.6418 to 0.8438) and a **panel-count MAE of 25.48 panels per frame** at its operating confidence
(19.65 to 31.94), on 63 images and 3,984 boxes. On val the same model scores 0.9745 and 1.88.
That gap is the main result of this module, and the Results section is mostly about it.

**The data is 8-bit rendered imagery, not radiometric.** All three channels of every D2 frame are
byte-identical, a greyscale render. Nothing in this module reads a temperature, and nothing it
outputs has physical units beyond pixels.

**D2's published splits are not used**: 34 of their 35 test images are byte-identical to a train or
val image. The split here is deduplicated and grouped by sortie (ADR 0002), so no number in this
module is comparable to a published number on D2.

---

## Why this stage exists in a PV inspection pipeline

A thermal survey of a solar site is a stream of frames, each holding tens to hundreds of
modules. Every later question is asked per module: which one is hot (classification), is it the
same module seen in the previous frame (tracking), how many modules did the flight actually cover
(inventory). None of those can be asked until something has said where the modules are. Detection
is that something, and its errors propagate: a missed module is a defect nobody is shown, and a
duplicate box is a module counted twice and possibly reported twice.

Two things follow for how this stage is measured.

- **AP is necessary but not enough.** AP summarises ranking quality over every confidence; a
  report is made at one confidence. So the stage also has an **operating confidence** and is
  scored there.
- **The domain metric is the panel count.** An inspection report states how many modules a frame
  or a block holds and how many of them are abnormal. Per-image panel-count error (detections at
  or above the operating confidence, minus ground-truth boxes) says directly whether that number
  is right, and its sign says which way it is wrong. AP can be high while counts are wrong: a
  duplicate box at low score costs AP little and costs the count a whole panel.

---

## How it works

### IoU, in continuous coordinates

Boxes are `[x1, y1, x2, y2]` in continuous pixel coordinates, area `(x2 - x1)(y2 - y1)`, with no
"+1" pixel convention. IoU is intersection area over union area, defined as 0 when the union is 0.

The consequence that matters here: **two boxes that share an edge intersect in a line, which has
zero area, so their IoU is 0.** Panels tiled side by side in a row, boxed tightly, are exactly that
case. It is pinned by a test (`test_boxes_sharing_an_edge_are_zero`) next to hand-computed partial
overlaps such as `[0,0,10,10]` against `[1,0,11,10]`: intersection 90, union 110, IoU 0.818.

### Greedy NMS and its strict `>` convention

A detector emits many overlapping candidate boxes per object; non-maximum suppression keeps one.
Greedy NMS sorts boxes by score, keeps the highest, removes every remaining box whose IoU with it
is **strictly greater** than the threshold, and repeats on what is left. A box at IoU exactly equal
to the threshold survives. That is torchvision's convention, and the from-scratch `nms` follows it
so the two agree box for box; ties in score keep the lower index first (a stable sort).

The hand-built test case, threshold 0.5: five boxes scored 0.90, 0.80, 0.70, 0.85, 0.60. In score
order, box 0 is kept; box 3 is disjoint from it and kept; box 1 overlaps box 0 at 0.818 and goes;
box 2 overlaps box 3 at 0.333 and stays; box 4 overlaps box 0 at 0.6 and goes. Result `[0, 3, 2]`.
A further test runs both implementations on seeded random boxes and requires identical output.

**On real detections**, the from-scratch NMS was run on the exact tensor Ultralytics suppresses: a
wrapper around Ultralytics' `non_max_suppression` copied each batch's raw head output during one
val pass at conf 0.25 (letterboxed, not clipped, 40,488 candidates over 51 images), and the
from-scratch `nms` was applied to it. Its per-image counts equal Ultralytics' on **all 51 images at
NMS IoU 0.3, 0.5 and 0.7**. (IoU does not change under the letterbox's uniform scale and shift, so
suppression in letterboxed coordinates is suppression in the frame.)

### Where NMS fails, measured

The tidy argument says that, since neighbouring panels share an edge and edge-sharing boxes have
IoU 0, NMS cannot merge two real panels and the threshold hardly matters. There are four real ways
it fails:

1. **Orientation, not density.** When rows cross the frame diagonally, or the view is oblique, the
   axis-aligned box of a tilted panel covers part of its neighbours, and neighbouring boxes overlap
   for real. This is why the solar-mapping literature moved to oriented boxes `(x, y, w, h, θ)`.
2. **Granularity.** If a model boxes a whole table of modules and also each module, the two boxes
   overlap heavily and NMS keeps one. That is a class-definition problem, not a threshold problem.
3. **`max_det` truncation.** NMS output is cut to a maximum number of boxes per image. Ultralytics
   defaults to 300. D2 has frames with 584 panels (next section).
4. **NMS may not exist.** DETR-family detectors predict a set directly and have no NMS step at all.

The first one is visible in D2's labels. The nearest-neighbour IoU of each ground-truth box (its
largest IoU with another box in the frame) has a median of **0.222 in train**, **0.048 in val** and
**0.148 in test**; 128 of the 138 train frames have a median above 0.2, 14 of the 51 val frames and
27 of the 63 test frames. D2 frames often show rows that run diagonally, and the boxes of adjacent
panels overlap by a fifth of their area. Val, the split that selected everything, is the one where
this happens least.

**The sweep.** At the operating confidence (0.25), one Ultralytics val pass per NMS IoU threshold
from 0.30 to 0.80 in steps of 0.05, panel-count error per image:

| NMS IoU (val, 51 images, 4,929 boxes) | count MAE | signed mean | count MAE without the 584-box frame | that frame's signed error |
|---|---|---|---|---|
| 0.30 | 3.490 | -3.216 | 1.94 | -81 |
| 0.50 | 2.922 | -2.373 | 1.82 | -58 |
| 0.70 | **1.882** | -0.980 | 1.82 | -5 |
| 0.75 | 2.176 | +0.020 | 1.78 | +22 |
| 0.80 | 3.412 | +1.529 | 1.98 | +75 |

On 50 of the 51 frames the tidy argument holds: without the densest frame, count MAE stays between
1.78 and 1.98 over the whole range. The whole shape of the curve is one frame of 584 panels whose
boxes are about half the usual size (median 25 x 12 px against 54 x 31 px). Its labelled boxes
barely overlap (nearest-neighbour IoU median 0.059), but its predicted boxes do: of the boxes kept at
0.70, 29.5% overlap another kept box at IoU above 0.3, against 2.0% on the other frames. On small,
dense panels a few pixels of localisation error is a large fraction of the box, so a low threshold
suppresses real neighbours (81 panels short at 0.30) and a high one keeps duplicates (75 over at
0.80). The config's `iou` stayed at 0.7, as declared before the sweep: it was a measurement, not a
selection.

### `max_det`, measured three times

Ultralytics keeps at most `max_det` boxes per image after NMS. Eight of D2's 252 unique frames carry
more than 300 panels (maximum 584), three of them in the test split, so the default of 300 would
have silently cut those frames and made their panel counts wrong with no error anywhere.

- **300** (default): binding on 8 frames by construction.
- **1000**: still binding. At conf 0.001 the 584-panel val frame returned exactly 1,000 detections,
  the lowest at score 0.00388, so its low-score tail was cut.
- **3000**: that frame returns 1,257 detections down to the conf floor of 0.001, and no val frame
  reaches the cap. Val AP@0.5 moved from 0.973185 to 0.974546 (13 more true positives at IoU 0.5);
  counts at conf 0.1 and above did not change. The single test pass returned at most 1,504 on one
  frame, also below the cap.

### AP: global accumulation, the matching rule, the envelope

AP for one IoU threshold is computed in four steps.

1. **Pool and sort globally.** Detections from every image are pooled and sorted by score,
   descending. Accumulating true and false positives per image and then averaging gives a
   different, wrong number: AP ranks detections against each other across the whole split.
2. **Match.** In that order, each detection is matched to the **highest-IoU ground truth in its own
   image that is still unmatched**, if that IoU is at least the threshold; otherwise it is a false
   positive. A second detection on an already matched panel is therefore a false positive. That is
   COCO's rule; VOC's devkit takes the best ground truth overall and calls the detection a false
   positive if that one is taken (a test pins the difference). On an exact IoU tie between two
   unmatched ground truths the lower index wins; pycocotools keeps the last.
3. **Envelope.** Precision and recall are taken at every rank, and precision is replaced by its
   envelope, the running maximum from the right, so it never increases with recall.
4. **Integrate**, two ways. **All-point** (VOC 2010 and later): the sum over recall steps of the step
   width times the envelope there, the exact area under the envelope. **101-point** (COCO): the mean
   of the envelope at the first rank whose recall reaches r, for r = 0, 0.01, ..., 1, counting 0
   where recall never gets there. 101-point is the headline because it is the definition the library
   cross-check computes.

The hand-worked toy in `tests/test_ap.py`: two images, three ground truths, four detections scored
0.9, 0.8, 0.7, 0.6 with outcomes TP, FP, TP, FP. Precision 1, 1/2, 2/3, 1/2; recall 1/3, 1/3, 2/3,
2/3; envelope 1, 2/3, 2/3, 1/2. All-point AP is 1/3 x 1 + 1/3 x 2/3 = **5/9**; 101-point is
(34 x 1 + 33 x 2/3) / 101 = **56/101**. Same detections, two numbers.

**AP@0.5:0.95** is the 101-point AP at each of ten IoU thresholds, 0.50 to 0.95 in steps of 0.05,
averaged, with each threshold matched afresh: a detection can be a true positive at 0.5 and a false
positive at 0.75. It rewards tight boxes, which AP@0.5 does not.

### The torchmetrics cross-check, and its two traps

torchmetrics' `MeanAveragePrecision` (pycocotools underneath) is the library reference. On seeded
synthetic frames (40 frames per seed, up to 600 randomly placed, mutually overlapping boxes), the
from-scratch 101-point AP@0.5 differed from it by at most 1.8e-4. Removing three input differences
(order of tied scores, pycocotools' float32 box rounding, and the IoU-tie rule above) took the
difference to 0 on three seeds and to 4.1e-6 and 3.9e-7 on the other two, where per-image TP counts
agree exactly; that residual is logged as unresolved. On D2's real predictions the two are equal at
AP@0.5 and AP@0.5:0.95, on val and on test.

Both traps were found on this data, not in the documentation:

- **The summary `map` becomes -1.** torchmetrics' default top max-detections setting is 100 per
  image, which would silently score a truncated list on D2. Once it is raised, the summary's `map`
  entry is -1, because pycocotools computes that entry with `maxDets=100` hard-coded and 100 is no
  longer one of the settings. Both APs are therefore read from the per-threshold precision tensor
  (the extended summary), averaging the entries that are not -1, as pycocotools' own summary does.
- **Its own per-image cap.** The reference was first written with a fixed cap of 1000. When the
  evaluation's `max_det` went to 3000, the from-scratch AP exceeded torchmetrics' by 1.4e-3 at AP@0.5
  and 1.8e-4 at AP@0.5:0.95: the reference was scoring only the top 1,000 of the 1,257 detections on
  the dense frame. The reference now takes the evaluation's `max_det`, and the difference is 0.

### Why Ultralytics' own AP differs, measured

Ultralytics reports val AP@0.5 0.973270 where the from-scratch 101-point value (equal to
torchmetrics') is 0.973185. Capturing Ultralytics' internal boxes and scoring them with the
from-scratch matcher reproduced both numbers exactly and separated two causes:

| boxes (val, 51 images, 4,929 boxes, `max_det` 1000) | TP at IoU 0.5 | AP@0.5 101-point | AP@0.5 trapezoid |
|---|---|---|---|
| Ultralytics' internal (letterboxed, not clipped) | 4,824 | 0.968584 | **0.973270** (its number) |
| `predictions.json` (clipped to the frame) | 4,836 | **0.973185** (headline) | 0.977917 |

(1) Ultralytics scores boxes that are not clipped to the frame; the saved predictions are clipped,
5,079 of 14,056 of them touch an edge, and edge panels gain IoU, 12 crossing 0.5. (2) Ultralytics
integrates a linearly interpolated 101-point curve with the trapezoid rule, which lands near the
all-point value rather than COCO's step. On these predictions the two effects nearly cancel, so its
number sits 8.5e-5 from the other two by coincidence. The matching rule is not a factor here: its
IoU-greedy rule and the score-greedy one give the same 4,836 true positives on the clipped boxes.

### YOLOv8: anchor-free, and why anchor assignment was not written

YOLOv8 predicts, at every location of three feature maps (strides 8, 16, 32), a class score and the
distances from that point to the four sides of a box. There are no anchor boxes. Which locations
are responsible for a ground-truth box during training is decided by the **task-aligned assigner**:
for each ground truth it scores candidate locations inside the box by
`score^0.5 x IoU^6` (classification score times localisation quality, Ultralytics' alpha 0.5,
beta 6) and takes the top 10. The loss is **binary cross-entropy** on the class scores, **CIoU** on
the boxes, and **distribution focal loss** (DFL) on the side distances, which are predicted as
discrete distributions; weighted 0.5, 7.5 and 1.5 by Ultralytics' defaults. Despite DFL's name, there
is no focal loss on the classification term; focal loss belongs to RetinaNet, and in this project it
was used where the imbalance actually is, in module 4.

Anchor assignment (IoU matching of ground truths to a fixed set of prior boxes, with positive and
negative thresholds) is the step YOLOv8's assigner replaced. Writing it from scratch while
fine-tuning an anchor-free model would have produced code the model never runs, so it was not
written.

### Keeping val and test apart

The failure this guards against has no symptom: if `val:` in the dataset config points at the test
images, Ultralytics evaluates test every epoch and saves `best.pt` on it, and the test number is
then a selection score. The guards:

- `splits()` resolves `configs/d2.yaml` the way Ultralytics does and refuses if train or val
  resolves to, inside, or above the test tree. Every detection script calls it before Ultralytics
  reads anything. Training manifests record train and val paths only.
- `scripts/eval_test_det.py` is the only code that reads test. It refuses unless `det_yolo.yaml` and
  `d2.yaml` are tracked, unmodified and contained in `origin/main`; refuses if a test evaluation
  already exists; and refuses any checkpoint whose SHA-256 differs from the `checkpoint_sha256`
  committed in the config. The checkpoint is not in git, so the hash is what fixes which model was
  chosen before the test number existed. The first fine-tune's `best.pt`, which gives identical val
  predictions, has a different hash and is refused.
- One pass, many views: every metric, and the bootstrap intervals, are computed from the
  predictions that single pass saved. There was no second pass.

### The tuning pass, declared before it ran

Two factors, crossed, three seeds each, twelve 100-epoch runs: `yolov8n` against `yolov8s`, and
brightness jitter (`hsv_v`) 0.4 against 0.0. D2 frames are greyscale, so brightness is the only
colour augmentation that acts, and in rendered thermal imagery brightness is the signal. Scored by
val AP@0.5:0.95 (101-point) of each run's `best.pt`. Written in the ledger and in
`configs/det_tuning.yaml` before the first arm ran:

- **Noise floor:** the largest three-seed standard deviation over the arms, rounded up to a multiple
  of 0.005. Arms closer than that are indistinguishable.
- **Rule:** take the highest-mean arm only if it beats the baseline by more than the floor; change
  each factor only if its single change beats the baseline by more than the floor; finally, if that
  leaves an arm more than the floor below the best, take the best. The selected arm's **seed 0** run
  goes forward, never its best seed, which would be selecting on noise.

| arm (val, 51 images, 4,929 boxes) | mean score (std), three seeds |
|---|---|
| yolov8n, hsv_v 0.4 (baseline) | 0.8291 (0.0043) |
| yolov8n, hsv_v 0.0 | 0.8194 (0.0065) |
| yolov8s, hsv_v 0.4 | 0.8328 (0.0050) |
| yolov8s, hsv_v 0.0 | 0.8293 (0.0069) |

Noise floor **0.010**. The best arm, yolov8s at 0.4, beats the baseline by 0.0037, inside the
floor, so the **baseline was kept** by the rule's first clause; the only pair further apart than the
floor is the best and the worst arm. Eight of the twelve runs were stopped early by patience 20.
The baseline's seed 0 run reproduced the first fine-tune's val predictions box for box.

### The operating confidence

From the selected model's val predictions at conf 0.001, panel-count error at every confidence on
the grid 0.05, 0.10, ..., 0.95, without another inference pass: a box at or above a threshold can
only be suppressed by a higher-scored box, which is also above it, so filtering a low-confidence
pass by score gives exactly the boxes a higher-confidence pass would. That holds while no frame hits
`max_det`, which the script checks. The declared rule takes the lowest count MAE, the lower
confidence on a tie.

| conf (val, 51 images, 4,929 boxes) | count MAE | signed mean | precision | recall |
|---|---|---|---|---|
| 0.10 | 3.843 | +3.294 | 0.9400 | 0.9720 |
| 0.20 | 2.039 | +0.039 | 0.9649 | 0.9653 |
| **0.25** | **1.882** | -0.980 | 0.9703 | 0.9604 |
| 0.30 | 2.431 | -1.647 | 0.9742 | 0.9576 |
| 0.50 | 5.608 | -5.216 | 0.9893 | 0.9359 |

**Operating confidence 0.25.** Below it the count runs over, above it under.

---

## Results

The single test evaluation: one Ultralytics pass over the test sortie at conf 0.001, NMS IoU 0.7,
`max_det` 3000, with the pinned checkpoint. Intervals are 95% percentile bootstrap intervals over
images (the 63 test images resampled with replacement, 2,000 times, seed 0), recomputed from that
pass's saved predictions; images are the unit because boxes within a frame are not independent.

| metric | test, 63 images, 3,984 boxes | 95% interval | val, same model, 51 images, 4,929 boxes |
|---|---|---|---|
| AP@0.5, 101-point (= torchmetrics) | **0.7552** | 0.6418 to 0.8438 | 0.9745 |
| AP@0.5:0.95, 101-point (= torchmetrics) | 0.5578 | 0.4675 to 0.6374 | 0.8249 |
| precision at conf 0.25, IoU 0.5 | 0.6025 | 0.4925 to 0.6872 | 0.9703 |
| recall at conf 0.25, IoU 0.5 | 0.7899 | 0.6969 to 0.8637 | 0.9604 |
| panel-count MAE at conf 0.25 | 25.48 | 19.65 to 31.94 | 1.88 |

Beside the headline, test: AP@0.5 all-point 0.7571; Ultralytics' AP@0.5 0.7562 and AP@0.5:0.95
0.5569, different for the two reasons measured on val above.

**Test is far below val, and the intervals do not reach the val values.** The split was built to
measure a shift of this kind: test is sortie 4, flown 16:34 to 16:53, while val is sortie 2 (15:35 to
15:47) and training is almost all sortie 3 (15:57 to 16:21); sun angle and thermal loading change
across the flight, and holding out a whole sortie tests that (ADR 0002). Val also did three jobs,
choosing the epoch inside each run, the arm across runs and the operating confidence, so its numbers
were expected to be optimistic. The gap is 0.22 in AP@0.5; how much of it is val's optimism and how
much is the difference between sorties is not separated.

**Panel counts.** At 0.25 the test count error is mostly over-counting: a signed mean of +19.7
panels per frame, from -100 to +105. The operating point did not transfer: on test, conf 0.5 has
the lower count MAE (14.10, signed mean -1.24). That is reported and not adopted, because choosing
the confidence on test would turn the test number into a val number. The three test frames with
more than 300 panels: 512 labelled, 456 counted (-56); 418, 462 (+44); 331, 364 (+33).

**What is and is not established about the gap.** Drawn over the frames, the saved boxes show both
failure directions: a frame of diagonal rows with 108 labelled panels and 213 counted boxes, many
of them overlapping, and a frame where 12 of 112 panels are counted. Diagonal rows alone do not
explain it, though: train has more neighbour overlap in its labels than test (median 0.222 against
0.148), so the model has seen that geometry. The cause of the drop is not established here.

---

## Limits

- **63 test images from one sortie of one flight.** Every D2 frame comes from one drone flight on
  one day; the test sortie is its later part. The intervals describe resampling those 63 frames,
  not another site, season or flight.
- **One camera, one geometry.** A drone camera at 640 x 512 looking down on the rows from above. A
  ground robot looking along a row sees panels at a different scale, angle and overlap, and nothing
  here measures that.
- **Axis-aligned boxes.** Rows that cross the frame diagonally make neighbouring boxes overlap in the
  labels themselves; the oriented-box formulation was not tried.
- **Val was used three times**: for epoch selection inside each run, arm selection across runs, and
  the operating confidence. Its numbers are optimistic by an amount that is not separated from the shift between sorties.
- **Not comparable to published D2 numbers**, because the published splits are contaminated and a
  deduplicated, sortie-grouped split is used instead.
- **Rendered thermal imagery only.** No radiometric data and no RGB; nothing here is a temperature.
- **Ultralytics is AGPL-3.0**, and so this repository is. A commercial system would need an
  enterprise licence or a differently licensed detector.
