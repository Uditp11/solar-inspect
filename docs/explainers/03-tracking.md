# Explainer 03: Tracking

Module 3 of the pipeline. Multi-object tracking of PV modules across frames, so that a module seen in
many frames is counted, and reported, once. A box Kalman filter, SORT and two baselines are written
from scratch; they run on synthetic sequences (D5) built from real D2 frames, with the shipped
detector's real detections.

Every number here traces to `docs/EXPERIMENTS.md` and a commit. The metric definitions, the rule
that chose each tracker's settings and every cell of the held-out sweep were committed in `91c597e`
before any metric number existed.

---

## What it is

Three trackers, one interface: per frame, a set of detection boxes in; per detection, a track id out
(or -1 while its track is not yet reported).

- **SORT** (Bewley et al. 2016): a constant-velocity Kalman filter per track, minimum-cost assignment
  of detections to predicted boxes on 1 - IoU, an IoU gate, and SORT's track management.
- **Greedy IoU**: SORT's track management with no motion model. A track's prediction is its last
  matched box, and pairs are taken greedily by highest IoU. It is the baseline that says whether the
  Kalman filter earns its place.
- **No tracker**: every detection is a new identity. The count-error floor.

The two halves of the seam:

- **Written from scratch, unit-tested against hand-computed values:** the Kalman filter (predict,
  Joseph-form update, area clamp, box conversions), SORT and greedy IoU (matching, gate, track
  management), the D5 generator, the ignore rule, the CLEAR MOT matching and ID-switch count, and the
  unique-module count error.
- **Library:** `scipy.optimize.linear_sum_assignment` solves the assignment; motmetrics computes
  MOTA, IDF1, ID switches, FP and FN; `cv2.warpPerspective` renders the D5 frames; the detector is the
  module 2 YOLOv8n checkpoint, unchanged.

**MOTA and IDF1 here are correctness checks of the implementation on synthetic data, never a
performance result.** D5 is a window moving over one still frame; the numbers say whether the
trackers and the metric code behave as designed, not how a tracker would do on a real flyover.

**The data is 8-bit rendered imagery, not radiometric.** Nothing in this module reads a temperature.

---

## Why this stage exists in a PV inspection pipeline

A drone or UAV survey sees each module in several consecutive frames, and a ground robot driving a
row sees it for longer still. Detection and classification work frame by frame, so without tracking
one module with one defect becomes as many findings as frames it appears in: the defect count, the
module inventory and every per-module statistic are inflated by the overlap between frames. Tracking
turns per-frame boxes into per-module identities. That is what makes three things possible:

- **Deduplication.** One module, one finding. The domain number is the count: how many distinct
  modules did the survey see, against how many there are.
- **Temporal voting.** A module classified in eight frames can be reported with the majority, or
  the confidence-weighted, class of those eight rather than whichever frame happened to be kept.
- **Persistence through occlusion.** A module hidden for a few frames (a shadow, a structure, the
  edge of the frame) should keep its identity when it comes back, not start a new one.

So the stage is measured by the unique-module count first ("N frames, M detections, K ids, vs G true
modules"), and by the tracking metrics as checks of the implementation.

---

## How it works

### The Kalman filter as Bayesian inference under Gaussian assumptions

The filter keeps a belief about each box's state as a Gaussian: a mean x and a covariance P. Each
frame does two things.

- **Predict: the prior.** The motion model F pushes the belief forward one step and the process
  noise Q widens it: x⁻ = F x, P⁻ = F P Fᵀ + Q. This is the prior for the new frame, the belief
  before its measurement is seen.
- **Update: the posterior.** A detection z is the measurement. Under the model z = H x + noise with
  covariance R, it defines the likelihood p(z | x) = N(H x, R). Prior times likelihood, normalised, is
  the posterior, and for a linear model with Gaussian prior and noise it is Gaussian again, in closed
  form: x = x⁻ + K (z - H x⁻), P = (I - K H) P⁻, with the gain K = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹.

The gain is where the two are weighed against each other by their covariances. H P⁻ Hᵀ is how
uncertain the prediction is about what will be measured; R is how uncertain the measurement is. When
R is small next to the prediction's uncertainty, K moves the estimate almost all the way to z; when R
is large, K is small and the prediction stands. The tests check both limits: R -> 0 puts the
posterior on the measurement, R -> infinity leaves it on the prior. The update is computed in Joseph
form, (I - K H) P⁻ (I - K H)ᵀ + K R Kᵀ, which is the same covariance and stays symmetric and positive
semi-definite under rounding.

**SORT's state** is x = [u, v, s, r, u', v', s']: box centre (u, v), area s, aspect ratio r = w/h,
and the velocities of u, v and s. The measurement is z = [u, v, s, r]. **r has no velocity**: an
object's aspect ratio is modelled as constant, so only measurements move it. Area has a velocity
because the object's apparent size changes as the camera's distance changes.

**P, Q and R are SORT's**, in SORT's units (px, px²), not retuned:

| matrix | position terms (u, v, s, r) | velocity terms | why |
|---|---|---|---|
| P₀ | 10 | 10,000 | the first box is a measurement; velocities are unobserved at birth, so early updates set them |
| Q | 1 | 0.01 (u', v'), 1e-4 (s') | the area velocity compounds into s every frame, so it may drift least |
| R | 1 (u, v), 10 (s, r) | | a detector's centre is its most reliable part; area and aspect wobble more |

SORT's code builds these as `P[4:,4:] *= 1000; P *= 10`, `Q[-1,-1] *= 0.01; Q[4:,4:] *= 0.01` and
`R[2:,2:] *= 10`, which is where 10,000, 1e-4 and the rest come from.

**The area clamp.** If the area velocity would carry s to zero or below in the next step, it is
zeroed first, as in SORT. Otherwise a shrinking box that coasts without measurements reaches a
negative area and a NaN width.

**The analytic test, and what "noiseless" means.** A box moving in a straight line at constant
velocity, measured exactly, with Q = 0: after two updates the filter must hold the true position and
velocity to 1e-9, and its next prediction must land on the next box. The track starts SORT's way,
from the first box with velocity 0 and SORT's P₀. One update is not enough, and the test asserts
that too: the filter still believes the first box had variance 10, so it splits the first innovation
between "the first box was off" and "the box moved" in the ratio 10 : 10,000, and the velocity comes
out at 10,000 / 10,010 of the truth. The second update has an exact previous position and an
innovation that is exactly the velocity error, so it fixes it. "Exact measurements" has to be told to
the filter, and the way to tell it is R -> 0: with SORT's R the filter would, correctly, still weigh
the prior. R exactly 0 does not work either: after one exact update the variance of r (no velocity)
is 0, Q = 0 keeps it there, and the innovation covariance becomes singular. So the test uses
R = 1e-12 I.

### Assignment: Hungarian against greedy, and where the gate goes

Each frame, every track is predicted, and IoU is computed between every detection and every
predicted box. The matrix is **rectangular**: there are rarely as many detections as tracks.

- **SORT** solves the minimum-cost one-to-one assignment on 1 - IoU with
  `scipy.optimize.linear_sum_assignment`, which accepts a rectangular matrix and leaves the surplus
  rows or columns unassigned. It maximises total IoU over the whole frame.
- **Greedy IoU** repeatedly takes the highest remaining IoU whose detection and track are both free.
  It can lose total IoU: with d0-t0 0.6, d0-t1 0.5, d1-t0 0.5 and d1-t1 0, greedy takes d0-t0 first
  and leaves d1 with nothing, a total of 0.6, where the assignment takes d0-t1 + d1-t0, 1.0.

**The gate is applied after assignment**: any assigned pair with IoU below the gate is dropped and
both sides become unmatched. Folding the gate into the cost, as an infinite cost below it, would make
a real but poor pair indistinguishable from the padding of a rectangular matrix, and would let the
solver reshuffle other pairs to avoid it.

**SORT's unambiguous-match shortcut was not reproduced.** SORT's reference code skips the assignment
when every detection and every track has at most one partner above the gate, and takes those pairs
directly. Plain assignment followed by the gate can differ from it. With IoU d0-t0 0.35, d0-t1 0.29,
d1-t0 0.29, d1-t1 0 and a gate of 0.3, the assignment prefers d0-t1 + d1-t0 (0.58) to d0-t0 (0.35),
and the gate then drops both pairs, leaving nothing matched; the shortcut would see one pair above
the gate in each row and column and keep d0-t0. The construction needs two pairs just under the gate to
outweigh one just over it, so it is rare, but the two are not the same algorithm. The implementation
here is assignment then gate, as the method is described.

### Track management

The same for SORT and greedy IoU, as in SORT's code:

- **Birth.** A detection that no track takes starts a new track. Ids count from 1.
- **`min_hits`.** A track's hit streak counts consecutive matched frames, its birth not included. It
  is reported once the streak reaches `min_hits` (and at any time in the first `min_hits` frames of
  the sequence). Until then its detections get id -1. A single missed frame resets the streak.
  This is what suppresses one-frame false positives: they never get an id.
- **`max_age`.** A track unmatched for more than `max_age` frames is deleted. While it survives, it
  keeps predicting (coasting) and can be matched again with its old id.

`max_age` is the parameter an occlusion needs. A module under a shadow for ten frames returns with
its old identity only if its track survives ten frames without a measurement, and only if the
coasting prediction still overlaps the module when it reappears. With `max_age` 1 every occlusion
longer than a frame becomes a new id, and the module is counted twice.

### D5: a synthetic flyover built from real frames

**Why it is synthetic.** D2's frames do not overlap: each is a separate view, and no two show the same
modules from neighbouring positions, so no real flyover can be rebuilt from them. D5 builds the next
best thing.
Each sequence is a 320 x 256 window moving for 60 frames over one 640 x 512 D2 val frame. Every
labelled panel keeps its identity in every frame, so the ground truth is exact.

**What keeps it from being degenerate:**

- **Motion with acceleration.** The window's velocity is a random walk (white acceleration, 0.25 px
  per frame² per axis), starting at the pan speed (4 px per frame by default) in a seeded direction.
  The trackers' constant-velocity model is deliberately mis-specified.
- **Yaw and scale.** The window rotates 0.15 degrees per frame (8.85 degrees over a sequence) and its
  scale ramps across 0.85 to 1.15, an altitude change. Each frame is a similarity transform, written
  as a homography and rendered with `cv2.warpPerspective`. Ground-truth boxes are the axis-aligned
  hulls of the warped label corners, clipped to the window.
- **An occluder.** An opaque band fixed on the ground (a cloud shadow, say), 80 px wide, drifts at 8
  px per frame, so a module is more than half covered for about 10 frames. Over all 30 sequences, 706
  occlusion runs: 84.8% last 5 to 15 frames, 380 exactly 10.
- **Real detections.** The boxes the trackers see are the shipped detector's, run on every rendered
  window (conf 0.25, NMS IoU 0.7, imgsz 320 so the window is seen at the pixel scale the detector was
  trained on). They carry the detector's real errors: on the development sequences 99.7% of visible
  ground-truth boxes have a detection at IoU 0.5 or more, and there are 11,874 detections against 7,320
  visible ground-truth boxes.

Ten development sequences (seeds 0 to 9) and twenty held-out sequences (seeds 100 to 119) use 30
different val frames. Anything may be tuned on the development seeds; the held-out seeds are scored
only in the sweep.

**Visibility and the ignore rule.** A panel is ground truth in a frame when at least 50% of its warped
area is inside the window and not under the band. The detector does not know that rule: it finds
panels at the window edge and partly under the band too. On the development sequences 61.9% of
detections overlap a visible ground-truth box at IoU 0.5, but 88.7% overlap the hull of some labelled
panel in the window. Against visible-only ground truth those real panels would be counted as false
positives. So every labelled panel with any part in the window that fails the visibility rule is an
**ignore box**: a hypothesis whose IoU is below 0.5 with every visible box but at least 0.5 with an
ignore box is dropped before the frame is scored, and ignore boxes are never misses. A hypothesis
that reaches 0.5 with a visible box is kept whatever it does with ignore boxes.

**Bounces, and their confound with pan speed.** The window must stay inside the source frame, so its
centre reflects off the bounds, and each reflection reverses one velocity component in a single frame.
The free range for the centre is a fraction of the source frame, so bounces become frequent as the
pan speed rises: 22, 34, 50 and 65 over the 20 held-out sequences at 2, 4, 6 and 8 px per frame.
Pan speed and velocity reversals therefore move together in the sweep, and the sweep cannot separate
them.

### Metrics

- **Matching (CLEAR MOT, as motmetrics' `MOTAccumulator` does it).** Per frame, a ground-truth box
  first keeps the hypothesis it was last matched to if their IoU is still at least 0.5; the rest are
  paired by minimum-cost assignment on 1 - IoU among pairs with IoU 0.5 or more. motmetrics'
  `iou_matrix` calls `np.asfarray`, which NumPy 2 removed, so the distances are motmetrics'
  `boxiou` with `iou_matrix`'s rule written out (1 - IoU, not allowed above 0.5).
- **MOTA** = 1 - (FN + FP + IDSW) / (ground-truth boxes). It is dominated by detection errors: an ID
  switch costs the same as one missed or false box, however long the wrong identity then persists, and
  a tracker that switches once and keeps the wrong id for the rest of the sequence pays once. It is not bounded below (no tracker scores -0.110 at
  defaults), and it says little about how long identities are held.
- **IDF1** matches whole ground-truth trajectories to whole track trajectories one-to-one over the
  sequence and takes the F1 of identity-correct detections. It measures association, and it can
  prefer a tracker that detects less if what it does detect it keeps under one id; it is not monotonic
  in detection quality.
- **HOTA** (Luiten et al., IJCV 2020) splits the two: at each localisation threshold α, DetA_α
  measures detection and AssA_α association, HOTA_α = √(DetA_α · AssA_α), the geometric mean, and HOTA
  averages it over α from 0.05 to 0.95. Neither part can hide the other, and the score degrades
  smoothly with localisation. It is the headline metric of the MOTChallenge benchmarks for that reason.
  **It is not computed here.**
- **ID switches, hand-written.** The same CLEAR MOT matching written out (scipy does the assignment
  step), then for each ground-truth id a walk over the frames where it is matched: a switch is each
  time its matched track id differs from the last one it was matched to. A gap is not a switch; a
  return to an earlier id is. It equals motmetrics' `num_switches` **on every sequence** of all 61
  development settings and all 48 sweep cell-tracker pairs.
- **Unique-module count error.** K = distinct reported track ids with at least one hypothesis left
  after the ignore rule; G = distinct ground-truth ids visible at least once. Reported as K - G and
  (K - G) / G, and as the dedup line "N frames, M detections, K ids, vs G true modules". It is the
  domain number: positive means modules counted more than once.

### Settings, chosen on the development seeds only

For SORT and greedy IoU separately, every combination of `max_age` {1, 3, 5, 10, 15}, `min_hits`
{1, 3} and `iou_gate` {0.1, 0.3, 0.5} ran on the ten development sequences. The rule, declared before
any run: lowest sum over the seeds of |K - G|, ties to the higher sum of per-seed IDF1, then the
smaller `max_age`. The choices were committed (`configs/tracking.yaml`, `7a5ed7a`) before any held-out
sequence was scored.

---

## Results

**Synthetic sequences: MOTA and IDF1 below are implementation correctness checks, not a performance
result.**

**Settings.** Both trackers chose `max_age` 15, `min_hits` 3, IoU gate 0.1: SORT with summed |K - G|
204 over the development seeds, greedy IoU with 191. Both sit at the grid's largest `max_age`; the
grid was declared and was not extended.

**Headline, held-out seeds 100 to 119, defaults** (pan 4 px per frame, occlusion 10 frames, detector
conf 0.25):

- no tracker: 1,200 frames, 27,518 detections, 21,261 ids, vs 705 true modules
- greedy IoU: 1,200 frames, 27,518 detections, 1,035 ids, vs 705 true modules
- SORT: 1,200 frames, 27,518 detections, 1,103 ids, vs 705 true modules

| tracker | MOTA | IDF1 | ID switches | FP | FN | K | G | (K - G) / G, median [range] over 20 sequences |
|---|---|---|---|---|---|---|---|---|
| no tracker | -0.110 | 0.035 | 17,801 | 2,756 | 156 | 21,261 | 705 | 30.256 [22.652, 48.417] |
| greedy IoU | 0.894 | 0.732 | 355 | 1,187 | 442 | 1,035 | 705 | 0.493 [0.222, 1.750] |
| SORT | 0.890 | 0.725 | 383 | 1,176 | 499 | 1,103 | 705 | 0.552 [0.231, 1.917] |

ID switches are the hand-written count, equal to motmetrics'. The ignore rule dropped 4,359 (greedy),
4,349 (SORT) and 6,257 (no tracker) hypotheses that sat on real panels outside the visibility rule.

Both trackers cut the id count from about 30 times the true modules to about 1.5 times. They still
over-count by about half, and by up to 1.75 to 1.92 times on the worst sequence.

**The sweep, one factor at a time, held-out.** K - G, greedy IoU / SORT:

| factor | values | greedy IoU | SORT |
|---|---|---|---|
| pan speed (px per frame) | 2 / 4 / 6 / 8 | 319 / 330 / 380 / 401 | 317 / 398 / 502 / 568 |
| occlusion (frames) | 5 / 10 / 15 | 318 / 330 / 358 | 382 / 398 / 417 |
| detector confidence | 0.1 / 0.25 / 0.5 | 458 / 330 / 256 | 547 / 398 / 315 |
| `max_age` | 1 / 5 / 15 | 497 / 464 / 330 | 494 / 491 / 398 |
| IoU gate | 0.1 / 0.3 / 0.5 | 330 / 413 / 443 | 398 / 449 / 515 |

G is 705 except along pan speed (628, 705, 767, 790: a faster window covers more panels) and
occlusion (710, 705, 699).

- **Greedy IoU and SORT give the same numbers in no cell.** Greedy IoU's count error is lower in 11 of
  13 distinct cells. SORT's is lower in two, by 2 and 3 ids: pan speed 2 (317 against 319) and
  `max_age` 1 (494 against 497). Greedy's IDF1 is higher in every cell but occlusion 5 (0.713 against
  0.723).
- **The gap grows with pan speed**: SORT minus greedy K - G is -2, 68, 122 and 167 at 2, 4, 6 and 8 px
  per frame, alongside the bounces (22, 34, 50, 65). Speed and velocity reversals are not separated.
- **Detector confidence** moves the count error most among the factors that are not motion:
  (K - G) / G median 0.630 / 0.493 / 0.355 for greedy and 0.747 / 0.552 / 0.438 for SORT at 0.1 /
  0.25 / 0.5, while greedy's FN rises from 426 to 634.
- **Longer memory and a looser gate** both lower the count error, as on the development seeds.
- **Occlusion length** from 5 to 15 frames changes K - G by at most 40 ids for either tracker at
  `max_age` 15.

**Why greedy wins here: one diagnostic, post-hoc.** Measured after the held-out sweep, to explain it;
it changes no setting or number. On ground-truth boxes only, along each panel's visible run, two
predictions of a later box were compared with the true box: "stay" (the last box, what greedy matches
against) and SORT's Kalman filter fed the true boxes so far; one frame ahead, and coasting 5 and 10
frames with no update. All 30 sequences at each pan speed.

| pan speed | median per-frame displacement | k = 1 median IoU, stay / Kalman | k = 1 below the 0.1 gate | k = 10 median IoU, stay / Kalman | k = 10 below the gate, stay / Kalman |
|---|---|---|---|---|---|
| 2 | 2.20 px | 0.891 / 0.938 | none | 0.344 / 0.609 | 11.8% / 6.0% |
| 4 | 3.80 px | 0.824 / 0.908 | none | 0.135 / 0.477 | 40.7% / 18.7% |
| 6 | 5.65 px | 0.761 / 0.867 | none | 0.009 / 0.319 | 71.7% / 34.4% |
| 8 | 7.31 px | 0.700 / 0.803 | none | 0.000 / 0.062 | 73.9% / 52.6% |

The median ground-truth box is about 62 x 37 px at every speed.

- One frame ahead, the per-frame motion is small next to the box: the last box always overlaps the
  next one, and no step at any speed falls below the 0.1 gate for either prediction. At that gate a
  motion model cannot rescue a one-frame association that the last box would lose. The Kalman
  prediction is still closer in median.
- Coasting, the constant-velocity prediction does **not** drift further than a stationary one on
  ground truth: its median IoU is higher at every speed and horizon, and fewer of its predictions fall
  below the gate at 10 frames everywhere. The exception is 5 frames at pan speed 2 and 4, where more
  Kalman predictions fall below the gate (162 against 12, 668 against 233 of about 19,000).
- So on true boxes the Kalman filter predicts better than "stay", and this diagnostic **does not
  explain** the sweep's result. What it leaves untested is the difference between true boxes and the
  detector's: SORT's filter runs on detections, with duplicates and box jitter, and the two trackers
  also differ in the assignment. Neither was measured.

---

## The arguments this stage has to survive

**"The platform knows its pose to centimetres and there is a digital twin of the site: project each
detection onto the twin and read off the module id. Why track?"** Where that works it is the better
answer, and tracking still has three jobs. Temporal voting: each module is seen in several frames,
and a per-frame classification is noisy; the identity, from the twin or from a tracker, is what lets
the views be combined. Robustness: pose degrades (RTK float or lost fix, timing offsets between camera
and positioning, a gimbal angle error), the twin goes stale (modules replaced, rows not built as drawn)
or does not exist (a site surveyed for the first time); image-space association still works then, and
disagreement between the two is itself a check on pose. Sub-module localisation: a twin locates the
module, not the hotspot, cell or junction box within it, which still comes from the image.

**"An image-space constant-velocity filter on a moving platform is a poor motion model."** It is.
The panels do not move; the camera does, so every track's apparent motion is the same ego-motion seen
through the camera, and each track estimates it separately from a handful of noisy boxes. Turns,
accelerations and altitude changes break the constant-velocity assumption for every track at once.
The two standard remedies: compensate the camera motion before associating, estimating a global
transform between frames (image registration, or the platform's odometry) and applying it to every
track's state, as BoT-SORT does; or move the filter off the image and onto the platform pose, so that
the modules become static points in a world frame and the filter tracks the camera. D5's camera has
random-walk acceleration and reflections precisely so that this mis-specification is present.

**"At 5 m/s and one frame per second, a 1 m panel moves 5 m between frames."** Then boxes from
consecutive frames do not overlap at all, IoU is zero for every pair, and IoU association, with or
without a Kalman filter, returns nothing: every detection starts a new track. D5's motion is the
opposite regime: a median 3.8 px per frame against a 62 x 37 px box, so the last box always overlaps
the next one. At low frame rates, or fast platforms, detections have to be moved into the current
frame with the platform motion from odometry or RTK first, and only then associated; or associated
in world coordinates directly.

**"After a long gap, re-identify by appearance."** Appearance re-identification is weak when the
objects look alike, and PV modules are close to identical by design. What can still discriminate:
soiling patterns, the position of the junction box and its thermal signature, a defect pattern
itself, and context (row ends, gaps, neighbours). Appearance features were not used here.

**"Why SORT rather than ByteTrack or OC-SORT?"** SORT is the minimal tracker in which every part (the
filter, the assignment, the management) can be written, tested and measured, and the baseline the
others are defined against. ByteTrack adds a second association round for low-confidence detections,
which recovers objects that the detector scores low during partial occlusion. OC-SORT re-estimates the
Kalman state from observations after an occlusion and adds an observation-based motion cost, which
targets drift during coasting under non-linear motion. On D5 the sweep shows that SORT's motion model
does not beat no motion model at all, so a better motion model was not the first thing to add; neither
was built.

---

## Limits

- **Synthetic sequences from 30 val frames.** Every D5 sequence is a window over one still D2 frame.
  The motion, the occluder and the ground truth are generated; only the frames' content and the
  detections are real. MOTA and IDF1 are correctness checks, not performance.
- **One detector.** The module 2 YOLOv8n checkpoint at conf 0.25 (0.1 and 0.5 in the sweep). D2's val
  frames have none of the steep-row geometry where that detector fails on test, so these sequences do
  not exercise its worst case.
- **Similarity motion only.** Translation, yaw and scale; no perspective, no parallax, no motion blur.
- **Reflections at the window bounds.** Velocity reverses in one frame at every bounce, and the
  number of bounces rises with pan speed (22 to 65 over 20 sequences), so speed and reversals are
  confounded in the sweep.
- **`max_age` at the grid edge.** Both trackers chose the largest `max_age` in the declared grid (15),
  and the sweep's count error was still falling from 5 to 15. A larger value was not tried.
- **HOTA not computed.** MOTA and IDF1 from motmetrics, ID switches and the count error by hand.
- **No appearance features** and **no camera-motion compensation**; no ByteTrack or OC-SORT.
- **Why SORT loses to greedy IoU is not established.** On ground-truth boxes the Kalman prediction is
  the better one; the effect of detector noise and of the assignment difference was not measured.
- **motmetrics' `iou_matrix` replaced** by its `boxiou` with the same rule, because `iou_matrix` fails
  on NumPy 2.
- **Rendered thermal imagery only.** No radiometric data; nothing here is a temperature.
