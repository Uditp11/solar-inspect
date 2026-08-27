# Experiments

Every training run, in order. A run with `dirty=yes` was made against a
working tree that did not match its commit, so its numbers are not
reproducible from that SHA -- treat them as indicative only.

**`split` is the first 8 hex of `configs/d1_split.json`'s sha256.** D1 was
deduplicated and re-split at `6f197e6`, so a row on `4cbb0c3d` and a row on
`af8781b1` are numbers on two different datasets and comparing them is a
mistake with no visible symptom. The column exists to make that impossible to
do by accident.

## The noise floor, declared before the ablation was run

**Any two ablation arms whose mean val macro-F1 differs by less than 0.02 are
reported as indistinguishable and are not ranked.**

This is written here, and in `configs/cls_ablation.yaml`, **before the first
ablation arm ran**. It comes from ADR 0004's three control arms, which are the
only three-seed standard deviations measured on split `af8781b1`: **0.0116,
0.0187 and 0.0191**. Rounding up over that range gives 0.02.

Two things follow that are easy to walk back from later and so are fixed now.
If every arm lands inside 0.02 of every other, the result is *"none of the three
imbalance treatments is measurably better than plain cross-entropy at this
budget"* -- not a ranking with a small margin, and not a winner. And the
threshold does not move once the numbers are in, in either direction.

The transfer is an assumption worth stating: those control arms ran at constant
LR and took the last epoch, while every ablation arm uses cosine decay and
selects the best val epoch. The per-arm standard deviations below say whether
0.02 turned out generous or tight; they do not get to redefine it.

## What the validation split is doing

By the end of Task 7 `val` carries four jobs at once: **epoch selection** inside
every run, **config selection** between the ablation arms, the **temperature fit**
for calibration, and the **leakage diagnostic** of ADR 0004. Each is a legitimate
use of a validation split and together they are still not a test set -- but four
uses is four chances to overfit it, and the val numbers here should be read as
optimistic by an unmeasured amount. Said here rather than left for a reader to
notice.

## Ablation result — three seeds per arm, val, split `af8781b1`, run at `f3d71fd`

| arm | val macro-F1, mean ± std | seeds | selected epochs | val accuracy |
|---|---:|---|---|---:|
| baseline (plain CE) | **0.5980 ± 0.0085** | 0.6067 · 0.5897 · 0.5975 | 25 · 21 · 27 | 0.784 |
| focal (γ = 2) | 0.5855 ± 0.0046 | 0.5906 · 0.5841 · 0.5818 | 25 · 22 · 30 | 0.776 |
| resampling (balanced) | 0.5804 ± 0.0039 | 0.5849 · 0.5775 · 0.5789 | 25 · 23 · 28 | 0.752 |
| class weights | 0.5615 ± 0.0065 | 0.5541 · 0.5660 · 0.5645 | 26 · 28 · 28 | 0.724 |

Against the **0.02 declared above**, only two of the six pairs separate:

| pair | Δ macro-F1 | called |
|---|---:|---|
| baseline − class weights | +0.0364 | baseline above class weights |
| focal − class weights | +0.0240 | focal above class weights |
| baseline − resampling | +0.0175 | **indistinguishable** |
| class weights − resampling | −0.0189 | **indistinguishable** |
| baseline − focal | +0.0125 | **indistinguishable** |
| focal − resampling | +0.0051 | **indistinguishable** |

**The result is that none of the three imbalance treatments beats plain
cross-entropy, and class weighting is measurably worse than two of the arms.**
The three arms that are supposed to help either did not, or did not by enough to
see at this budget. That is the finding. `baseline` has the highest mean and is
**not** thereby the winner: it is inside the floor of both focal and resampling.

Two things this table is doing that are worth pointing at rather than hiding:

**The ordering is not a total order, and that is correct.** Baseline beats class
weights; resampling does not; baseline does not beat resampling. A noise floor
applied honestly produces exactly this — comparisons that resolve and comparisons
that do not, in the same table. Forcing them into a ranking is the thing the
floor exists to prevent.

**The measured per-arm std, 0.004–0.009, is tighter than the 0.012–0.019 the
floor was built from.** The floor still stands at 0.02 for these runs, because it
was declared for these runs; a threshold re-derived from the numbers it is about
to judge is not a threshold. What the tighter spread means is that cosine decay
plus best-epoch selection stabilised the runs relative to the constant-LR control
arms, and a *future* ablation on this budget could pre-declare something nearer
0.015. It does not license reopening `baseline − focal` at +0.0125.

### Where the treatments did act: the four smallest val classes

Mean F1 over the three seeds. Supports are 26–37 images, so a single misclassified
crop moves a cell by 2–4 points — read the direction, not the digits.

| arm | Diode-Multi (26) | Soiling (30) | Hot-Spot-Multi (36) | Hot-Spot (37) |
|---|---:|---:|---:|---:|
| baseline | 0.7443 | 0.1554 | **0.4594** | 0.4365 |
| class weights | 0.7239 | 0.2041 | 0.3789 | 0.2980 |
| focal | 0.7249 | 0.1644 | 0.4008 | **0.4397** |
| resampling | **0.7703** | **0.2151** | 0.3514 | 0.3450 |

The treatments are not inert — they trade *within* the tail rather than lifting
it. Resampling and class weighting both improve Soiling, the worst class in the
project, by 5–6 points, and both lose 8–14 points on Hot-Spot-Multi and Hot-Spot.
Macro-F1 averages that trade to roughly nothing. The honest reading is that at
57:1, reweighting moves which rare class the model is willing to over-predict,
and no arm here made the tail as a whole easier.

### Cosine decay and best-epoch selection are worth about the noise floor

The ablation's baseline arm is plain cross-entropy on the same data as
`configs/cls_baseline.yaml`, differing only in cosine decay and selecting the
best val epoch instead of the 30th: **0.5980 ± 0.0085 against 0.5799** on seed 0.
That gap is about one noise floor wide and is a change in the *stopping rule*,
not in the model — which is exactly why the budget was fixed identically across
arms before any arm ran.

## Transfer-learning baseline — ImageNet ResNet-18, val, run at `0d15ffe`

| arm | val macro-F1, mean ± std | seeds | selected epochs | val accuracy | params |
|---|---:|---|---|---:|---:|
| resnet18, fine-tuned | **0.6770 ± 0.0074** | 0.6735 · 0.6855 · 0.6719 | 13 · 28 · 17 | 0.832 | 11,176,396 |

Against the best from-scratch arm (0.5980 ± 0.0085) that is **+0.079 macro-F1,
four times the declared floor** — the one comparison in Task 7 that is not close.
It is also a real teacher–student gap, which is what makes the distillation in
`configs/cls_kd.yaml` a measurement rather than a ceremony. On the four smallest
val classes it is ahead everywhere: Diode-Multi 0.785, Soiling 0.266,
Hot-Spot-Multi 0.541, Hot-Spot 0.562, against the small CNN's best of 0.770 /
0.215 / 0.459 / 0.440.

The two adaptations it needs — the distorting 96×96 resize, and summing conv1's
weights across the input channel rather than replicating the grey channel — are
argued in `src/solar_inspect/classification/resnet.py` beside the code that does
them, each with the alternative that was not taken. Neither is ablated. Its LR is
3e-4 rather than the 3e-3 the from-scratch CNN uses, taken as the standard
fine-tuning default and not swept.

## THE TEST EVALUATION — once, and not again

**Config commit: [`c1df507`](../../../commit/c1df507).** `configs/cls_final.yaml`
was committed and pushed to `origin/main` before `scripts/eval_test_d1.py` read
the test split, and the script refuses to run otherwise — it checks that the file
is tracked, unmodified, and an ancestor of `origin/main`, and prints the SHA it
found. Run `20260827T205903Z`, split `af8781b1`, seed 0 fixed in the config.

| | macro-F1 | accuracy | null model (always No-Anomaly) |
|---|---:|---:|---:|
| **test, 3,007 images** | **0.6956** | 0.8251 | 0.4988 |
| val, same model, for reference | 0.6735 | 0.8310 | 0.5013 |

Per class, with the support beside every rate — 27 images is not 1,500 and a
recall computed on it does not mean the same thing:

| class | support | recall | precision | F1 |
|---|---:|---:|---:|---:|
| No-Anomaly | 1,500 | 0.9753 | 0.9076 | 0.9402 |
| Diode | 225 | 0.9378 | 0.9505 | 0.9441 |
| Diode-Multi | 27 | 0.7778 | 0.8400 | 0.8077 |
| Offline-Module | 124 | 0.6613 | 0.8200 | 0.7321 |
| Shadowing | 158 | 0.6519 | 0.8047 | 0.7203 |
| Vegetation | 247 | 0.7085 | 0.7202 | 0.7143 |
| Hot-Spot | 38 | 0.5789 | 0.8462 | 0.6875 |
| Cell | 282 | 0.7270 | 0.6231 | 0.6710 |
| Cracking | 142 | 0.6338 | 0.6923 | 0.6618 |
| Hot-Spot-Multi | 38 | 0.4737 | 0.6923 | 0.5625 |
| Cell-Multi | 194 | 0.4072 | 0.5524 | 0.4688 |
| Soiling | 32 | 0.3750 | 0.5217 | 0.4364 |

The confusion matrix is in `runs/cls_final_test_20260827T205903Z/` as both a PNG
and the raw counts in `manifest.json`. Its three largest off-diagonal cells:

- **Cell-Multi → Cell, 51 of 194.** The largest single error in the matrix, and
  the same pair Task 6's small CNN got wrong most often. Distinguishing one hot
  cell from several is a counting problem at 40×24, and the model is not doing it.
- **Shadowing → No-Anomaly, 37 of 158**, and **Offline-Module → No-Anomaly, 36 of
  124.** Both are misses of a real defect, which is the expensive direction: an
  offline module called normal is a module nobody is sent to look at.
- **Vegetation → Cell, 32 of 247.**

### The leaky/clean subgroup on test, from this same run's predictions

Computed from the per-image predictions of the one test pass, not from a second
evaluation. Class-matched, exactly as ADR 0004 does it on val.

| | n | leaky acc | clean, class-matched | difference | z |
|---|---:|---:|---:|---:|---:|
| **test, final ResNet-18** | 133 / 2,874 | 0.9098 ± 0.0248 | 0.8893 ± 0.0076 | **+0.0204 ± 0.0260** | **+0.79** |
| val, same ResNet-18 | 129 / 2,859 | 0.9612 ± 0.0170 | 0.9282 ± 0.0068 | +0.0331 ± 0.0183 | +1.80 |
| val, the small CNN (ADR 0004 §A) | 129 / 2,859 | 0.9535 ± 0.0187 | 0.8484 ± 0.0097 | +0.105 ± 0.021 | +5.09 |

All three rows are one model each, with binomial standard errors, so they are
comparable in kind. The small-CNN row is the single-seed measurement, not the
three-seed control table — those ± are seed spreads and mean something else.

**This qualifies ADR 0004's bound, and in the direction opposite to the one that
ADR predicted.** On test the leaky advantage is +0.020 and inside its own error
bar. The val row for the *same* ResNet-18 is what isolates the cause: it is the
**model**, not the split. Swapping the 116k-parameter CNN for the fine-tuned
ResNet-18 takes the class-matched advantage from +0.105 (z = 5.09) to +0.033
(z = 1.80) on identical images.

The likely mechanism is a ceiling, and it is visible in the per-class rows. The
leaky subset is 70–80% No-Anomaly, and the ResNet-18 scores **1.0000** on leaky
No-Anomaly against **0.9814** on clean No-Anomaly — under two points of headroom
for any advantage to live in, where the small CNN left eight. A better model
closes the gap the leakage was exploiting. ADR 0004's consequences section said
the opposite would happen; it now carries the correction and the retraction.

**The causal claim in ADR 0004 is untouched by this.** That claim rests on the
controlled removal experiment — 4.9 points lost under neighbour removal, 0.0000
under the size- and class-matched random control — not on the size of the
class-matched gap. What moves is how much the leakage is *worth to a given
model*, which was always stated as model-specific. What it costs the headline
number is still bounded at 0.005 macro-F1 and is still inside the noise floor.

**The test split is not read again for the rest of the project.**

## Calibration — T fitted on val, never on test

`scripts/posthoc_cls_d1.py`, from the saved logits of the one test run. No
training, no second read of the test split.

**T = 2.0463.** The model was overconfident by a factor of two. Val NLL
0.8542 → 0.5873; test NLL 0.8544 → 0.5899.

| | global ECE | class-balanced ECE | mean confidence | accuracy |
|---|---:|---:|---:|---:|
| val, raw | 0.1183 | 0.2965 | 0.9488 | 0.8310 |
| val, T-scaled | 0.0229 | 0.2099 | 0.8451 | 0.8310 |
| test, raw | 0.1215 | 0.2645 | 0.9461 | 0.8251 |
| **test, T-scaled** | **0.0211** | **0.1835** | 0.8419 | 0.8251 |

**The two columns are a factor of nine apart, and that gap is the result.** Global
ECE says the calibrated model is nearly perfect: 0.021. Class-balanced ECE — the
mean of the per-class ECEs, so every class counts once — says 0.184. Global ECE is
half No-Anomaly by construction, and No-Anomaly is the class the model is most
confident and most often right about, so it drags the global number to zero and
takes the eleven defect classes with it.

Per-class ECE on test, and the row that matters is the first and the second-last:

| class | support | raw | T-scaled |
|---|---:|---:|---:|
| Soiling | 32 | 0.5356 | **0.4133** |
| Cell-Multi | 194 | 0.4793 | 0.3374 |
| Hot-Spot-Multi | 38 | 0.4372 | 0.3327 |
| Hot-Spot | 38 | 0.3409 | 0.2011 |
| Shadowing | 158 | 0.2950 | 0.1838 |
| Cracking | 142 | 0.2451 | 0.1492 |
| Offline-Module | 124 | 0.2422 | 0.1456 |
| Diode-Multi | 27 | 0.1538 | 0.1424 |
| Vegetation | 247 | 0.2078 | 0.1052 |
| **No-Anomaly** | **1,500** | **0.0153** | **0.0681** |
| Cell | 282 | 0.1790 | 0.0658 |
| Diode | 225 | 0.0424 | 0.0570 |

**One global temperature made No-Anomaly worse to make everything else better.**
Its ECE goes 0.0153 → 0.0681, a four-fold degradation on the class carrying half
the data, and the global ECE still improved six-fold because the other eleven
classes improved more. A single scalar cannot fix eleven differently-miscalibrated
heads; it moves confidence mass in one direction for all of them. Soiling is still
at 0.41 after scaling — the model's stated probability on Soiling is off by 41
points on average, and no amount of temperature will fix that because the problem
is that it is wrong about Soiling, not that it is confidently wrong.

Reliability diagram: `runs/cls_final_test_20260827T205903Z/reliability_test.png`.

## Base-rate correction — D1 is 49.9% No-Anomaly, the field is 2.2%

**Two assumptions, in the same place as the table, because the arithmetic is three
lines and the assumptions are the content.**

- **A1.** Per-class sensitivity, and the whole class-conditional confusion
  structure P(pred = c | true = k), transport unchanged from D1 to the field. D1
  is already a mixture — pooled midwave and longwave sensors at 3–15 cm/px — so
  "the field" is a *different* mixture, not a special case of this one.
- **A2.** Only the anomaly/no-anomaly ratio shifts; the relative mix among the
  eleven anomaly classes holds. This is the weaker of the two. Soiling and
  Vegetation are seasonal and site-specific in a way a bypass-diode fault is not.

Neither is obviously true, and the table below is worth exactly what they are.

Under A2 the reweighting is a single constant on every anomaly class, so **recall
is invariant to the prevalence shift and only precision moves.** There is no
"recall at 2.2%" column because there is no such quantity.

| class | test n | prior | field prior | recall | precision as measured | **precision @ 2.2%** |
|---|---:|---:|---:|---:|---:|---:|
| No-Anomaly | 1,500 | 0.4988 | 0.97800 | 0.9753 | 0.9076 | 0.9977 |
| Diode-Multi | 27 | 0.0090 | 0.00039 | 0.7778 | 0.8400 | **0.8400** |
| Diode | 225 | 0.0748 | 0.00328 | 0.9378 | 0.9505 | 0.5978 |
| Soiling | 32 | 0.0106 | 0.00047 | 0.3750 | 0.5217 | **0.5217** |
| Cracking | 142 | 0.0472 | 0.00207 | 0.6338 | 0.6923 | 0.5182 |
| Cell-Multi | 194 | 0.0645 | 0.00283 | 0.4072 | 0.5524 | 0.4232 |
| Vegetation | 247 | 0.0821 | 0.00361 | 0.7085 | 0.7202 | 0.3794 |
| Cell | 282 | 0.0938 | 0.00412 | 0.7270 | 0.6231 | 0.3022 |
| Hot-Spot-Multi | 38 | 0.0126 | 0.00055 | 0.4737 | 0.6923 | 0.2584 |
| Offline-Module | 124 | 0.0412 | 0.00181 | 0.6613 | 0.8200 | **0.2022** |
| Shadowing | 158 | 0.0525 | 0.00231 | 0.6519 | 0.8047 | 0.1977 |
| Hot-Spot | 38 | 0.0126 | 0.00055 | 0.5789 | 0.8462 | **0.1941** |

Offline-Module goes from 0.82 precision to 0.20: four out of five modules the
model calls offline would not be, in a field where only 2.2% of modules have
anything wrong. Hot-Spot goes 0.85 → 0.19. **This is the number that decides
whether a model is deployable, and it is not the macro-F1.**

**Diode-Multi and Soiling do not move at all**, and the reason is exact rather
than lucky: the No-Anomaly row of the confusion matrix has a **zero** in both of
those columns. A class the majority is never confused with has no majority-driven
false positives to inflate, so reweighting the majority cannot touch its
precision. Prevalence hurts precisely the classes No-Anomaly leaks into.

### The threshold, chosen for a stated precision target rather than for macro-F1

Flag a module if the calibrated P(any anomaly) ≥ t. The ranking is
prevalence-invariant — the reweighting is one constant per group, so it is a
monotone transform — which is why the sweep runs on the ordinary score and only
the precision column is reweighted.

| t | flags per 1,000 modules | recall | precision as measured | **precision @ 2.2%** |
|---:|---:|---:|---:|---:|
| 0.10 | 244.6 | 0.9834 | 0.8125 | 0.0884 |
| 0.25 | 103.9 | 0.9589 | 0.9192 | 0.2030 |
| 0.50 | 58.2 | 0.9244 | 0.9600 | 0.3497 |
| **0.7151** | **39.3** | **0.8958** | — | **0.5019** |
| 0.90 | 25.1 | 0.8454 | 0.9922 | 0.7404 |
| 0.95 | 22.2 | 0.8036 | 0.9943 | 0.7948 |
| 0.99 | 15.1 | 0.6271 | 0.9979 | 0.9136 |

**Stated target: 50% precision at field prevalence — one real defect per two
dispatches. It is met at t = 0.7151, with recall 0.8958 and 39.3 flags per 1,000
modules.** That threshold is chosen for precision, not for macro-F1; the
macro-F1-optimal point is plain argmax, which sits far to the left of this and
flags 245 modules per 1,000 at 9% precision.

**The binary triage decision survives the prevalence shift much better than the
twelve-way label does**, and the two tables together are the honest summary. You
can send a crew to 39 modules per 1,000 and expect half of them to be real. You
cannot tell that crew *what* is wrong: at that operating point Offline-Module
precision is 0.20 and Hot-Spot is 0.19. Whether a module is faulty and which fault
it has are different problems, and only the first one is answered at 2.2%.

## Runs

The `config` column carries `path#arm` for a config that declares several arms.

| run | git SHA | dirty | split | config | seed | eval split | epochs | wall | macro-F1 | acc | null acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260827T134744Z | `6f1f6fe` | yes | `4cbb0c3d` | `configs/cls_baseline.yaml` | 0 | val | 30 | 14 s | **0.5907** | 0.7783 | 0.5008 |
| 20260827T135123Z | `6bc2d76` | no | `4cbb0c3d` | `configs/cls_baseline.yaml` | 0 | val | 30 | 14 s | **0.5907** | 0.7783 | 0.5008 |
| 20260827T191639Z | `427111e` | no | `af8781b1` | `configs/cls_baseline.yaml` | 0 | val | 30 | 23 s | **0.5799** | 0.7567 | 0.5013 |
| 20260827T203538Z | `133c67b` | no | `af8781b1` | `configs/cls_baseline.yaml` | 0 | val | 30 | 23 s | **0.5799** | 0.7567 | 0.5013 |
| 20260827T203719Z_baseline_s0 | `f3d71fd` | no | `af8781b1` | `configs/cls_ablation.yaml#baseline` | 0 | val | 30 | 23 s | **0.6067** | 0.7892 | 0.5013 |
| 20260827T203719Z_baseline_s1 | `f3d71fd` | no | `af8781b1` | `configs/cls_ablation.yaml#baseline` | 1 | val | 30 | 22 s | **0.5897** | 0.7811 | 0.5013 |
| 20260827T203719Z_baseline_s2 | `f3d71fd` | no | `af8781b1` | `configs/cls_ablation.yaml#baseline` | 2 | val | 30 | 22 s | **0.5975** | 0.7815 | 0.5013 |
| 20260827T203719Z_class-weights_s0 | `f3d71fd` | no | `af8781b1` | `configs/cls_ablation.yaml#class-weights` | 0 | val | 30 | 22 s | **0.5541** | 0.7222 | 0.5013 |
| 20260827T203719Z_class-weights_s1 | `f3d71fd` | no | `af8781b1` | `configs/cls_ablation.yaml#class-weights` | 1 | val | 30 | 22 s | **0.5660** | 0.7239 | 0.5013 |
| 20260827T203719Z_class-weights_s2 | `f3d71fd` | no | `af8781b1` | `configs/cls_ablation.yaml#class-weights` | 2 | val | 30 | 21 s | **0.5645** | 0.7249 | 0.5013 |
| 20260827T203719Z_focal_s0 | `f3d71fd` | no | `af8781b1` | `configs/cls_ablation.yaml#focal` | 0 | val | 30 | 26 s | **0.5906** | 0.7808 | 0.5013 |
| 20260827T203719Z_focal_s1 | `f3d71fd` | no | `af8781b1` | `configs/cls_ablation.yaml#focal` | 1 | val | 30 | 26 s | **0.5841** | 0.7687 | 0.5013 |
| 20260827T203719Z_focal_s2 | `f3d71fd` | no | `af8781b1` | `configs/cls_ablation.yaml#focal` | 2 | val | 30 | 25 s | **0.5818** | 0.7771 | 0.5013 |
| 20260827T203719Z_resampling_s0 | `f3d71fd` | no | `af8781b1` | `configs/cls_ablation.yaml#resampling` | 0 | val | 30 | 22 s | **0.5849** | 0.7537 | 0.5013 |
| 20260827T203719Z_resampling_s1 | `f3d71fd` | no | `af8781b1` | `configs/cls_ablation.yaml#resampling` | 1 | val | 30 | 22 s | **0.5775** | 0.7490 | 0.5013 |
| 20260827T203719Z_resampling_s2 | `f3d71fd` | no | `af8781b1` | `configs/cls_ablation.yaml#resampling` | 2 | val | 30 | 21 s | **0.5789** | 0.7537 | 0.5013 |
| 20260827T204537Z_resnet18_s0 | `0d15ffe` | no | `af8781b1` | `configs/cls_resnet18.yaml#resnet18` | 0 | val | 30 | 163 s | **0.6735** | 0.8310 | 0.5013 |
| 20260827T204537Z_resnet18_s1 | `0d15ffe` | no | `af8781b1` | `configs/cls_resnet18.yaml#resnet18` | 1 | val | 30 | 162 s | **0.6855** | 0.8330 | 0.5013 |
| 20260827T204537Z_resnet18_s2 | `0d15ffe` | no | `af8781b1` | `configs/cls_resnet18.yaml#resnet18` | 2 | val | 30 | 161 s | **0.6719** | 0.8333 | 0.5013 |
| 20260827T205903Z | `c1df507` | no | `af8781b1` | `configs/cls_final.yaml` | 0 | test | 30 | 164 s | **0.6956** | 0.8251 | 0.4988 |
