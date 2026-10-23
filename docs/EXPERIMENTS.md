# Experiments

**The history was rewritten on 2026-08-28 (commit messages only).
Every SHA recorded here before that date was remapped to its new commit;
trees, author dates and committer dates are unchanged, so the runs themselves are the same
runs. A ledger that silently changed its own primary keys would be worse than one that
says it did.**

Every training run, in order. A run with `dirty=yes` was made against a
working tree that did not match its commit, so its numbers are not
reproducible from that SHA -- treat them as indicative only.

**`split` is the first 8 hex of `configs/d1_split.json`'s sha256.** D1 was
deduplicated and re-split at `4b65b1d`, so a row on `4cbb0c3d` and a row on
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

By the end of the classification study `val` carries four jobs at once: **epoch selection** inside
every run, **config selection** between the ablation arms, the **temperature fit**
for calibration, and the **leakage diagnostic** of ADR 0004. Each is a legitimate
use of a validation split and together they are still not a test set -- but four
uses is four chances to overfit it, and the val numbers here should be read as
optimistic by an unmeasured amount. Said here rather than left for a reader to
notice.

## Ablation result — three seeds per arm, val, split `af8781b1`, run at `93af4b5`

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
cross-entropy at a learning rate held fixed across arms and tuned for none of
them, and class weighting is measurably worse than two of the arms.**
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
0.010–0.015.

**Which calls would flip, stated rather than left for a reader to work out.** A
floor re-derived from the observed 0.004–0.009 spread the way the declared one was
derived from 0.012–0.019 lands somewhere around 0.010–0.015, and at that threshold
the two comparisons the headline finding rests on both resolve: `baseline −
resampling` at **+0.0175** flips under either end of that range, and `baseline −
focal` at **+0.0125** flips under the lower end. Both would resolve in baseline's
favour, which would turn "no treatment beat plain cross-entropy" into "plain
cross-entropy beat two of the three" — a stronger claim than the one made here.
(`class weights − resampling` at −0.0189 would also resolve, but it says nothing
about the headline finding.) **The pre-declared 0.02 floor was kept anyway and the
weaker claim is the one that ships**, because a floor moved after seeing the
numbers it judges is not a floor, and the direction it would move in here happens
to favour the result.

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

## Transfer-learning baseline — ImageNet ResNet-18, val, run at `d4100d6`

| arm | val macro-F1, mean ± std | seeds | selected epochs | val accuracy | params |
|---|---:|---|---|---:|---:|
| resnet18, fine-tuned | **0.6770 ± 0.0074** | 0.6735 · 0.6855 · 0.6719 | 13 · 28 · 17 | 0.832 | 11,176,396 |

Against the best from-scratch arm (0.5980 ± 0.0085) that is **+0.079 macro-F1,
four times the declared floor** — the one comparison in the classification study that is not close.
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

**Config commit: [`1a0fdfa`](../../../commit/1a0fdfa).** `configs/cls_final.yaml`
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
  the same pair the baseline run's small CNN got wrong most often. Distinguishing one hot
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

## Distillation — three rows, on val, and the KD gain is nothing

Teacher is the exact seed-0 ResNet-18 behind the test number (config commit
`1a0fdfa`). Both student arms are trained by the same `fit()` in the same script
on the same three seeds at the same 30-epoch cosine budget, differing in one
config key. `configs/cls_kd.yaml` and α = 0.5, T = 4.0, declared before the run
and not swept.

| | val macro-F1 | params | CPU ms / batch of 64 | ms / image |
|---|---:|---:|---:|---:|
| teacher, ResNet-18 | 0.6735 (1 seed) | 11,176,396 | 706.08 | 11.0325 |
| student, from scratch | 0.5980 ± 0.0085 | 115,948 | 228.65 | 3.5727 |
| student, distilled | 0.5987 ± 0.0075 | 115,948 | 222.86 | 3.4822 |

**KD gain: +0.0007 macro-F1. The declared floor is 0.02. Indistinguishable.**
Distillation did not measurably help this student, at this budget, at α = 0.5 and
T = 4. The teacher–student gap it was trying to close is **+0.0756** — real, four
noise floors wide, and untouched.

Three things keep that from being a null result nobody can check:

**The middle row is why the number means anything.** A distilled student at 0.5987
is uninterpretable on its own. It is only against the same student on hard labels
at the same budget, 0.5980, that +0.0007 can be read at all — and the
from-scratch arm reproduces the ablation's baseline arm seed-for-seed (0.6067 ·
0.5897 · 0.5975), which is the evidence that it is the same experiment and not a
second one.

**The KD term is switched on, and that is checked rather than hoped.** Two of the
three ways this loss silently fails make distillation look useless, which is
exactly the result above, so "it didn't help" and "I wrote it wrong" have to be
told apart. `tests/test_cls_metrics.py` pins all three: `batchmean` is exactly
`num_classes` times `mean` (the 1/12 bug, as a number); α = 1 reduces the loss to
plain cross-entropy; and the T² factor holds the soft term flat in T where the
unscaled term falls as 1/T². The training trajectories also differ — the distilled
arm selects epochs 20 · 23 · 18 against the from-scratch arm's 25 · 21 · 27 — so
the term is carrying real gradient weight.

**What is not ruled out** is that KD helps at some other (α, T). It was not swept,
deliberately: a sweep with val selection over twelve runs finds something above
the floor by chance, and the honest write-up of that has to begin "I searched".

### Latency measures the framework, not the model

**96× fewer parameters buys 3.1× less wall time.** That is not a disappointing
speedup, it is the wrong question being asked of a 40×24 model: at this size a
forward pass is a few hundred microseconds of arithmetic wrapped in Python
attribute lookups and PyTorch dispatch, and per-op overhead dominates. The student
runs 18 ops; the teacher runs several dozen. That ratio, not the FLOP ratio, is
what the 3.1× is measuring.

Conditions, stated because the number is meaningless without them: **CPU, one
thread, batch 64, 50 warm-up passes discarded, median of 200.** Anything that
would make this a FLOP-bound measurement — a real input size, INT8, ONNX Runtime
or TensorRT, operator fusion, batch 1 on an embedded target — is exactly the list
of levers this project has *not* pulled, and the honest version of "4× smaller" has
to say what the 4× actually bought.

### Where soft targets were supposed to act: Cell ↔ Cell-Multi

The baseline run's confusion matrix made this the prediction to check. Cell ↔ Cell-Multi is
the largest off-diagonal pair in the project, and confusability between
near-identical classes is precisely the structure "dark knowledge" is supposed to
carry. Mean counts over the three seeds, on val:

| | Cell → Cell-Multi | Cell-Multi → Cell | Cell F1 | Cell-Multi F1 |
|---|---:|---:|---:|---:|
| teacher | 22.0 | 45.0 | 0.6712 | 0.4985 |
| student, from scratch | 39.7 | 42.0 | 0.6211 | 0.4099 |
| student, distilled | 41.0 | 38.3 | 0.6057 | 0.4148 |

**It did not act.** The distilled student moved 3.7 errors from one direction of
the pair to the other and lost 1.5 points of Cell F1 to gain 0.5 of Cell-Multi.
The teacher's own pattern is qualitatively different from either student's — it is
half as likely to call a Cell a Cell-Multi and slightly *more* likely to do the
reverse — and none of that asymmetry transferred.

The most likely reason is capacity rather than the loss. The student is 116k
parameters on 40×24 crops with three downsamples available to it; telling one hot
cell from several is a counting problem, and a model that cannot represent the
distinction does not learn it from being told the teacher's uncertainty about it.
That is a hypothesis, not a measurement — the experiment that would test it is a
capacity sweep on the student, which was not run.

## Runs

The `config` column carries `path#arm` for a config that declares several arms.

| run | git SHA | dirty | split | config | seed | eval split | epochs | wall | macro-F1 | acc | null acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260827T134744Z | `1b931b0` | yes | `4cbb0c3d` | `configs/cls_baseline.yaml` | 0 | val | 30 | 14 s | **0.5907** | 0.7783 | 0.5008 |
| 20260827T135123Z | `4ff1b1e` | no | `4cbb0c3d` | `configs/cls_baseline.yaml` | 0 | val | 30 | 14 s | **0.5907** | 0.7783 | 0.5008 |
| 20260827T191639Z | `9e2c96f` | no | `af8781b1` | `configs/cls_baseline.yaml` | 0 | val | 30 | 23 s | **0.5799** | 0.7567 | 0.5013 |
| 20260827T203538Z | `ba365e4` | no | `af8781b1` | `configs/cls_baseline.yaml` | 0 | val | 30 | 23 s | **0.5799** | 0.7567 | 0.5013 |
| 20260827T203719Z_baseline_s0 | `93af4b5` | no | `af8781b1` | `configs/cls_ablation.yaml#baseline` | 0 | val | 30 | 23 s | **0.6067** | 0.7892 | 0.5013 |
| 20260827T203719Z_baseline_s1 | `93af4b5` | no | `af8781b1` | `configs/cls_ablation.yaml#baseline` | 1 | val | 30 | 22 s | **0.5897** | 0.7811 | 0.5013 |
| 20260827T203719Z_baseline_s2 | `93af4b5` | no | `af8781b1` | `configs/cls_ablation.yaml#baseline` | 2 | val | 30 | 22 s | **0.5975** | 0.7815 | 0.5013 |
| 20260827T203719Z_class-weights_s0 | `93af4b5` | no | `af8781b1` | `configs/cls_ablation.yaml#class-weights` | 0 | val | 30 | 22 s | **0.5541** | 0.7222 | 0.5013 |
| 20260827T203719Z_class-weights_s1 | `93af4b5` | no | `af8781b1` | `configs/cls_ablation.yaml#class-weights` | 1 | val | 30 | 22 s | **0.5660** | 0.7239 | 0.5013 |
| 20260827T203719Z_class-weights_s2 | `93af4b5` | no | `af8781b1` | `configs/cls_ablation.yaml#class-weights` | 2 | val | 30 | 21 s | **0.5645** | 0.7249 | 0.5013 |
| 20260827T203719Z_focal_s0 | `93af4b5` | no | `af8781b1` | `configs/cls_ablation.yaml#focal` | 0 | val | 30 | 26 s | **0.5906** | 0.7808 | 0.5013 |
| 20260827T203719Z_focal_s1 | `93af4b5` | no | `af8781b1` | `configs/cls_ablation.yaml#focal` | 1 | val | 30 | 26 s | **0.5841** | 0.7687 | 0.5013 |
| 20260827T203719Z_focal_s2 | `93af4b5` | no | `af8781b1` | `configs/cls_ablation.yaml#focal` | 2 | val | 30 | 25 s | **0.5818** | 0.7771 | 0.5013 |
| 20260827T203719Z_resampling_s0 | `93af4b5` | no | `af8781b1` | `configs/cls_ablation.yaml#resampling` | 0 | val | 30 | 22 s | **0.5849** | 0.7537 | 0.5013 |
| 20260827T203719Z_resampling_s1 | `93af4b5` | no | `af8781b1` | `configs/cls_ablation.yaml#resampling` | 1 | val | 30 | 22 s | **0.5775** | 0.7490 | 0.5013 |
| 20260827T203719Z_resampling_s2 | `93af4b5` | no | `af8781b1` | `configs/cls_ablation.yaml#resampling` | 2 | val | 30 | 21 s | **0.5789** | 0.7537 | 0.5013 |
| 20260827T204537Z_resnet18_s0 | `d4100d6` | no | `af8781b1` | `configs/cls_resnet18.yaml#resnet18` | 0 | val | 30 | 163 s | **0.6735** | 0.8310 | 0.5013 |
| 20260827T204537Z_resnet18_s1 | `d4100d6` | no | `af8781b1` | `configs/cls_resnet18.yaml#resnet18` | 1 | val | 30 | 162 s | **0.6855** | 0.8330 | 0.5013 |
| 20260827T204537Z_resnet18_s2 | `d4100d6` | no | `af8781b1` | `configs/cls_resnet18.yaml#resnet18` | 2 | val | 30 | 161 s | **0.6719** | 0.8333 | 0.5013 |
| 20260827T205903Z | `1a0fdfa` | no | `af8781b1` | `configs/cls_final.yaml` | 0 | test | 30 | 164 s | **0.6956** | 0.8251 | 0.4988 |
| 20260827T210715Z_student-from-scratch_s0 | `82872a8` | no | `af8781b1` | `configs/cls_kd.yaml#student-from-scratch` | 0 | val | 30 | 22 s | **0.6067** | 0.7892 | 0.5013 |
| 20260827T210715Z_student-from-scratch_s1 | `82872a8` | no | `af8781b1` | `configs/cls_kd.yaml#student-from-scratch` | 1 | val | 30 | 22 s | **0.5897** | 0.7811 | 0.5013 |
| 20260827T210715Z_student-from-scratch_s2 | `82872a8` | no | `af8781b1` | `configs/cls_kd.yaml#student-from-scratch` | 2 | val | 30 | 21 s | **0.5975** | 0.7815 | 0.5013 |
| 20260827T210715Z_student-distilled_s0 | `82872a8` | no | `af8781b1` | `configs/cls_kd.yaml#student-distilled` | 0 | val | 30 | 26 s | **0.6036** | 0.7888 | 0.5013 |
| 20260827T210715Z_student-distilled_s1 | `82872a8` | no | `af8781b1` | `configs/cls_kd.yaml#student-distilled` | 1 | val | 30 | 26 s | **0.5901** | 0.7821 | 0.5013 |
| 20260827T210715Z_student-distilled_s2 | `82872a8` | no | `af8781b1` | `configs/cls_kd.yaml#student-distilled` | 2 | val | 30 | 26 s | **0.6023** | 0.7744 | 0.5013 |

## D4 regression runs

Same ledger rules as above (`git SHA` is HEAD when the run started, `dirty` is the tree at that
moment), with the columns a regression needs instead of a classifier's. Each run fits
AC_POWER on IRRADIATION, MODULE_TEMPERATURE and AMBIENT_TEMPERATURE per plant, values given as
Plant 1 / Plant 2. R² is **in-sample**: nothing is held out, so it measures fit, not prediction.
The diff columns are the largest absolute coefficient difference from `np.linalg.lstsq`, over both
plants, daylight fit. The last column was added later, copied from the same run's `fit.json`.
Full output: `runs/ols_d4_<run>/fit.json`.

| run | git SHA | dirty | script | daylight n | R² daylight | R² with night | normal eq. vs lstsq | grad. descent vs lstsq | n with night |
|---|---|---|---|---|---|---|---|---|---|
| 20261021T160204Z | `89aaa3e` | no | `scripts/ols_d4.py` | 38,376 / 38,722 | **0.9602** / **0.4354** | 0.9790 / 0.6136 | 2.6e-10 | 5.9e-06 | 68,774 / 67,698 |

## D4 plant-level diagnostics runs

Same ledger rules. Diagnostics of the plant-level daylight fit above (same rows, same
coefficients), values Plant 1 / Plant 2. The fit pools 22 inverter series, so Durbin-Watson and the
Newey-West lags are computed within each inverter's series; Driscoll-Kraay (statsmodels
`hac-groupsum`, scores summed per timestamp) also allows for the error shared by inverters under one
weather sensor. SE ratios are the range over the four coefficients of that covariance's standard
error divided by the classical one, at the HAC lag L shown. Full output, including the per-coefficient
standard errors, Breusch-Pagan and the ACF: `runs/d4_diag_<run>/diagnostics.json`.

| run | git SHA | dirty | script | n / inverters / timestamps | R² (adjusted) | VIF range | DW within inverter | HAC L | SE ratio, Newey-West / classical | SE ratio, Driscoll-Kraay / classical |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261021T162706Z | `a6cf29f` | no | `scripts/diagnostics_d4.py` | 38,376 / 22 / 1,754 ; 38,722 / 22 / 1,862 | 0.9602 (0.9602) / 0.4354 (0.4354) | 4.4 to 23.1 / 6.7 to 30.5 | 1.010 / 0.253 | 15 / 15 | 1.31 to 1.90 / 1.62 to 3.09 | 3.53 to 5.33 / 3.50 to 5.17 |

## D4 per-inverter runs

Same ledger rules. Rules from `configs/d4_analysis.yaml`, committed before any per-inverter number
existed: counted intervals have IRRADIATION ≥ 0.1 kW/m², producing means AC_POWER > 0. Values are
Plant 1 / Plant 2, 22 inverters each. The slope is the IRRADIATION coefficient of a per-inverter OLS
of AC_POWER on IRRADIATION over its producing counted intervals, with a 95% Newey-West interval;
"CIs at median" counts inverters whose interval contains the plant's median slope, i.e. that cannot
be told apart from it. Spearman columns compare the slope ranking with the robustness model's
(MODULE_TEMPERATURE added) and with the sister-ratio ranking; the last column is how many of the
bottom three by slope are also in the bottom three by sister ratio. The common-window column is the
Spearman agreement of the declared availability ranking with availability recomputed only at the
timestamps where every inverter of the plant has a row (declared in `ab75d6b`; a robustness check,
never a replacement). The last column counts inverters whose 95% interval excludes the plant median
and, in brackets, how many of those are still excluded by a Bonferroni interval (z for 0.05 / 22,
two-sided, about 3.05): at 95%, about one of 22 is expected to miss by chance. The last two columns
were added later, copied from each run's own `rank.json`: counted rows (IRRADIATION ≥ 0.1 kW/m²)
with the number of counted timestamps, and the common window's timestamps out of all. Full output:
`runs/d4_rank_<run>/rank.json`.

| run | git SHA | dirty | script | producing counted rows | availability median (min) | CIs at median | Spearman slope vs robustness | Spearman slope vs sister ratio | bottom-3 overlap | Spearman availability vs common window | flagged at 95% (surviving Bonferroni) | counted rows (counted timestamps) | common-window timestamps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261021T162847Z | `f77ec31` | no | `scripts/rank_d4.py` | 30,655 / 25,044 | 1.0000 (0.9857) / 0.8714 (0.7716) | 13 / 13 | 0.805 / 0.587 | 0.740 / 0.907 | 2 / 2 | not computed | not computed | 30,718 (1,397) / 28,818 (1,392) | not computed |
| 20261021T180059Z | `277d21c` | no | `scripts/rank_d4.py` | 30,655 / 25,044 | 1.0000 (0.9857) / 0.8714 (0.7716) | 13 / 13 | 0.805 / 0.587 | 0.740 / 0.907 | 2 / 2 | 1.000 / 0.914 | not computed | 30,718 (1,397) / 28,818 (1,392) | 3,057 of 3,157 / 2,355 of 3,259 |
| 20261021T180327Z | `de4fb59` | no | `scripts/rank_d4.py` | 30,655 / 25,044 | 1.0000 (0.9857) / 0.8714 (0.7716) | 13 / 13 | 0.805 / 0.587 | 0.740 / 0.907 | 2 / 2 | 1.000 / 0.914 | 9 (4) / 9 (5) | 30,718 (1,397) / 28,818 (1,392) | 3,057 of 3,157 / 2,355 of 3,259 |

## D4 post-hoc checks

Same ledger rules. **Post-hoc**: each check here was made after the per-inverter rankings were
seen, to explain a number rather than to choose a model, and none replaces a declared fit. The
R² split refits the plant-level three-feature model of `scripts/ols_d4.py` on four row sets;
"daylight" is IRRADIATION > 0 (the declared fit), "counted" is IRRADIATION ≥ 0.1 kW/m², and
"producing" keeps AC_POWER > 0 only. Values are R² (n rows), Plant 1 / Plant 2, in-sample. Full
output: `runs/d4_r2split_<run>/r2_split.json`.

| run | git SHA | dirty | script | daylight, zeros kept (declared) | daylight, producing | counted, zeros kept | counted, producing |
|---|---|---|---|---|---|---|---|
| 20261021T171435Z | `9153f89` | no | `scripts/r2_split_d4.py` | 0.9602 (38,376) / 0.4354 (38,722) | 0.9695 (36,823) / 0.8854 (32,022) | 0.9316 (30,718) / 0.1999 (28,818) | 0.9520 (30,655) / 0.8118 (25,044) |

## D4 confounder runs

Same ledger rules. Checks of what lies behind the per-inverter rankings, values Plant 1 / Plant 2;
each describes structure in the data and none names a cause. Zero-output intervals are counted
intervals (IRRADIATION ≥ 0.1 kW/m²) with AC_POWER = 0, with the number of stretches they form; each
is classed by how many inverters with a counted row at that timestamp are at zero too (plant-wide:
more than half; single: that inverter only; partial: in between), Plant 2 only. Near max is the
median over inverters of the share of producing counted rows within 2% of that inverter's maximum
AC_POWER; slope bend is the median over inverters of the slope on all producing counted rows over
the slope below the plant's 90th percentile of irradiance. Night readings are timestamps before
05:00 or from 19:00 with IRRADIATION > 0. The DC column is the Spearman agreement of a DC_POWER
slope ranking with the AC_POWER one. The last column is the median over inverters of the standard
error of "slope minus the mean slope of the other 21" from a pooled fit with Driscoll-Kraay errors
(statsmodels), over the same standard error with the per-inverter Newey-West fits treated as
independent. Full output: `runs/d4_conf_<run>/confounders.json`.

| run | git SHA | dirty | script | zero-output intervals (stretches) | P2 zero intervals plant-wide / partial / single | near max | slope bend | night readings | Spearman DC vs AC slope | contrast SE, Driscoll-Kraay / independent |
|---|---|---|---|---|---|---|---|---|---|---|
| 20261021T180557Z | `9e0b4cd` | no | `scripts/confounders_d4.py` | 63 (15) / 3,774 (531) | 423 / 3,264 / 87 | 0.0007 / 0.0016 | 0.958 / 0.994 | 5 / 64 | 0.999 / 1.000 | 0.535 / 0.717 |

## D2 AP@0.5 cross-check on synthetic boxes

Same ledger rules. No model and no D2 data: seeded synthetic frames (a metric test fixture
described in `scripts/crosscheck_ap.py`, 40 frames per seed, 0 to 600 ground-truth boxes each,
placed at random so ground truths overlap one another, unlike tiled panels). From-scratch AP@0.5
(`src/solar_inspect/detection/ap.py`) against torchmetrics 1.9.0 / pycocotools 2.0.11 AP@0.5 at
1000 detections per image. "Controlled" removes three differences from the from-scratch side's
inputs: score ties broken into unique ranks (the order of equal scores is
implementation-defined), boxes rounded as torchmetrics hands them to pycocotools (float32 x, y,
w, h), and ground truths reversed per image (on an exact IoU tie between two ground truths
pycocotools keeps the last, `ap.match` the first). Run from `34c508b`, clean tree.

| seed | ground truths | detections | all-point | 101-point | torchmetrics | 101-point minus torchmetrics, raw | controlled |
|---|---|---|---|---|---|---|---|
| 0 | 14,589 | 16,632 | 0.752867 | 0.753324 | 0.753404 | -8.0e-05 | 0 |
| 1 | 14,225 | 16,191 | 0.760882 | 0.759263 | 0.759094 | +1.7e-04 | +4.1e-06 |
| 2 | 10,882 | 12,345 | 0.749159 | 0.748431 | 0.748529 | -9.8e-05 | 0 |
| 3 | 11,030 | 12,554 | 0.752369 | 0.747151 | 0.747327 | -1.8e-04 | +3.9e-07 |
| 4 | 12,971 | 14,706 | 0.753961 | 0.749804 | 0.749630 | +1.7e-04 | 0 |

**Unresolved:** the controlled residual of 4.1e-06 (seed 1) and 3.9e-07 (seed 3). Per-image TP
counts agree exactly on every frame of both seeds, so it sits in the global accumulation; it is
not pycocotools' epsilon in the precision denominator, not float32 summation of the precision
tensor, and not frames without ground truth (there are none). Left at that after about 15
minutes of a 60-minute reconciliation budget.

## D2 detection runs

Same ledger rules. Split: the sortie split from `configs/d2.yaml` (ADR 0002); training reads train
(138 images, 10,612 boxes) and selects on val; **every val number here is on val, 51 images, 4,929
boxes**. Nothing here reads test. Config `configs/det_yolo.yaml` (YOLOv8n from `yolov8n.pt`, seed 0,
batch 16, imgsz 640, augmentation as declared in `4b0848c`); overrides are listed where used.
Training rows give Ultralytics' in-training val numbers for the best epoch, which Ultralytics 8.4
selects on mAP50-95. Full output: `runs/det_<run>/manifest.json`, `results.csv`, `args.yaml`.

| run | git SHA | dirty | script | what ran | epochs run (best) | wall | peak VRAM reserved | val mAP50 / mAP50-95 at best epoch, 51 images, 4,929 boxes |
|---|---|---|---|---|---|---|---|---|
| 20261022T072913Z | `92a01b4` | no | `scripts/train_det.py` | smoke, `epochs=2` | 2 (2) | 85 s | not recorded | 0.00081 / 0.00016 |
| 20261022T073134Z | `d4ea608` | no | `scripts/train_det.py` | smoke, `epochs=2`, after the manifest fix | 2 (2) | 67 s | 4,468 MiB | 0.00081 / 0.00016 |
| 20261022T073325Z | `d4ea608` | no | `scripts/train_det.py` | full config, no overrides | 100 (81) | 803 s | 5,450 MiB | 0.9733 / 0.8213 |
| 20261022T111038Z | `86aed25` | no | `scripts/train_det.py` | smoke, `model=yolov8s.pt epochs=2`, batch 16 (the tuning pass's batch check); smoke, scores not informative | 2 (2) | 54 s | 5,862 MiB | 0.00029 / 0.00006 |

The first smoke run's peak VRAM was printed (4,478 MiB) but its manifest is from before the
fix and is otherwise identical; the two smoke runs give the same numbers to five decimals.
Patience 20 never triggered: the best epoch was 81 and training ran all 100.

### Val evaluation of `det_20261022T073325Z/weights/best.pt`

One Ultralytics validation pass with `conf 0.001`, `iou 0.7`, `max_det 1000`; every number except
Ultralytics' own is computed from the `predictions.json` that pass wrote. Run
`det_val_20261022T074726Z` from `d4ea608`, clean tree, `scripts/eval_val_det.py`. Val, 51 images,
4,929 boxes, 14,056 detections.

| metric (val, 51 images, 4,929 boxes) | Ultralytics | from scratch, all-point | from scratch, 101-point | torchmetrics |
|---|---|---|---|---|
| AP@0.5 | 0.973270 | 0.977807 | 0.973185 | 0.973185 |
| AP@0.5:0.95 | 0.822093 | | 0.824685 (10 thresholds, 101-point; computed in the reconciliation below) | 0.824685 |

| confidence (val, 51 images, 4,929 boxes) | precision | recall | panel-count MAE | signed mean | signed min, max |
|---|---|---|---|---|---|
| 0.1 | 0.9400 | 0.9720 | 3.84 | +3.29 | -3, +116 |
| 0.25 | 0.9703 | 0.9604 | 1.88 | -0.98 | -11, +6 |
| 0.5 | 0.9893 | 0.9359 | 5.61 | -5.22 | -130, +2 |

Ultralytics' own precision / recall at its max-F1 confidence: 0.9751 / 0.9576. No operating point
is chosen here.

**Reconciled, three ways.** The from-scratch 101-point AP equals torchmetrics' at both AP@0.5
and AP@0.5:0.95. Ultralytics differs for two reasons, each measured by capturing its validator's
internal boxes and scoring them with `ap.match`:

| boxes (val, 51 images, 4,929 boxes) | TP at IoU 0.5 | AP@0.5, 101-point | AP@0.5, Ultralytics' trapezoid | AP@0.5:0.95, 101-point | AP@0.5:0.95, Ultralytics' trapezoid |
|---|---|---|---|---|---|
| Ultralytics' internal (letterboxed, not clipped) | 4,824 | 0.968584 | **0.973270** | 0.818904 | **0.822093** |
| `predictions.json` (clipped to the 640 x 512 frame) | 4,836 | **0.973185** | 0.977917 | **0.824685** | 0.827932 |

Bold values reproduce the reported numbers exactly. (1) Ultralytics computes its metrics on boxes
that are not clipped to the frame; `predictions.json` clips them, and 5,079 of the 14,056 saved
boxes touch a frame edge, so edge panels gain IoU and 12 more pass 0.5. (2) Ultralytics
integrates a linearly interpolated 101-point curve with the trapezoid rule, which lands near the
all-point value, not COCO's step. On these predictions the two effects nearly cancel at AP@0.5,
which is why Ultralytics' number sits 8.5e-5 from the other two; that closeness is a coincidence,
not agreement. Matching rule is not a factor here: Ultralytics' IoU-greedy rule and the score-greedy
rule give the same 4,836 TPs on the json boxes.

**`max_det` binds on one frame.** The val frame with 584 ground-truth boxes
(`DJI_20230213154730_0155`) returns exactly 1,000 detections at `conf 0.001`, the lowest kept at
0.00388, so its tail below that score is truncated. It does not touch any confidence threshold in
the table above; its effect on AP is limited to that tail and was not measured.

### Re-evaluation of the same checkpoint at `max_det 3000`

Same checkpoint (`det_20261022T073325Z/weights/best.pt`), same pass settings except the override
`max_det=3000`, `scripts/eval_val_det.py`. Val, 51 images, 4,929 boxes.

| run | git SHA | dirty | what ran | detections | on `DJI_20230213154730_0155` (584 boxes) | max per image |
|---|---|---|---|---|---|---|
| det_val_20261022T074726Z | `d4ea608` | no | `max_det 1000` (the evaluation above) | 14,056 | 1,000 (the cap), lowest 0.00388 | 1,000 |
| det_val_20261022T105551Z | `28e4e6f` | no | `max_det=3000`, reference still capped at 1000 | 14,313 | 1,257, lowest 0.00100 | 1,257 |
| det_val_20261022T110900Z | `5b9b346` | no | `max_det=3000`, reference capped at `max_det` | 14,313 | 1,257, lowest 0.00100 | 1,257 |

| metric (val, 51 images, 4,929 boxes) | `max_det 1000` | `max_det 3000` |
|---|---|---|
| TP at IoU 0.5 | 4,836 | 4,849 |
| AP@0.5, from scratch 101-point | 0.973185 | 0.974546 |
| AP@0.5, torchmetrics | 0.973185 | 0.974546 |
| AP@0.5, from scratch all-point | 0.977807 | 0.979162 |
| AP@0.5:0.95, from scratch 101-point | 0.824685 | 0.824860 |
| AP@0.5:0.95, torchmetrics | 0.824685 | 0.824860 |
| AP@0.5 / AP@0.5:0.95, Ultralytics | 0.973270 / 0.822093 | 0.973270 / 0.822093 |

Precision, recall and panel-count error at conf 0.1, 0.25 and 0.5 are identical in both runs (the
extra 257 detections all score below 0.00388), so the operating-point table above stands.

Three things came out of it. **The cap was binding on AP**: the 257 extra detections on the dense
frame add 13 true positives at IoU 0.5. **The torchmetrics reference had its own cap**: the first
re-evaluation (`105551Z`) still read torchmetrics at 1000 detections per image, and showed a
from-scratch-minus-torchmetrics gap of +1.4e-3 at AP@0.5 and +1.8e-4 at AP@0.5:0.95 that was the
reference truncating its input. Fixed in `5b9b346` (the reference takes the evaluation's
`max_det`); the second re-evaluation agrees exactly. **Ultralytics' own numbers did not move** to
six decimals, although its `predictions.json` carries the 1,257 detections. Not explained; not
pursued, since no number taken forward depends on it. `max_det` was raised to 3000 in `19495cc`.

## Detection tuning pass, declared before the first arm ran

Written here, and in `configs/det_tuning.yaml`, **before any arm of the pass has run**. Nothing
below changes once the first number is in.

**1. Arms.** Everything as in `configs/det_yolo.yaml` at `19495cc` (`max_det 3000`, 100 epochs,
patience 20, imgsz 640, the declared augmentation), with two factors crossed and three seeds each:
`model` in {`yolov8n.pt`, `yolov8s.pt`} x `hsv_v` in {0.4, 0.0} x `seed` in {0, 1, 2}, twelve
training runs. `yolov8n.pt` with `hsv_v 0.4` is the committed config and is the **baseline**. The
`hsv_v` question: D2 frames are greyscale, so brightness jitter is the only colour augmentation that
acts, and in rendered thermal imagery brightness is the signal; `0.0` turns it off. Patience 20 stays,
so two runs can train for different numbers of epochs; that is the committed config's behaviour, kept
rather than changed for the pass.

**2. Batch.** 16 for both models. The rule was: 16 if a `yolov8s` smoke run at batch 16 peaks at or
below 6,500 MiB reserved with no out-of-memory error, otherwise `yolov8s` at batch 8 with the
confound named. The smoke run (`det_20261022T111038Z`, 2 epochs, at `86aed25`) peaked at 5,862 MiB
reserved, so model and batch do not change together.

**3. Score.** Each run's `best.pt` (selected by Ultralytics on its in-training val mAP50-95) is
evaluated by `scripts/eval_val_det.py`; the score is **val AP@0.5:0.95, 101-point** (equal to
torchmetrics'). Reported beside it and never used for selection: AP@0.5 (101-point) and panel-count
MAE at conf 0.25. Val therefore carries two jobs here, epoch selection inside each run and arm
selection across runs, and its numbers are optimistic by an unmeasured amount.

**4. Noise floor.** The largest three-seed standard deviation (ddof 1) of the score over the four
arms, rounded up to a multiple of 0.005, never below 0.005. Two arms whose mean scores differ by less
than it are reported as indistinguishable and are not ranked. This floor is computed from the pass
itself, because no earlier measurement of seed spread exists for this detector on this split; the
rule that computes it is what is fixed now.

**5. Selection rule.** Take the arm with the highest mean score. If it does not beat the baseline's
mean by more than the noise floor, keep the baseline. Each factor is judged the same way: the
selected arm takes the baseline's `model` and `hsv_v`, and changes a factor to the other level only
if (a) the arm with the highest mean has that level, and (b) the arm that differs from the baseline in
that factor alone beats the baseline's mean by more than the noise floor. Finally, if the arm this
gives has a mean more than the noise floor below the highest-mean arm's, the highest-mean arm is
taken instead. A difference exactly equal to the floor does not count as beating it. The selected arm's **seed 0** run is the model taken
forward, never its best seed, which would be selecting on noise. The rule is applied only if all
twelve runs completed; if any failed, the per-arm results are reported with their seed counts and
the decision goes to the next block rather than being made on fewer seeds.

*Added 22 Oct 2026, before any full-length arm ran* (only the 1-epoch smoke runs existed, and their
scores carry no information): the "Finally" clause above. Without it, when the effect is an
interaction the factor-by-factor rule can keep an arm clearly worse than the best one, for example
the baseline when only (`yolov8s.pt`, 0.0) beats it, or (`yolov8s.pt`, 0.4) when (`yolov8s.pt`, 0.0)
beats that by more than the floor.

**6. What comes after, also fixed now.** On the selected model, on val: the operating confidence is
the value on the grid 0.05, 0.10, ..., 0.95 that minimises panel-count MAE at NMS IoU 0.7; then the
NMS IoU threshold is swept from 0.3 to 0.8 in steps of 0.05 at that confidence, panel-count error
against threshold. Both are val-only and run in the next block. Then the final config is committed
and pushed, and the test split is evaluated once.

## Detection tuning pass: driver smoke

`python scripts/tune_det.py --smoke` at `9f774ae`, clean tree: the twelve runs of the pass, each at
`epochs=1`, in the declared seed-major order, each trained by `scripts/train_det.py` and scored on val
by `scripts/eval_val_det.py`. **Smoke, 1 epoch, scores not informative**: the run checks the driver's
code paths (order, logging, resume, the end-of-pass block) and the peak memory per model, nothing
else. A second invocation of the same command found all twelve complete and trained nothing (resume
check). Summary: `runs/det_tuning_smoke/summary.csv` and `summary.json`; logs in
`runs/det_tuning_smoke/logs/`. Val, 51 images, 4,929 boxes.

| arm | seed | train run (`det_`) | eval run (`det_val_`) | git SHA | dirty | wall, train + eval | peak VRAM reserved | score, val AP@0.5:0.95 101-point |
|---|---|---|---|---|---|---|---|---|
| yolov8n-hsv0.4 | 0 | 20261022T111621Z | 20261022T111722Z | `9f774ae` | no | 119 s | 4,462 MiB | 0.000158 |
| yolov8n-hsv0.0 | 0 | 20261022T111819Z | 20261022T111924Z | `9f774ae` | no | 134 s | 4,462 MiB | 0.000150 |
| yolov8s-hsv0.4 | 0 | 20261022T112033Z | 20261022T112132Z | `9f774ae` | no | 119 s | 5,862 MiB | 0.000082 |
| yolov8s-hsv0.0 | 0 | 20261022T112232Z | 20261022T112329Z | `9f774ae` | no | 115 s | 5,862 MiB | 0.000082 |
| yolov8n-hsv0.4 | 1 | 20261022T112427Z | 20261022T112521Z | `9f774ae` | no | 113 s | 4,854 MiB | 0.000107 |
| yolov8n-hsv0.0 | 1 | 20261022T112620Z | 20261022T112713Z | `9f774ae` | no | 109 s | 4,854 MiB | 0.000098 |
| yolov8s-hsv0.4 | 1 | 20261022T112809Z | 20261022T112904Z | `9f774ae` | no | 100 s | 5,304 MiB | 0.003317 |
| yolov8s-hsv0.0 | 1 | 20261022T112950Z | 20261022T113045Z | `9f774ae` | no | 101 s | 5,302 MiB | 0.003505 |
| yolov8n-hsv0.4 | 2 | 20261022T113131Z | 20261022T113223Z | `9f774ae` | no | 109 s | 4,810 MiB | 0.000098 |
| yolov8n-hsv0.0 | 2 | 20261022T113320Z | 20261022T113412Z | `9f774ae` | no | 109 s | 4,810 MiB | 0.000098 |
| yolov8s-hsv0.4 | 2 | 20261022T113510Z | 20261022T113606Z | `9f774ae` | no | 129 s | 5,260 MiB | 0.002100 |
| yolov8s-hsv0.0 | 2 | 20261022T113719Z | 20261022T113815Z | `9f774ae` | no | 126 s | 5,260 MiB | 0.002119 |

Whole pass 23.1 min against a printed estimate of 20 min. No out-of-memory error; the largest peak,
5,862 MiB, is the `yolov8s.pt` batch-16 figure of the batch check above. The end-of-pass block ran
(noise floor 0.005, every pair indistinguishable, baseline selected) under the rule as it stood at
`9f774ae`, before the final clause of item 5 was added; with these scores the amended rule gives the
same arm. None of it is a result.

## Detection tuning pass: result

`python scripts/tune_det.py` at `067f375`, clean tree, the twelve runs as declared above, all
`done`, none retried. Every score is val AP@0.5:0.95, 101-point (= torchmetrics), on val, 51 images,
4,929 boxes. Summary: `runs/det_tuning/summary.csv` and `summary.json`; logs in
`runs/det_tuning/logs/`. Wall is train + val evaluation per run; the twelve add up to 5,685 s
(94.8 min).

| arm | seed | train run (`det_`) | eval run (`det_val_`) | epochs run (best) | wall | peak VRAM reserved | score, AP@0.5:0.95 | AP@0.5, 101-point | count MAE at conf 0.25 |
|---|---|---|---|---|---|---|---|---|---|
| yolov8n-hsv0.4 | 0 | 20261022T163213Z | 20261022T163802Z | 100 (81) | 389 s | 5,394 MiB | 0.824860 | 0.974546 | 1.88 |
| yolov8n-hsv0.0 | 0 | 20261022T163843Z | 20261022T164156Z | 49 (29) | 274 s | 5,450 MiB | 0.812835 | 0.971613 | 4.98 |
| yolov8s-hsv0.4 | 0 | 20261022T164317Z | 20261022T165512Z | 100 (81) | 757 s | 6,640 MiB | 0.837309 | 0.979037 | 2.14 |
| yolov8s-hsv0.0 | 0 | 20261022T165554Z | 20261022T170932Z | 100 (83) | 851 s | 6,680 MiB | 0.835554 | 0.978592 | 2.41 |
| yolov8n-hsv0.4 | 1 | 20261022T171005Z | 20261022T171842Z | 100 (89) | 556 s | 6,044 MiB | 0.833398 | 0.977292 | 1.59 |
| yolov8n-hsv0.0 | 1 | 20261022T171920Z | 20261022T172436Z | 59 (39) | 383 s | 6,044 MiB | 0.819432 | 0.972280 | 3.18 |
| yolov8s-hsv0.4 | 1 | 20261022T172544Z | 20261022T173143Z | 51 (31) | 398 s | 7,092 MiB | 0.827413 | 0.974494 | 1.90 |
| yolov8s-hsv0.0 | 1 | 20261022T173221Z | 20261022T173803Z | 41 (21) | 392 s | 7,096 MiB | 0.821958 | 0.968575 | 2.88 |
| yolov8n-hsv0.4 | 2 | 20261022T173853Z | 20261022T174549Z | 84 (64) | 456 s | 5,606 MiB | 0.828999 | 0.975730 | 3.12 |
| yolov8n-hsv0.0 | 2 | 20261022T174629Z | 20261022T175233Z | 70 (50) | 432 s | 5,604 MiB | 0.825891 | 0.974726 | 2.14 |
| yolov8s-hsv0.4 | 2 | 20261022T175341Z | 20261022T175936Z | 58 (38) | 394 s | 6,568 MiB | 0.833746 | 0.978193 | 2.39 |
| yolov8s-hsv0.0 | 2 | 20261022T180015Z | 20261022T180610Z | 42 (22) | 403 s | 6,402 MiB | 0.830430 | 0.978190 | 2.57 |

Per arm, three seeds, mean and standard deviation (ddof 1); recomputed from `summary.csv`
independently of the driver and equal to its `summary.json` `result`:

| arm (val, 51 images, 4,929 boxes) | n | score mean | score std | AP@0.5 mean (std) | count MAE at 0.25 mean (std) |
|---|---|---|---|---|---|
| yolov8n-hsv0.4 (baseline) | 3 | 0.829086 | 0.004270 | 0.975856 (0.001378) | 2.20 (0.81) |
| yolov8n-hsv0.0 | 3 | 0.819386 | 0.006528 | 0.972873 (0.001639) | 3.43 (1.44) |
| yolov8s-hsv0.4 | 3 | 0.832823 | 0.005012 | 0.977241 (0.002416) | 2.14 (0.25) |
| yolov8s-hsv0.0 | 3 | 0.829314 | 0.006866 | 0.975119 (0.005671) | 2.62 (0.24) |

**Noise floor 0.010**: the largest std, 0.006866 (yolov8s-hsv0.0), rounded up to a multiple of 0.005.

| pair, difference of mean scores | call |
|---|---|
| yolov8n-hsv0.4 - yolov8n-hsv0.0: +0.009700 | indistinguishable |
| yolov8n-hsv0.4 - yolov8s-hsv0.4: -0.003737 | indistinguishable |
| yolov8n-hsv0.4 - yolov8s-hsv0.0: -0.000229 | indistinguishable |
| yolov8n-hsv0.0 - yolov8s-hsv0.4: -0.013437 | yolov8s-hsv0.4 above |
| yolov8n-hsv0.0 - yolov8s-hsv0.0: -0.009928 | indistinguishable |
| yolov8s-hsv0.4 - yolov8s-hsv0.0: +0.003508 | indistinguishable |

**Selected: yolov8n-hsv0.4, the baseline**, by the first clause of rule 5: the highest-mean arm,
yolov8s-hsv0.4, beats the baseline's mean by 0.003737, not by more than the floor of 0.010, so the
baseline is kept. (The factor clause adopts nothing: the single change to `yolov8s.pt` is that same
arm; the best arm has `hsv_v` 0.4, the baseline's level. The final clause does not apply: the gap to
the best arm is the same 0.003737.) The model taken forward is the baseline's **seed 0 run,
`det_20261022T163213Z`**, evaluated in `det_val_20261022T163802Z`. Its val evaluation is identical, down
to every box and score in `predictions.npz`, to `det_val_20261022T110900Z`, the `max_det 3000` evaluation of the first
fine-tune `det_20261022T073325Z` (same config, same seed): at seed 0 the training reproduced.
`configs/det_yolo.yaml` needs no change of `model` or `hsv_v`.

Read the other way, the pass is a measured null result: at three seeds the larger model and turning
off brightness jitter move the score by less than the seed-to-seed spread, with one exception, the
gap between the best and worst arms (yolov8s-hsv0.4 over yolov8n-hsv0.0, 0.013).

**Stopped early (patience 20)**, 8 of 12 runs: yolov8n-hsv0.0 at all three seeds (49, 59, 70 epochs),
yolov8n-hsv0.4 seed 2 (84), yolov8s-hsv0.4 seeds 1 and 2 (51, 58), yolov8s-hsv0.0 seeds 1 and 2
(41, 42). The yolov8s runs of seeds 1 and 2 kept a best epoch between 21 and 38. Patience was
declared unchanged for the pass, so this is part of what each arm is; it is also why the wall times
differ by up to a factor of three within an arm.

**Peak memory:** the yolov8s runs peaked at up to 7,096 MiB reserved, about 1,200 MiB above the
2-epoch batch check (5,862 MiB), as the yolov8n full run had done relative to its smoke. No run ran
out of memory; 8,188 MiB is the card's total.

## Detection operating confidence, on val

Declaration item 6, first half. `scripts/operating_point_det.py` on the selected model's val
evaluation `det_val_20261022T163802Z` (checkpoint `det_20261022T163213Z`, the tuning pass's selected seed-0 run), no new
inference: the boxes of the pass at conf 0.001, NMS IoU 0.7, `max_det` 3000 (largest frame 1,257
detections, so no frame was truncated) filtered by score at each grid value. Run
`det_oppoint_20261022T193813Z`, from `21a945a`, clean tree. Figure (count MAE and signed mean against
confidence): `runs/det_oppoint_20261022T193813Z/count_error_vs_conf.png`. Val, 51 images, 4,929
boxes; precision and recall at IoU 0.5; signed error is predicted minus ground-truth count.

| conf (val, 51 images, 4,929 boxes) | count MAE | signed mean | signed min, max | precision | recall |
|---|---|---|---|---|---|
| 0.05 | 6.373 | +6.255 | -2, +187 | 0.9162 | 0.9755 |
| 0.10 | 3.843 | +3.294 | -3, +116 | 0.9400 | 0.9720 |
| 0.15 | 2.824 | +1.569 | -6, +68 | 0.9543 | 0.9698 |
| 0.20 | 2.039 | +0.039 | -7, +20 | 0.9649 | 0.9653 |
| **0.25** | **1.882** | -0.980 | -11, +6 | 0.9703 | 0.9604 |
| 0.30 | 2.431 | -1.647 | -23, +6 | 0.9742 | 0.9576 |
| 0.35 | 3.137 | -2.392 | -45, +5 | 0.9782 | 0.9539 |
| 0.40 | 4.000 | -3.333 | -74, +4 | 0.9813 | 0.9475 |
| 0.45 | 4.843 | -4.333 | -104, +2 | 0.9860 | 0.9418 |
| 0.50 | 5.608 | -5.216 | -130, +2 | 0.9893 | 0.9359 |
| 0.55 | 6.745 | -6.510 | -167, +2 | 0.9924 | 0.9255 |
| 0.60 | 8.176 | -8.020 | -218, +1 | 0.9960 | 0.9134 |
| 0.65 | 9.902 | -9.863 | -272, +1 | 0.9966 | 0.8949 |
| 0.70 | 11.647 | -11.647 | -317, 0 | 0.9972 | 0.8771 |
| 0.75 | 14.647 | -14.647 | -378, 0 | 0.9978 | 0.8466 |
| 0.80 | 19.314 | -19.314 | -430, -1 | 0.9980 | 0.7985 |
| 0.85 | 31.941 | -31.941 | -485, -2 | 0.9991 | 0.6689 |
| 0.90 | 59.706 | -59.706 | -544, -5 | 1.0000 | 0.3822 |
| 0.95 | 94.275 | -94.275 | -582, -12 | 1.0000 | 0.0245 |

**Operating confidence 0.25**, the unique minimum of count MAE (no tie). The rows at 0.10, 0.25 and
0.50 equal the evaluation's own at-confidence block. Over-counting below it, under-counting above
it. The extremes away from 0.25 are the 584-box frame `DJI_20230213154730_0155` (+187 at 0.05, -130
at 0.50); at 0.25 the worst frame is 11 short of 212. The curve is flat between 0.20 and 0.25 in MAE (2.04, 1.88) but not in sign: 0.20 is
unbiased on average (+0.04), 0.25 slightly under (-0.98). The rule picks on MAE alone, as declared.
Chosen on val, which has now also selected the epoch and the arm.

## Detection NMS IoU sweep, on val

Declaration item 6, second half: reported, not used to choose. **`iou` stays 0.7.** Selected model
`det_20261022T163213Z`, conf 0.25 (the operating confidence above), `max_det` 3000, one Ultralytics
validation pass per threshold through `evaluate()`: `scripts/nms_sweep_det.py`, run
`det_nms_sweep_20261022T194018Z`, from `aabbd61`, clean tree. The detections a pass returns are the
panels it counts. Figure (count MAE and signed mean against NMS IoU):
`runs/det_nms_sweep_20261022T194018Z/count_error_vs_nms_iou.png`. Val, 51 images, 4,929 boxes.

| NMS IoU (val, 51 images, 4,929 boxes) | count MAE | signed mean | signed min, max | detections | frames whose count differs from 0.7 | count MAE without the 584-box frame | that frame's signed error |
|---|---|---|---|---|---|---|---|
| 0.30 | 3.490 | -3.216 | -81, +3 | 4,765 | 16 | 1.94 | -81 |
| 0.35 | 3.275 | -2.922 | -75, +3 | 4,780 | 14 | 1.84 | -75 |
| 0.40 | 3.157 | -2.765 | -70, +3 | 4,788 | 13 | 1.82 | -70 |
| 0.45 | 3.039 | -2.647 | -65, +3 | 4,794 | 13 | 1.80 | -65 |
| 0.50 | 2.922 | -2.373 | -58, +3 | 4,808 | 12 | 1.82 | -58 |
| 0.55 | 2.784 | -2.039 | -48, +4 | 4,825 | 8 | 1.88 | -48 |
| 0.60 | 2.471 | -1.725 | -34, +4 | 4,841 | 8 | 1.84 | -34 |
| 0.65 | 2.235 | -1.490 | -25, +4 | 4,853 | 5 | 1.78 | -25 |
| **0.70** | **1.882** | -0.980 | -11, +6 | 4,879 | 0 | 1.82 | -5 |
| 0.75 | 2.176 | +0.020 | -8, +22 | 4,930 | 16 | 1.78 | +22 |
| 0.80 | 3.412 | +1.529 | -7, +75 | 5,007 | 24 | 1.98 | +75 |

The 0.70 row equals the operating-point table's 0.25 row (a pass at conf 0.25 against filtering the
conf 0.001 pass at 0.25), which checks the filtering argument of `operating_point_det.py` on counts.

**Against section 6.2's claim.** On 50 of the 51 frames the claim holds as measured: count MAE stays
between 1.78 and 1.98 from 0.30 to 0.80. The whole shape of the curve is one frame,
`DJI_20230213154730_0155` (584 panels, median box 25 x 12 px against 54 x 31 px on the other frames),
whose count runs from 81 short at 0.30 to 75 over at 0.80. Its ground-truth boxes barely overlap
(nearest-neighbour IoU median 0.059, maximum 0.29), but its predicted boxes do: of the boxes kept at
0.70, 29.5% overlap another kept box at IoU > 0.3, against 2.0% on the other 50 frames. So the
edge-sharing premise is true of the labels everywhere and of the predictions on normal frames; on
small, densely packed panels a few pixels of localisation error is a large fraction of the box, and
the predicted boxes of neighbours overlap enough that a low threshold suppresses real panels and a high
one keeps duplicates. On the whole of val, ground-truth neighbours are not at IoU exactly 0 either:
99.5% of the 4,929 boxes overlap their nearest neighbour a little (median 0.048), 26 pairs exceed 0.3,
none 0.5.

### Cross-check: the from-scratch NMS on the same detections

`scripts/nms_crosscheck_det.py`, run `det_nms_xcheck_20261022T195450Z`, from `8d8ab7a`, clean tree,
about 10 minutes of the 45-minute cap. The boxes before NMS are copied from inside one Ultralytics
validation pass at conf 0.25 by wrapping `non_max_suppression` (the exact tensor the library
suppresses: letterboxed, not clipped, the model's own precision, candidates with score strictly
above conf as the library filters); `solar_inspect.detection.nms.nms` is applied to each image's
candidates and its per-image count compared with the sweep's Ultralytics pass at the same threshold.
Ultralytics' NMS here is `torchvision.ops.nms`. 40,488 candidates over the 51 val images.

| NMS IoU (val, 51 images) | from scratch, detections | Ultralytics, detections | images with equal counts |
|---|---|---|---|
| 0.3 | 4,765 | 4,765 | 51 / 51 |
| 0.5 | 4,808 | 4,808 | 51 / 51 |
| 0.7 | 4,879 | 4,879 | 51 / 51 |

Equal on every image at all three thresholds: the from-scratch NMS reproduces the library's on real
detections. The first run of the script (`det_nms_xcheck_20261022T194625Z`, at `2ccc4b6`) stopped
before comparing: the validator calls NMS once before its loop, which shifted the pairing of captured
inputs to batches by one and matched 36 of 51 images; fixed in `8d8ab7a`.

## Detection final config, before the test split is read

`configs/det_yolo.yaml` as committed with this entry: `model` and `hsv_v` unchanged (the tuning pass
kept the baseline), `conf 0.001`, `iou 0.7` and `max_det 3000` unchanged, `conf_operating 0.25`
added (chosen on val above), and `checkpoint_sha256
1bd7908cc646b5a4b0af5eaa23bd69cbc97b7a593af9c459cc19f71f9419976a`, the SHA-256 of
`runs/det_20261022T163213Z/weights/best.pt`, the selected arm's seed 0 run. `scripts/eval_test_det.py`
refuses any checkpoint with another hash (the first fine-tune's `best.pt`, which gives identical val
predictions, hashes to `583beee52ffe...` and is refused), as well as configs not in `origin/main` and
a second test evaluation.

## D2 detection: THE TEST EVALUATION, once, and not again

**Config commits: `configs/det_yolo.yaml` [`83e51cb`](../../../commit/83e51cb), `configs/d2.yaml`
[`9b95550`](../../../commit/9b95550)**, both in `origin/main` before `scripts/eval_test_det.py` read
the test split; the script refuses otherwise, and printed both SHAs. Checkpoint
`runs/det_20261022T163213Z/weights/best.pt`, SHA-256
`1bd7908cc646b5a4b0af5eaa23bd69cbc97b7a593af9c459cc19f71f9419976a`, equal to the config's
`checkpoint_sha256` (the script refuses any other file). Run `det_test_20261022T200544Z`, from
`83e51cb`, clean tree, one Ultralytics pass at conf 0.001, NMS IoU 0.7, `max_det` 3000: 25,637
detections, at most 1,504 on one frame, so no frame reached the cap. Test is sortie 4 of the sortie
split (ADR 0002); val is sortie 2.

Intervals: percentile bootstrap over images (63 test images resampled with replacement, 2,000
resamples, seed 0) from this run's saved predictions only, `scripts/bootstrap_test_det.py`, run
`det_boot_20261022T200914Z` from `89a40d2`. It checks that its per-image matching reproduces every
point value below exactly. (A first run, `det_boot_20261022T200836Z` from `a50f1db`, computed only the
two APs and the MAE; same draws, same intervals for those three.)

| metric | **test, 63 images, 3,984 boxes** | 95% interval | val, same model, 51 images, 4,929 boxes |
|---|---|---|---|
| AP@0.5, 101-point (= torchmetrics) | **0.7552** | [0.6418, 0.8438] | 0.9745 |
| AP@0.5:0.95, 101-point (= torchmetrics) | **0.5578** | [0.4675, 0.6374] | 0.8249 |
| precision at conf 0.25 (operating), IoU 0.5 | **0.6025** | [0.4925, 0.6872] | 0.9703 |
| recall at conf 0.25 (operating), IoU 0.5 | **0.7899** | [0.6969, 0.8637] | 0.9604 |
| panel-count MAE at conf 0.25 (operating) | **25.48** | [19.65, 31.94] | 1.88 |
| panel-count signed mean, min, max at 0.25 | +19.67, -100, +105 | | -0.98, -11, +6 |

Beside the headline, test, 63 images, 3,984 boxes: AP@0.5 all-point **0.7571**; Ultralytics' AP@0.5
**0.7562** and AP@0.5:0.95 **0.5569**. Ultralytics differs for the two reasons measured on val (it
scores letterboxed boxes not clipped to the frame, and integrates its 101-point curve with the
trapezoid rule); not re-measured on test, which would need its internal boxes from a second pass.
From scratch minus torchmetrics: 0 at AP@0.5, 2e-16 at AP@0.5:0.95.

The other two fixed confidences, test, 63 images, 3,984 boxes (no intervals; reported, not chosen):

| conf | precision | recall | count MAE | signed mean | signed min, max |
|---|---|---|---|---|---|
| 0.10 | 0.5047 | 0.8429 | 45.06 | +42.37 | -83, +190 |
| 0.25 | 0.6025 | 0.7899 | 25.48 | +19.67 | -100, +105 |
| 0.50 | 0.7268 | 0.7126 | 14.10 | -1.24 | -154, +52 |

The three test frames with more than 300 boxes, at conf 0.25: `DJI_20230213164450_0091` 512 boxes,
456 counted (**-56**); `DJI_20230213164408_0083` 418, 462 (**+44**); `DJI_20230213164426_0087` 331,
364 (**+33**).

**What the gap says.** Test is far below val on every number, and the intervals do not reach the val
values. Val did three jobs (epoch, arm and operating point), and its numbers were optimistic by an
amount that is now measured, not small. The operating point did not transfer: on test the count error
at 0.25 is mostly over-counting (+19.7 panels per frame on average), and 0.50 has the lower count MAE
there. That is a fact about this test set, not a reason to change `conf_operating`: choosing it on
test would make the test number a val number.

One measurement made afterwards, from labels and these saved predictions, without a model pass: how
much neighbouring ground-truth boxes overlap (median nearest-neighbour IoU per box). Train 0.222
(128 of 138 frames with a median above 0.2), val **0.048** (14 of 51), test 0.148 (27 of 63). Rows that
run diagonally across the frame make axis-aligned boxes overlap; val, the split that selected
everything, is the one where they do least. Train has more of it than test, so this does not by itself
explain the drop; the cause is not established here.

## D2 detection: post-hoc error analysis of the test evaluation

Thu 22 Oct 2026, `scripts/error_analysis_det.py` at [`d9fbb14`](../../../commit/d9fbb14), clean tree,
run `det_posthoc_20261022T205533Z` (deterministic; a development run of the same code gave identical
output). **Post-hoc: made after the single test evaluation; nothing here changes the model, the
config, the operating point or the reported numbers.**

Inputs only: the test evaluation `det_test_20261022T200544Z` and the selected model's val evaluation
`det_val_20261022T163802Z` (`predictions.npz`, `eval.json`), and the D2 labels and images. No model
pass on any split. Shipped settings: the passes ran at NMS IoU 0.7, detections kept at conf 0.25;
true and false positives from `ap.match` at IoU 0.5 as `evaluate()` takes them. The script checks
its totals equal the evaluations' own: test precision 0.6025, recall 0.7899, count MAE 25.48, signed
mean +19.67; val 0.9703, 0.9604, 1.88, -0.98. It reproduces the neighbour-IoU medians measured
earlier (0.222, 0.048, 0.148).

### False positives by class

Rule, applied in this order (`detection/errors.py`, tested on hand-made frames): **duplicate**, IoU >
0.3 with a true positive of the same frame; **localisation**, best IoU with any ground-truth box in
[0.1, 0.5); **background**, best IoU with every ground-truth box < 0.1. A false positive whose best
ground-truth IoU is >= 0.5 lost that box to a higher-scoring detection; if it also overlaps no true
positive by more than 0.3 it passes none of the three tests and is counted as **unclassified**.

| conf 0.25, IoU 0.5 | test, 63 images, 3,984 boxes | val, 51 images, 4,929 boxes |
|---|---|---|
| detections kept / TP / FP / FN | 5,223 / 3,147 / 2,076 / 837 | 4,879 / 4,734 / 145 / 195 |
| duplicate | 794 (38.2%) | 101 (69.7%) |
| localisation | 1,251 (60.3%) | 29 (20.0%) |
| background | 22 (1.1%) | 15 (10.3%) |
| unclassified | 9 (0.4%) | 0 |

Per frame, signed count error (counted - labelled) is FP - FN; the script checks this on every
frame. Mean per frame:

| mean per frame | test | val |
|---|---|---|
| duplicate | +12.60 | +1.98 |
| localisation | +19.86 | +0.57 |
| background | +0.35 | +0.29 |
| unclassified | +0.14 | 0 |
| all FP | +32.95 | +2.84 |
| FN (missed) | -13.29 | -3.82 |
| **signed count error** | **+19.67** | **-0.98** |

Test's +19.67 is localisation (+19.86) and duplicates (+12.60), partly offset by misses (-13.29);
background false positives add +0.35. The classes are by IoU with the labels, so they do not separate
causes cleanly: a detection on an unlabelled panel that sits next to a labelled one can overlap that
neighbour and land in localisation or duplicate, not background.

### Misses

| unmatched ground truth at conf 0.25 | test | val |
|---|---|---|
| missed boxes | 837 | 195 |
| with a kept detection at IoU [0.1, 0.5) (found, badly placed) | 56.4% | 49.7% |
| no kept detection at IoU >= 0.1 | 43.4% | 49.2% |
| best kept detection at IoU >= 0.5 (it matched another box) | 0.2% | 1.0% |
| touching the frame edge (within 1 px) | 25.6% | 59.0% |
| share of all labels touching the edge | 27.2% | 22.9% |
| miss rate, edge boxes / interior boxes | 19.7% / 21.5% | 10.2% / 2.1% |

Area terciles over all 19,525 D2 labels (train, val, test): cut at 906 and 3,251 px².

| miss rate by area tercile | small (< 906 px²) | middle | large (> 3,251 px²) |
|---|---|---|---|
| test | 16.0% (of 1,466) | 18.5% (of 990) | 27.4% (of 1,528) |
| val | 14.1% (of 1,323) | 0.1% (of 3,132) | 1.1% (of 474) |

On val the misses are small boxes and edge boxes. On test they are spread over the frame and are
most frequent among the largest boxes.

### What differs between the sorties, labels and pixels only

Medians with the interquartile range in brackets. Box sizes in pixels on the 640 x 512 frames.
Neighbour IoU: per box, its largest IoU with another labelled box of the same frame (boxes alone in
their frame left out). DN: the 8-bit rendered value, uncalibrated; per frame, its mean, its standard
deviation, and the mean DN inside the labelled boxes minus outside them (an intra-frame contrast in
DN, not a physical quantity).

| | train (sorties 1, 3, 5) | val (sortie 2) | test (sortie 4) |
|---|---|---|---|
| frames, boxes | 138, 10,612 | 51, 4,929 | 63, 3,984 |
| box width, px | 59.9 [32.3, 73.4] | 52.0 [39.7, 63.3] | 46.8 [29.7, 78.7] |
| box height, px | 46.0 [23.3, 57.1] | 30.3 [23.1, 35.8] | 40.5 [17.4, 58.2] |
| box area, px² | 2,603 [755, 4,128] | 1,497 [821, 2,285] | 1,864 [553, 4,556] |
| aspect ratio w/h | 1.32 [1.24, 1.43] | 1.87 [1.57, 1.99] | 1.67 [1.00, 1.95] |
| boxes per frame, median (range) | 66 (0 to 375) | 87 (12 to 584) | 44 (13 to 512) |
| neighbour IoU, median | 0.222 | 0.048 | 0.148 |
| neighbour IoU > 0.3, share of boxes | 5.4% | 1.0% | 5.9% |
| frames with median neighbour IoU > 0.2 | 128 of 138 | 14 of 51 | 27 of 63 |
| frame DN mean | 120.8 [110.1, 129.5] | 131.9 [124.5, 136.5] | 115.6 [109.3, 118.4] |
| frame DN std | 53.9 [50.3, 56.8] | 55.0 [53.0, 58.0] | 51.9 [49.1, 55.8] |
| DN inside boxes minus outside | 80.8 [74.0, 87.2] | 94.3 [87.2, 102.5] | 77.6 [67.0, 87.1] |

On these marginal statistics test is not outside train's range: train's boxes are larger, squarer
and more overlapped than test's. Val is the split that differs most from both, with the most
elongated boxes, the least neighbour overlap and the highest inside-minus-outside DN.

### Per-frame correlations

Spearman correlation over frames of per-frame AP@0.5 (101-point, the frame alone) and of per-frame
signed count error at conf 0.25 with five frame properties. All twenty are reported. Declared reading:
"clear" means p < 0.005, i.e. 0.05 over the ten correlations of a split.

| test, 63 frames | AP@0.5 rho (p) | signed count error rho (p) |
|---|---|---|
| median box area | -0.29 (0.020) | +0.03 (0.84) |
| boxes per frame | +0.20 (0.11) | +0.00 (0.99) |
| median neighbour IoU | -0.27 (0.036) | +0.20 (0.12) |
| median aspect ratio w/h | **+0.57 (1.1e-6), clear** | **-0.49 (5.4e-5), clear** |
| DN inside minus outside | +0.15 (0.25) | +0.01 (0.91) |

| val, 51 frames | AP@0.5 rho (p) | signed count error rho (p) |
|---|---|---|
| median box area | +0.25 (0.078) | +0.28 (0.046) |
| boxes per frame | -0.32 (0.022) | -0.32 (0.022) |
| median neighbour IoU | +0.12 (0.39) | +0.28 (0.045) |
| median aspect ratio w/h | -0.22 (0.12) | **-0.45 (0.00097), clear** |
| DN inside minus outside | -0.36 (0.0098) | -0.13 (0.38) |

Three of the twenty are clear, all with aspect ratio: on test, frames whose boxes are squarer have
lower AP and more over-counting; on val, more over-counting. The rest are not clear at 63 and 51
frames. The properties are correlated with each other, so the aspect ratio is not separable from
them: on test, aspect ratio against neighbour IoU -0.75, against the contrast +0.47, against box area
-0.31 (box area against boxes per frame -0.75). Squarer axis-aligned boxes are what rows crossing the
frame diagonally produce, so the clear correlations are consistent with row orientation and not
separable, from these outputs, from the neighbour overlap that comes with it.

### The frames with the largest count error

| test frame | labelled | counted | error | TP | dup | loc | bg | uncl | FN | AP@0.5 | median area px² | aspect |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `DJI_20230213164043_0066` | 108 | 213 | +105 | 95 | 73 | 44 | 0 | 1 | 13 | 0.658 | 1,947 | 0.99 |
| `DJI_20230213164649_0100` | 112 | 12 | -100 | 12 | 0 | 0 | 0 | 0 | 100 | 0.268 | 2,309 | 0.74 |
| `DJI_20230213164057_0068` | 81 | 178 | +97 | 40 | 34 | 104 | 0 | 0 | 41 | 0.191 | 1,936 | 0.77 |
| `DJI_20230213164125_0071` | 42 | 137 | +95 | 37 | 53 | 47 | 0 | 0 | 5 | 0.410 | 3,586 | 1.09 |
| `DJI_20230213164055_0067` | 90 | 165 | +75 | 82 | 50 | 32 | 0 | 1 | 8 | 0.691 | 2,849 | 1.02 |
| `DJI_20230213164450_0091` | 512 | 456 | -56 | 381 | 56 | 15 | 4 | 0 | 131 | 0.798 | 278 | 2.15 |

Five of the six have a median aspect ratio between 0.74 and 1.09 (test's median is 1.67); the sixth
is the densest test frame, with the smallest boxes. Four are consecutive or nearly consecutive frames
(0066 to 0071). Overlays of these six and of the two val frames with the largest error (-11, -7), a
contact sheet, and the per-frame numbers of all 63 test frames are in
`runs/det_posthoc_20261022T205533Z/` (not in git). Whether the background boxes are unlabelled
panels is a visual question these numbers do not answer.

### Visual check and the orientation tail

Thu 22 Oct 2026, `scripts/error_analysis_det.py` at [`5b1872b`](../../../commit/5b1872b), clean tree,
run `det_posthoc_20261022T214418Z`. **Post-hoc: made after the single test evaluation; nothing here
changes the model, the config, the operating point or the reported numbers.** Every number of the
first post-hoc run above is reproduced exactly by this run (the per-frame CSV is byte-identical).

**Visual check.** The eight overlays of `det_posthoc_20261022T205533Z` were looked at in review,
after the first run. What they show:

- The labels cover the visible panels in all six test frames. Background (red) boxes appear only on
  the densest frame (4 of them). There is no sign of unlabelled panels, so missing labels are not
  supported as an explanation of the over-count.
- Five of the six test frames are close views of rows that cross the frame steeply, near vertical.
  There the axis-aligned boxes of neighbouring modules overlap heavily, in the labels as in the
  predictions. Four of them (0066 to 0071) over-fire, with stacks of overlapping boxes along each row
  (duplicate and localisation false positives). Frame 0100, with the same geometry, under-fires: 12
  of 112 panels counted.
- The two val frames and the densest test frame show rows running across the frame nearly
  horizontally, with small boxes that do not overlap.

The medians in "What differs between the sorties" compare the middles of the distributions. The
overlays point at a tail: test's lower aspect quartile is 1.00 against train's 1.24, boxes taller than
wide, which is what a steep row gives. The tail is measured below.

**Declared before the measurement:** a box is *steep* when its aspect ratio w/h is below 1.1; a frame
is steep when the median aspect ratio of its boxes is below 1.1. The 1.1 is the value the overlays
suggested, fixed as a constant in the script before this ran, not tuned afterwards. Area terciles as
above (906 and 3,251 px², over all 19,525 labels).

| labels only | train | val | test |
|---|---|---|---|
| boxes with aspect w/h < 1.0 | 9.9% | 8.0% | 25.1% |
| boxes with aspect w/h < 1.1 (steep) | 12.8% | 9.8% | 27.6% |
| steep frames (median box aspect < 1.1) | **0 of 136** | **0 of 51** | **22 of 63** |
| lowest per-frame median aspect | 1.22 | 1.22 | 0.72 |

(Train's 136 frames are the 138 less its two empty ones.)

| labels by steepness and area tercile | small | middle | large |
|---|---|---|---|
| train, steep | 529 | 630 | **196** |
| train, not steep | 3,190 | 1,757 | 4,310 |
| val, steep | 293 | 124 | 65 |
| val, not steep | 1,030 | 3,008 | 409 |
| test, steep | 159 | 369 | **570** |
| test, not steep | 1,307 | 621 | 958 |

Steep boxes exist in train (1,355 of 10,612), but no train frame is dominated by them, and large
steep boxes are 196 of 10,612 train labels (1.8%) against 570 of 3,984 test labels (14.3%).

The test evaluation split by frame geometry, from the saved predictions (conf 0.25, NMS IoU 0.7,
IoU 0.5; AP@0.5 101-point pooled over the group's frames). Val has no steep frame, so it has only the
second row.

| | frames | labelled | AP@0.5 | precision | recall | count MAE | signed mean |
|---|---|---|---|---|---|---|---|
| test, steep frames | 22 | 1,009 | **0.2253** | 0.2391 | 0.4470 | 48.95 | +39.86 |
| test, other frames | 41 | 2,975 | **0.9223** | 0.8079 | 0.9062 | 12.88 | +8.83 |
| val, all frames (none steep) | 51 | 4,929 | 0.9745 | 0.9703 | 0.9604 | 1.88 | -0.98 |

The 22 steep frames carry 877 of the 1,239 panels of test's net over-count (71%; 22 x 39.86 and
63 x 19.67). The other 41 test frames are still below val (AP@0.5 0.9223 against 0.9745, count MAE
12.88 against 1.88), so the steep frames do not account for the whole gap.

| test miss rate (missed / labelled) | small | middle | large |
|---|---|---|---|
| steep boxes | 26.4% (42 / 159) | 35.5% (131 / 369) | **68.9% (393 / 570)** |
| not steep | 14.8% (193 / 1,307) | 8.4% (52 / 621) | 2.7% (26 / 958) |

(Val, for reference: steep 18.8%, 1.6%, 1.5% of 293, 124, 65; not steep 12.8%, 0.03%, 1.0% of 1,030,
3,008, 409.) On test the rise of the miss rate with box size seen above is a property of the steep
boxes: large steep boxes are missed 68.9% of the time, large other boxes 2.7%.

Reading, from these numbers only: test's drop is concentrated in frames whose rows cross the frame
steeply, a geometry no training frame has and whose large boxes are rare in train. That is consistent
with a change in row orientation relative to the camera between sorties, and not separable from the
neighbour overlap that comes with it. The remaining gap on the other 41 frames is not explained here.

## D5 flyover and trackers: build

Fri 23 Oct 2026. Code: box Kalman filter [`f4722ab`](../../../commit/f4722ab),
[`97b7e66`](../../../commit/97b7e66), [`a8844cb`](../../../commit/a8844cb); SORT, greedy IoU and
no tracker [`8b848bb`](../../../commit/8b848bb); generator [`4afc1ab`](../../../commit/4afc1ab);
`configs/flyover.yaml` [`5a1d4b6`](../../../commit/5a1d4b6), committed before any sequence was
generated; `scripts/make_flyover.py` [`051e111`](../../../commit/051e111);
`scripts/dedup_flyover.py` [`378ecbe`](../../../commit/378ecbe).

**D5 is synthetic.** Each sequence is a 320 x 256 window moving for 60 frames over one D2 val frame
(random-walk velocity reflecting off the frame bounds, yaw ramp 0.15 deg per frame, scale ramp
across [0.85, 1.15]), with an opaque band drifting over the ground to occlude panels. Detections are
the shipped detector's (checkpoint SHA-256 `1bd7908cc646...`, conf 0.25, NMS IoU 0.7, max_det 3000,
imgsz 320 so the window is seen at the trained pixel scale). Development seeds 0-9 and held-out
seeds 100-119 use 30 different val frames.

Run `flyover_20261022T221314Z` at `051e111`, clean tree, 30 sequences in 19 s. A first run of the
same code before its commit gave byte-identical arrays for all 30, detections included.

Checks on the development sequences:

- Visible fractions (share of a panel's warped area inside the window and not under the band)
  against a 4x supersampled pixel mask, seeds 0 and 2, 930 panel-frames: largest difference 0.022 of
  the panel's area.
- Detection boxes against ground-truth boxes: median centre offset (-0.11, +0.20) px; 99.7% of the
  7,320 visible ground-truth boxes have a detection at IoU >= 0.5.
- 11,874 detections against 7,320 visible ground-truth boxes. 61.9% of detections overlap a visible
  ground-truth box at IoU >= 0.5 and 88.7% overlap the hull of some labelled panel in the window,
  visible or not: most of the excess is real panels that the 50% visibility rule leaves out of the
  ground truth (less than half inside the window, or half under the band), plus duplicates. Against
  visible-only ground truth those count as false positives.
- Occlusion: 706 runs over all 30 sequences, 84.8% of 5 to 15 frames, 380 of exactly 10 (the
  configured length); the shorter runs are panels that are also leaving the window. 13 panels on
  development seeds are occluded at every frame they are inside the window and so never visible.

| dev seed | labels in source | GT boxes | true modules | detections | bounces |
|---|---|---|---|---|---|
| 0 | 48 | 379 | 16 | 874 | 2 |
| 1 | 105 | 915 | 38 | 1,418 | 3 |
| 2 | 97 | 908 | 38 | 1,462 | 2 |
| 3 | 111 | 917 | 46 | 1,414 | 2 |
| 4 | 87 | 987 | 35 | 1,419 | 1 |
| 5 | 27 | 47 | 4 | 176 | 2 |
| 6 | 77 | 1,081 | 43 | 1,687 | 2 |
| 7 | 90 | 704 | 23 | 1,146 | 0 |
| 8 | 44 | 470 | 16 | 906 | 2 |
| 9 | 82 | 912 | 35 | 1,372 | 2 |

Seed 5's window stays over trees beside the corner of a panel block, so it has few visible panels.
The held-out sequences are generated and cached, and not scored.

Raw deduplication count, `scripts/dedup_flyover.py` at `378ecbe`, each tracker at its defaults
(max_age 1, min_hits 3, IoU gate 0.3), **development seeds, before any metric code**. Ids are the
distinct reported track ids; true modules are the distinct ground-truth ids visible at least once.
No matching against ground truth is done, so these are counts, not accuracy.

- no tracker: 600 frames, 11874 detections, 11874 ids, vs 294 true modules
- greedy IoU: 600 frames, 11874 detections, 654 ids, vs 294 true modules (1,016 tracks started)
- SORT: 600 frames, 11874 detections, 660 ids, vs 294 true modules (1,075 tracks started)

## D5 tracking metrics and sweep, declared before the first metric ran

Fri 23 Oct 2026. Written and committed before any tracking metric has been computed on any D5
sequence, development or held-out. Code: `src/solar_inspect/tracking/metrics.py` (tests first, on
hand-made sequences).

**MOTA and IDF1 here are correctness checks of the implementation on synthetic data, never a
performance result.** D5 is a window moving over one still frame; the numbers say whether the
trackers and the metric code behave as designed, not how a tracker would do on a real flyover.

### Metrics

- **Hypotheses.** Per frame, the detections a tracker reports an id for. A detection whose track is
  not yet reported (id -1) is not a hypothesis. For no tracker every detection is a hypothesis.
- **Ground truth.** The generator's visible boxes: at least 50% of the panel's warped area inside the
  window and not under the band.
- **Ignore boxes.** Every other labelled panel with any part of its warped area inside the window:
  under 50% inside the window, or under the band. Its box is the same clipped hull the generator uses
  for visible boxes, recomputed from the cached per-frame homography, the cached band and the val
  label boxes with the generator's own functions (the generator is not changed).
- **Ignore rule.** Before a frame is scored, every hypothesis whose IoU with every visible box is
  below 0.5 and whose IoU with some ignore box is at least 0.5 is dropped. A hypothesis that reaches
  0.5 with a visible box is kept, whatever it does with ignore boxes. Ignore boxes are never misses.
  The number of hypotheses dropped is reported per tracker.
- **Matching (motmetrics, library).** `MOTAccumulator`, CLEAR MOT matching: per frame, a ground-truth
  box first keeps the hypothesis it was last matched to if their IoU is still at least 0.5; the rest
  are paired by minimum-cost assignment (scipy) on 1 - IoU, pairs above distance 0.5 not allowed.
  Distances are `motmetrics.distances.boxiou` on (x, y, w, h) with `iou_matrix`'s rule written out
  (1 - IoU, NaN above 0.5): `iou_matrix` itself calls `np.asfarray`, which NumPy 2 removed.
- **motmetrics metrics:** MOTA, IDF1, IDSW (`num_switches`), FP, FN, per sequence and over sequences
  (`compute_many(..., generate_overall=True)`, which sums the counts before dividing).
- **Hand-written IDSW** (`id_switches`): the same CLEAR MOT matching written out (scipy does the
  assignment step), then, for each ground-truth id, a walk over the frames where it is matched; a
  switch is each time its matched track id differs from the last one it was matched to. It must equal
  motmetrics' `num_switches` on every sequence; if it does not, the reconciliation is timeboxed at 45
  minutes and logged here as unresolved.
- **Unique-module count** (`count_error`, hand-written): K = distinct reported track ids with at
  least one hypothesis left after the ignore rule; G = distinct ground-truth ids visible at least
  once. Reported: K, G, K - G and (K - G) / G. The dedup line: "N frames, M detections, K ids, vs G
  true modules", M = all detections.

### Tracker settings, chosen on the development seeds only

For SORT and greedy IoU separately: every combination of `max_age` in {1, 3, 5, 10, 15}, `min_hits`
in {1, 3}, `iou_gate` in {0.1, 0.3, 0.5} (30 settings) is run on development seeds 0 to 9. The chosen
setting has the lowest sum over the 10 seeds of |K - G|; ties go to the higher sum of per-seed IDF1,
then to the smaller `max_age`, then (reached only by settings with identical scores) to the smaller
`min_hits` and then the smaller `iou_gate`. No setting is tuned on a held-out seed. The no-tracker floor has no
settings. The full grid and both choices are recorded here and the choices committed in a config
before any held-out sequence is scored.

### The held-out sweep, seeds 100 to 119, one factor at a time

Defaults: pan speed 4 px per frame, occlusion 10 frames, detector confidence 0.25 (the shipped
`conf_operating`), each tracker at its chosen setting. One factor varies, the others stay at default:

| factor | values | how |
|---|---|---|
| pan speed (px per frame) | 2, 4, 6, 8 | sequences regenerated, same seeds; bounces recorded per sequence |
| occlusion length (frames) | 5, 10, 15 | sequences regenerated, same seeds |
| detector confidence | 0.1, 0.25, 0.5 | detector re-run on the default sequences |
| `max_age` | 1, 5, 15 | SORT and greedy; the rest of each tracker's chosen setting |
| `iou_gate` | 0.1, 0.3, 0.5 | SORT and greedy; the rest of each tracker's chosen setting |

The default cells reuse run `flyover_20261022T221314Z`. Every regeneration runs
`scripts/make_flyover.py` with the committed generator and detector, the one overridden parameter
given on the command line and recorded in the new run's manifest.

Per cell and tracker (no tracker, greedy IoU, SORT), summed over the 20 sequences: MOTA, IDF1, IDSW
(hand-written, motmetrics' beside it), FP, FN, K, G, K - G, hypotheses dropped by the ignore rule,
bounces (pan-speed rows), and the per-sequence median and range of (K - G) / G. Where greedy and SORT
give the same numbers, that is stated. Headline at defaults, held-out: the dedup line per tracker.

## D5 tracking: development grid and held-out sweep

Fri 23 Oct 2026. Definitions as declared above ([`91c597e`](../../../commit/91c597e)). Code: metrics
[`8e04ee6`](../../../commit/8e04ee6); one-parameter override for `scripts/make_flyover.py`
[`44e45da`](../../../commit/44e45da); `scripts/track_flyover.py` [`d777597`](../../../commit/d777597);
`scripts/plot_track_sweep.py` [`330df77`](../../../commit/330df77).

**Synthetic data: MOTA and IDF1 below check the implementation; they are not a performance result.**

Checks that held on every frame scored: the visible ground truth recomputed from the cached homography
and band equals the cached ground truth (ids exact, boxes to float tolerance), so the ignore boxes come
from the same geometry; and the hand-written ID-switch count equals motmetrics' `num_switches` on every
sequence of all 61 development settings and all 48 sweep cell-tracker pairs, so no reconciliation was
needed.

Regenerated runs for the sweep, each `scripts/make_flyover.py --set` at `7a5ed7a`, clean tree, same
checkpoint and seeds: pan speed 2 / 6 / 8 `flyover_20261022T235341Z` / `T235402Z` / `T235422Z`;
occlusion 5 / 15 `T235441Z` / `T235458Z`; detector conf 0.1 / 0.5 `T235510Z` / `T235528Z`. The two conf
runs reproduce the default run's ground truth exactly, and the conf 0.5 detections equal the default
run's detections with score >= 0.5 on all 20 held-out sequences.

Dev grid, development seeds 0-9 (600 frames, 11,874 detections, G = 294), run `flyover_20261022T221314Z`, `scripts/track_flyover.py dev` at `d777597`, clean tree. Sum |K - G| is over the 10 seeds; sum IDF1 is the sum of the 10 per-seed IDF1 values; IDSW is summed, hand-written / motmetrics.

No tracker: K 8,689, sum |K - G| 8,395, sum IDF1 0.3802, 3,185 hypotheses dropped by the ignore rule.

| max_age | min_hits | IoU gate | greedy sum abs(K - G) | greedy sum IDF1 | greedy K | greedy IDSW | SORT sum abs(K - G) | SORT sum IDF1 | SORT K | SORT IDSW |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.1 | 341 | 6.7577 | 635 | 167 / 167 | 347 | 6.7459 | 641 | 173 / 173 |
| 1 | 1 | 0.3 | 343 | 6.7824 | 637 | 167 / 167 | 359 | 6.7860 | 653 | 172 / 172 |
| 1 | 1 | 0.5 | 317 | 6.8758 | 611 | 166 / 166 | 395 | 6.6966 | 689 | 199 / 199 |
| 1 | 3 | 0.1 | 251 | 6.9062 | 545 | 166 / 166 | 253 | 6.9077 | 547 | 169 / 169 |
| 1 | 3 | 0.3 | 245 | 6.9220 | 539 | 166 / 166 | 248 | 6.9401 | 542 | 168 / 168 |
| 1 | 3 | 0.5 | 231 | 6.9775 | 525 | 166 / 166 | 253 | 6.8458 | 547 | 186 / 186 |
| 3 | 1 | 0.1 | 325 | 6.7852 | 619 | 167 / 167 | 327 | 6.7758 | 621 | 172 / 172 |
| 3 | 1 | 0.3 | 341 | 6.8202 | 635 | 167 / 167 | 347 | 6.8110 | 641 | 175 / 175 |
| 3 | 1 | 0.5 | 314 | 6.9155 | 608 | 166 / 166 | 393 | 6.7072 | 687 | 204 / 204 |
| 3 | 3 | 0.1 | 250 | 6.9161 | 544 | 165 / 165 | 249 | 6.9228 | 543 | 166 / 166 |
| 3 | 3 | 0.3 | 245 | 6.9345 | 539 | 166 / 166 | 250 | 6.9402 | 544 | 170 / 170 |
| 3 | 3 | 0.5 | 231 | 6.9900 | 525 | 166 / 166 | 252 | 6.8424 | 546 | 189 / 189 |
| 5 | 1 | 0.1 | 307 | 6.7814 | 601 | 169 / 169 | 321 | 6.7877 | 615 | 170 / 170 |
| 5 | 1 | 0.3 | 334 | 6.8113 | 628 | 171 / 171 | 347 | 6.8059 | 641 | 175 / 175 |
| 5 | 1 | 0.5 | 312 | 6.9097 | 606 | 170 / 170 | 396 | 6.7120 | 690 | 205 / 205 |
| 5 | 3 | 0.1 | 236 | 6.9262 | 530 | 164 / 164 | 246 | 6.9449 | 540 | 165 / 165 |
| 5 | 3 | 0.3 | 241 | 6.9339 | 535 | 167 / 167 | 248 | 6.9395 | 542 | 170 / 170 |
| 5 | 3 | 0.5 | 226 | 6.9877 | 520 | 167 / 167 | 253 | 6.8524 | 547 | 191 / 191 |
| 10 | 1 | 0.1 | 279 | 6.8532 | 573 | 159 / 159 | 301 | 6.9977 | 595 | 160 / 160 |
| 10 | 1 | 0.3 | 323 | 6.8494 | 617 | 162 / 162 | 348 | 6.8356 | 642 | 172 / 172 |
| 10 | 1 | 0.5 | 307 | 6.9922 | 601 | 156 / 156 | 400 | 6.7161 | 694 | 207 / 207 |
| 10 | 3 | 0.1 | 209 | 7.0162 | 503 | 157 / 157 | 225 | 7.1753 | 519 | 155 / 155 |
| 10 | 3 | 0.3 | 226 | 6.9838 | 520 | 159 / 159 | 246 | 6.9824 | 540 | 166 / 166 |
| 10 | 3 | 0.5 | 212 | 7.0754 | 506 | 153 / 153 | 256 | 6.8671 | 550 | 192 / 192 |
| 15 | 1 | 0.1 | 256 | 6.8331 | 550 | 160 / 160 | 276 | 6.9195 | 570 | 162 / 162 |
| 15 | 1 | 0.3 | 301 | 6.8830 | 595 | 162 / 162 | 337 | 6.7932 | 631 | 173 / 173 |
| 15 | 1 | 0.5 | 302 | 6.9865 | 596 | 158 / 158 | 405 | 6.7353 | 699 | 212 / 212 |
| 15 | 3 | 0.1 | **191** | 7.0087 | 485 | 156 / 156 | **204** | 7.0829 | 498 | 156 / 156 |
| 15 | 3 | 0.3 | 212 | 7.0131 | 506 | 158 / 158 | 229 | 6.9451 | 523 | 167 / 167 |
| 15 | 3 | 0.5 | 207 | 7.0767 | 501 | 153 / 153 | 253 | 6.8937 | 547 | 193 / 193 |

Chosen (bold): both trackers at max_age 15, min_hits 3, IoU gate 0.1, the grid's largest max_age. Committed in `configs/tracking.yaml` [`7a5ed7a`](../../../commit/7a5ed7a) before any held-out sequence was scored.

Held-out sweep, seeds 100-119 (20 sequences, 1,200 frames per cell), `scripts/track_flyover.py sweep` at `7a5ed7a`, clean tree, run `track_sweep_20261022T235853Z`. Settings: greedy and SORT both max_age 15, min_hits 3, IoU gate 0.1 unless the row varies one. Counts are summed over the 20 sequences; MOTA and IDF1 are motmetrics' overall values (counts summed before dividing). The (K - G)/G spread is over the 20 per-sequence values. Hand-written IDSW equals motmetrics' on every sequence of every cell.

**pan_speed** (default 4)

| value | tracker | MOTA | IDF1 | IDSW hand / mm | FP | FN | K | G | K - G | (K - G)/G median [min, max] | dropped (ignore) | detections | bounces |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | no tracker | -0.111 | 0.032 | 17,687 / 17,687 | 2,678 | 163 | 20,993 | 628 | 20,365 | 32.750 [27.514, 64.667] | 6,216 | 27,209 | 22 (per sequence 0-3) |
| 2 | greedy IoU | 0.899 | 0.733 | 322 / 322 | 1,155 | 381 | 947 | 628 | 319 | 0.536 [0.167, 2.000] | 4,726 | 27,209 | 22 (per sequence 0-3) |
| 2 | SORT | 0.898 | 0.729 | 324 / 324 | 1,169 | 395 | 945 | 628 | 317 | 0.487 [0.222, 2.000] | 4,700 | 27,209 | 22 (per sequence 0-3) |
| 4 | no tracker | -0.110 | 0.035 | 17,801 / 17,801 | 2,756 | 156 | 21,261 | 705 | 20,556 | 30.256 [22.652, 48.417] | 6,257 | 27,518 | 34 (per sequence 1-3) |
| 4 | greedy IoU | 0.894 | 0.732 | 355 / 355 | 1,187 | 442 | 1,035 | 705 | 330 | 0.493 [0.222, 1.750] | 4,359 | 27,518 | 34 (per sequence 1-3) |
| 4 | SORT | 0.890 | 0.725 | 383 / 383 | 1,176 | 499 | 1,103 | 705 | 398 | 0.552 [0.231, 1.917] | 4,349 | 27,518 | 34 (per sequence 1-3) |
| 6 | no tracker | -0.109 | 0.038 | 17,804 / 17,804 | 2,818 | 165 | 21,389 | 767 | 20,622 | 27.596 [21.721, 41.286] | 6,141 | 27,530 | 50 (per sequence 1-4) |
| 6 | greedy IoU | 0.880 | 0.733 | 438 / 438 | 1,185 | 617 | 1,147 | 767 | 380 | 0.501 [0.224, 1.357] | 3,851 | 27,530 | 50 (per sequence 1-4) |
| 6 | SORT | 0.870 | 0.715 | 504 / 504 | 1,181 | 756 | 1,269 | 767 | 502 | 0.631 [0.269, 1.636] | 3,838 | 27,530 | 50 (per sequence 1-4) |
| 8 | no tracker | -0.107 | 0.040 | 17,439 / 17,439 | 2,749 | 155 | 20,978 | 790 | 20,188 | 25.724 [20.587, 36.562] | 6,396 | 27,374 | 65 (per sequence 2-5) |
| 8 | greedy IoU | 0.867 | 0.708 | 507 / 507 | 1,108 | 837 | 1,191 | 790 | 401 | 0.512 [0.183, 1.250] | 3,788 | 27,374 | 65 (per sequence 2-5) |
| 8 | SORT | 0.840 | 0.659 | 703 / 703 | 1,092 | 1,140 | 1,358 | 790 | 568 | 0.688 [0.302, 1.788] | 3,822 | 27,374 | 65 (per sequence 2-5) |

**occlusion_frames** (default 10)

| value | tracker | MOTA | IDF1 | IDSW hand / mm | FP | FN | K | G | K - G | (K - G)/G median [min, max] | dropped (ignore) | detections |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | no tracker | -0.120 | 0.032 | 19,772 / 19,772 | 3,197 | 239 | 23,679 | 710 | 22,969 | 33.417 [26.298, 48.769] | 6,945 | 30,624 |
| 5 | greedy IoU | 0.888 | 0.713 | 390 / 390 | 1,418 | 513 | 1,028 | 710 | 318 | 0.469 [0.153, 1.154] | 4,894 | 30,624 |
| 5 | SORT | 0.886 | 0.723 | 390 / 390 | 1,407 | 560 | 1,092 | 710 | 382 | 0.524 [0.256, 1.538] | 4,867 | 30,624 |
| 10 | no tracker | -0.110 | 0.035 | 17,801 / 17,801 | 2,756 | 156 | 21,261 | 705 | 20,556 | 30.256 [22.652, 48.417] | 6,257 | 27,518 |
| 10 | greedy IoU | 0.894 | 0.732 | 355 / 355 | 1,187 | 442 | 1,035 | 705 | 330 | 0.493 [0.222, 1.750] | 4,359 | 27,518 |
| 10 | SORT | 0.890 | 0.725 | 383 / 383 | 1,176 | 499 | 1,103 | 705 | 398 | 0.552 [0.231, 1.917] | 4,349 | 27,518 |
| 15 | no tracker | -0.119 | 0.039 | 15,874 / 15,874 | 2,694 | 147 | 19,267 | 699 | 18,568 | 27.564 [19.800, 52.500] | 5,700 | 24,967 |
| 15 | greedy IoU | 0.885 | 0.749 | 328 / 328 | 1,122 | 474 | 1,057 | 699 | 358 | 0.477 [0.176, 2.300] | 3,965 | 24,967 |
| 15 | SORT | 0.882 | 0.740 | 356 / 356 | 1,106 | 509 | 1,116 | 699 | 417 | 0.566 [0.339, 2.300] | 3,959 | 24,967 |

**conf** (default 0.25)

| value | tracker | MOTA | IDF1 | IDSW hand / mm | FP | FN | K | G | K - G | (K - G)/G median [min, max] | dropped (ignore) | detections |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | no tracker | -0.209 | 0.034 | 17,803 / 17,803 | 4,595 | 154 | 23,102 | 705 | 22,397 | 32.714 [26.109, 54.083] | 7,352 | 30,454 |
| 0.1 | greedy IoU | 0.866 | 0.719 | 369 / 369 | 1,714 | 426 | 1,163 | 705 | 458 | 0.630 [0.361, 2.500] | 4,946 | 30,454 |
| 0.1 | SORT | 0.863 | 0.712 | 394 / 394 | 1,672 | 492 | 1,252 | 705 | 547 | 0.747 [0.382, 2.167] | 4,868 | 30,454 |
| 0.25 | no tracker | -0.110 | 0.035 | 17,801 / 17,801 | 2,756 | 156 | 21,261 | 705 | 20,556 | 30.256 [22.652, 48.417] | 6,257 | 27,518 |
| 0.25 | greedy IoU | 0.894 | 0.732 | 355 / 355 | 1,187 | 442 | 1,035 | 705 | 330 | 0.493 [0.222, 1.750] | 4,359 | 27,518 |
| 0.25 | SORT | 0.890 | 0.725 | 383 / 383 | 1,176 | 499 | 1,103 | 705 | 398 | 0.552 [0.231, 1.917] | 4,349 | 27,518 |
| 0.5 | no tracker | -0.058 | 0.036 | 17,779 / 17,779 | 1,792 | 178 | 20,275 | 705 | 19,570 | 28.990 [21.152, 45.750] | 5,196 | 25,471 |
| 0.5 | greedy IoU | 0.900 | 0.743 | 343 / 343 | 890 | 634 | 961 | 705 | 256 | 0.355 [0.103, 1.583] | 3,682 | 25,471 |
| 0.5 | SORT | 0.898 | 0.739 | 361 / 361 | 874 | 667 | 1,020 | 705 | 315 | 0.438 [0.103, 1.667] | 3,687 | 25,471 |

**max_age** (chosen 15)

| value | tracker | MOTA | IDF1 | IDSW hand / mm | FP | FN | K | G | K - G | (K - G)/G median [min, max] | dropped (ignore) | detections |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | no tracker | -0.110 | 0.035 | 17,801 / 17,801 | 2,756 | 156 | 21,261 | 705 | 20,556 | 30.256 [22.652, 48.417] | 6,257 | 27,518 |
| 1 | greedy IoU | 0.886 | 0.713 | 392 / 392 | 1,177 | 550 | 1,202 | 705 | 497 | 0.679 [0.441, 1.750] | 4,341 | 27,518 |
| 1 | SORT | 0.887 | 0.712 | 395 / 395 | 1,168 | 547 | 1,199 | 705 | 494 | 0.677 [0.471, 1.917] | 4,288 | 27,518 |
| 5 | no tracker | -0.110 | 0.035 | 17,801 / 17,801 | 2,756 | 156 | 21,261 | 705 | 20,556 | 30.256 [22.652, 48.417] | 6,257 | 27,518 |
| 5 | greedy IoU | 0.887 | 0.720 | 380 / 380 | 1,184 | 540 | 1,169 | 705 | 464 | 0.626 [0.441, 1.750] | 4,379 | 27,518 |
| 5 | SORT | 0.887 | 0.712 | 398 / 398 | 1,161 | 548 | 1,196 | 705 | 491 | 0.668 [0.471, 1.917] | 4,323 | 27,518 |
| 15 | no tracker | -0.110 | 0.035 | 17,801 / 17,801 | 2,756 | 156 | 21,261 | 705 | 20,556 | 30.256 [22.652, 48.417] | 6,257 | 27,518 |
| 15 | greedy IoU | 0.894 | 0.732 | 355 / 355 | 1,187 | 442 | 1,035 | 705 | 330 | 0.493 [0.222, 1.750] | 4,359 | 27,518 |
| 15 | SORT | 0.890 | 0.725 | 383 / 383 | 1,176 | 499 | 1,103 | 705 | 398 | 0.552 [0.231, 1.917] | 4,349 | 27,518 |

**iou_gate** (chosen 0.1)

| value | tracker | MOTA | IDF1 | IDSW hand / mm | FP | FN | K | G | K - G | (K - G)/G median [min, max] | dropped (ignore) | detections |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | no tracker | -0.110 | 0.035 | 17,801 / 17,801 | 2,756 | 156 | 21,261 | 705 | 20,556 | 30.256 [22.652, 48.417] | 6,257 | 27,518 |
| 0.1 | greedy IoU | 0.894 | 0.732 | 355 / 355 | 1,187 | 442 | 1,035 | 705 | 330 | 0.493 [0.222, 1.750] | 4,359 | 27,518 |
| 0.1 | SORT | 0.890 | 0.725 | 383 / 383 | 1,176 | 499 | 1,103 | 705 | 398 | 0.552 [0.231, 1.917] | 4,349 | 27,518 |
| 0.3 | no tracker | -0.110 | 0.035 | 17,801 / 17,801 | 2,756 | 156 | 21,261 | 705 | 20,556 | 30.256 [22.652, 48.417] | 6,257 | 27,518 |
| 0.3 | greedy IoU | 0.892 | 0.725 | 370 / 370 | 1,090 | 563 | 1,118 | 705 | 413 | 0.576 [0.359, 1.833] | 4,259 | 27,518 |
| 0.3 | SORT | 0.896 | 0.720 | 400 / 400 | 941 | 593 | 1,154 | 705 | 449 | 0.597 [0.385, 1.917] | 4,155 | 27,518 |
| 0.5 | no tracker | -0.110 | 0.035 | 17,801 / 17,801 | 2,756 | 156 | 21,261 | 705 | 20,556 | 30.256 [22.652, 48.417] | 6,257 | 27,518 |
| 0.5 | greedy IoU | 0.893 | 0.724 | 378 / 378 | 781 | 842 | 1,148 | 705 | 443 | 0.633 [0.333, 1.833] | 4,019 | 27,518 |
| 0.5 | SORT | 0.890 | 0.706 | 447 / 447 | 496 | 1,101 | 1,220 | 705 | 515 | 0.721 [0.410, 2.000] | 3,505 | 27,518 |

Greedy IoU and SORT give the same numbers in no cell. They come closest at pan speed 2 (K - G 319 vs
317) and at max_age 1 (497 vs 494), the two cells where SORT's count error is lower, by 2 and 3 ids.
In the other 11 of 13 distinct cells greedy IoU's count error is lower. IDF1 is higher for greedy in
every cell except occlusion 5 (0.713 vs 0.723); MOTA differs by under 0.005 everywhere except
pan speed 6 (0.880 vs 0.870) and 8 (0.867 vs 0.840), greedy higher in both.

Headline dedup lines, defaults, held-out:

- no tracker: 1,200 frames, 27,518 detections, 21,261 ids, vs 705 true modules
- greedy IoU: 1,200 frames, 27,518 detections, 1,035 ids, vs 705 true modules
- SORT: 1,200 frames, 27,518 detections, 1,103 ids, vs 705 true modules

K counts the reported ids left after the ignore rule; ids on panels that fail the visibility rule
(at the window edge or under the band) are not in K. The detections are all detections.

### What the sweep shows

- Both trackers cut the id count from 21,261 (no tracker, (K - G)/G median 30.3) to about 1.5 times the
  true modules: (K - G)/G median 0.49 for greedy IoU and 0.55 for SORT at defaults, per-sequence range
  0.22 to 1.75 and 0.23 to 1.92.
- The constant-velocity Kalman filter does not reduce the count error on D5. SORT reports more surplus
  ids than greedy IoU in 11 of 13 cells, and the gap grows with pan speed: SORT minus greedy K - G is
  -2, 68, 122, 167 at 2, 4, 6, 8 px per frame. The bounces grow with pan speed too (22, 34, 50, 65
  over 20 sequences), and every bounce reverses the window's velocity in one frame, so these rows
  cannot separate speed from velocity reversals.
- Detector confidence moves the count error most among the non-motion factors: (K - G)/G median
  0.630 / 0.493 / 0.355 for greedy and 0.747 / 0.552 / 0.438 for SORT at conf 0.1 / 0.25 / 0.5, while FN
  rises from 426 to 634 (greedy).
- Longer track memory lowers the count error: greedy K - G 497 / 464 / 330 and SORT 494 / 491 / 398 at
  max_age 1 / 5 / 15. A looser IoU gate does too: greedy 443 / 413 / 330 and SORT 515 / 449 / 398 at
  0.5 / 0.3 / 0.1. Both agree with the development grid, where the largest max_age and the 0.1 gate won.
- Occlusion length 5 to 15 frames changes K - G by at most 40 ids for either tracker at max_age 15.
- The ignore rule drops 4,359 (greedy), 4,349 (SORT) and 6,257 (no tracker) hypotheses at defaults:
  without it they would count as false positives on real panels.

## D5 tracking: post-hoc motion diagnostic

Fri 23 Oct 2026. **Post-hoc: measured after the held-out sweep, to explain it; changes no setting or
number.** `scripts/motion_flyover.py` at [`cd77fff`](../../../commit/cd77fff), clean tree, run
`track_motion_20261023T000731Z`, 39 s. Ground truth only (no detector, no tracker), all 30 sequences
(development and held-out) of the default run and the three pan-speed runs.

For each ground-truth id, along each run of consecutive visible frames: "stay" predicts the next box as
the last true box (what greedy IoU matches against); "Kalman" is SORT's filter with SORT's constants,
started at the run's first box and updated with every true box so far. k = 1 is the next frame; k = 5
and 10 coast with no update, against the true box k frames later (the occlusion case). The stay
prediction's k = 1 IoU is the IoU of a true box with itself one frame later. Gate = 0.1.

Median ground-truth box 62.4 to 62.9 x 36.7 to 36.8 px at every speed. Median per-frame centre
displacement: 2.20 / 3.80 / 5.65 / 7.31 px at pan speed 2 / 4 / 6 / 8.

| pan speed | k | predictions | stay median | stay 10th pct | stay below 0.1 | Kalman median | Kalman 10th pct | Kalman below 0.1 |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 25,260 | 0.891 | 0.811 | 0 | 0.938 | 0.811 | 0 |
| 2 | 5 | 19,853 | 0.578 | 0.357 | 0.1% (12) | 0.789 | 0.519 | 0.8% (162) |
| 2 | 10 | 13,976 | 0.344 | 0.082 | 11.8% | 0.609 | 0.216 | 6.0% |
| 4 | 1 | 24,401 | 0.824 | 0.743 | 0 | 0.908 | 0.739 | 0 |
| 4 | 5 | 18,592 | 0.391 | 0.201 | 1.3% (233) | 0.711 | 0.287 | 3.6% (668) |
| 4 | 10 | 12,517 | 0.135 | 0.000 | 40.7% | 0.477 | 0.000 | 18.7% |
| 6 | 1 | 24,500 | 0.761 | 0.670 | 0 | 0.867 | 0.641 | 0 |
| 6 | 5 | 18,071 | 0.251 | 0.088 | 11.9% | 0.623 | 0.087 | 10.8% |
| 6 | 10 | 11,774 | 0.009 | 0.000 | 71.7% | 0.319 | 0.000 | 34.4% |
| 8 | 1 | 24,294 | 0.700 | 0.598 | 0 | 0.803 | 0.537 | 0 |
| 8 | 5 | 17,394 | 0.151 | 0.000 | 37.4% | 0.448 | 0.000 | 23.7% |
| 8 | 10 | 11,142 | 0.000 | 0.000 | 73.9% | 0.062 | 0.000 | 52.6% |

What it shows, and what it does not:

- One frame ahead, the last box always overlaps the next one: no step at any speed falls below the 0.1
  gate for either prediction (median displacement 2.2 to 7.3 px against a 62 x 37 px box). At the
  chosen gate a motion model cannot rescue a one-frame association that the last box would lose.
  The Kalman prediction has the higher median IoU (0.908 vs 0.824 at pan speed 4), and a lower 10th
  percentile at pan speed 6 and 8 (0.641 vs 0.670, 0.537 vs 0.598).
- Coasting, the constant-velocity prediction does **not** drift further than a stationary one on true
  boxes: its median IoU is higher at every speed and horizon, and fewer of its predictions fall below
  the gate at k = 10 everywhere and at k = 5 for pan speed 6 and 8. The exception is k = 5 at pan
  speed 2 and 4, where more Kalman predictions fall below the gate (162 vs 12, 668 vs 233).
- So on ground-truth boxes the Kalman filter predicts better than "stay", and this diagnostic does not
  explain the sweep's result. What it leaves untested is the difference between true boxes and the
  detector's: SORT's filter runs on detections, with duplicates and box jitter, and greedy and SORT
  also differ in the assignment. Neither was measured here.

## D3 segmentation: build

Fri 23 Oct 2026. Split [`f2b0465`](../../../commit/f2b0465) (ADR 0005); U-Net
[`ab9b2d5`](../../../commit/ab9b2d5); loader and training loop [`227d183`](../../../commit/227d183);
`configs/seg_unet.yaml` [`bc7df9d`](../../../commit/bc7df9d), committed before any training run;
colour baseline [`fbeb0f2`](../../../commit/fbeb0f2).

**Split** `b5924638`, by scene, scenes linked over all three subsets at name-coordinate distance
<= 4.5 (touching, corners included): 64 scenes. SteelTile is cut into blocks along the long axis of
its two large scenes; every other scene goes whole (seed 0, 20,000 draws, about 70 / 15 / 15 per
subset); train tiles touching a val or test tile are guard tiles and are used for nothing.

| Subset | train | val | test | guard | fg train | fg val | fg test |
|---|---:|---:|---:|---:|---:|---:|---:|
| Brick | 98 | 21 | 19 | 0 | 0.0482 | 0.0647 | 0.0578 |
| FlatConcrete | 293 | 60 | 60 | 0 | 0.2999 | 0.4123 | 0.4391 |
| SteelTile | 54 | 14 | 14 | 12 | 0.5278 | 0.5692 | 0.2547 |

Minimum distance from any val or test tile to any train tile, per subset (val / test): Brick
5.099 / 5.0, FlatConcrete 18.111 / 7.0, SteelTile 5.0 / 5.831. All above 4.5. The split script
rebuilds the assignment byte-identically.

**U-Net**, `torch.nn` layers only, base width 32, depth 4, bilinear upsampling then a 3x3 conv:
8,633,441 parameters. Normalisation: per-channel mean [0.5312, 0.5323, 0.5396] and std [0.2478,
0.2475, 0.2561], train tiles only.

**Smoke run** `seg_unet_20261023T144455Z` at `bc7df9d`, clean tree, override `epochs=5` (the cosine
schedule therefore spans 5 epochs, not 100). RTX 4060 Laptop GPU, AMP on. Median 4.92 s per epoch
(4.59 s training, the rest val), wall 27.0 s, peak VRAM 2,077 MiB allocated / 3,526 MiB reserved.

| epoch | train loss | val IoU Brick | FlatConcrete | SteelTile | mean |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.8692 | 0.1946 | 0.5713 | 0.7318 | 0.4992 |
| 2 | 0.6909 | 0.5186 | 0.6762 | 0.8206 | 0.6718 |
| 3 | 0.5722 | 0.3949 | 0.7615 | 0.8406 | 0.6656 |
| 4 | 0.5313 | 0.5678 | 0.8596 | 0.8658 | 0.7644 |
| 5 | 0.4638 | 0.4374 | 0.8319 | 0.8568 | 0.7087 |

These are five-epoch numbers, a check that the loop learns, not a result. Brick val IoU moves by
up to 0.17 between epochs (21 tiles, 6% panel pixels).

**Colour baseline**, `scripts/baseline_seg.py` at `fbeb0f2`. One threshold on R, G, B or 2B - R - G,
above or below, every integer value, chosen on the mean of the three per-subset **train** IoUs.
Chosen: **panel where 2B - R - G >= 24** (mean train IoU 0.4299; the blue channel alone, B >= 106,
reaches 0.3620). Scored once on val:

| Subset | train IoU | val IoU | val Dice | val precision | val recall |
|---|---:|---:|---:|---:|---:|
| Brick | 0.2626 | 0.3967 | 0.5680 | 0.4201 | 0.8766 |
| FlatConcrete | 0.4612 | 0.6611 | 0.7960 | 0.7902 | 0.8019 |
| SteelTile | 0.5658 | 0.7313 | 0.8448 | 0.7968 | 0.8989 |

The fixed rule scores higher on val than on train in every subset. Val holds more panel pixels than
train for Brick and FlatConcrete, and a fixed rule's IoU rises with the positive share, so val and
train IoUs on this split are not directly comparable.

Null all-background model, pixel accuracy per split (its IoU is 0 everywhere):

| Split | Brick | FlatConcrete | SteelTile |
|---|---:|---:|---:|
| train | 0.9518 | 0.7001 | 0.4722 |
| val | 0.9353 | 0.5877 | 0.4308 |
| test | 0.9422 | 0.5609 | 0.7453 |

## D3 segmentation: results

Fri 23 Oct 2026. Evaluation code [`66c7574`](../../../commit/66c7574) (`segmentation/evaluate.py`,
`scripts/eval_seg.py`); evaluation config `configs/seg_eval.yaml` [`49c44d9`](../../../commit/49c44d9),
committed before the test split was read; overlays and post-hoc diagnostic
[`c1655f6`](../../../commit/c1655f6); thermal check declared in `configs/seg_thermal_negative.yaml`
[`0826cca`](../../../commit/0826cca) before its first run, script [`4a98605`](../../../commit/4a98605).

**Full run** `seg_unet_20261023T145359Z`, from `c1f0e88` (config `bc7df9d`), clean tree, no overrides:
100 epochs, wall 524.6 s, median 5.17 s per epoch (4.82 s training), peak VRAM 2,080 MiB allocated /
3,526 MiB reserved, train loss 0.8692 at epoch 1 and 0.0859 at epoch 100. Selected epoch **87**
(highest mean of the three per-subset val IoUs). `best.pt` SHA-256
`9a2653125d19df6c0883c8c6e0dacfcb42d651738558ae6c710bbfeae0bb8724`, pinned in the evaluation config.

### Selection check, val

From the run's `epochs.jsonl`, computed by `scripts/eval_seg.py val` (run
`seg_eval_val_20261023T153105Z` at `49c44d9`, clean):

| Subset | val IoU at epoch 87 | mean of epochs 91-100 | difference | min, max over epochs 71-100 |
|---|---:|---:|---:|---|
| Brick | 0.9074 | 0.9000 | +0.0073 | 0.8428, 0.9074 |
| FlatConcrete | 0.9272 | 0.9265 | +0.0007 | 0.9196, 0.9277 |
| SteelTile | 0.9455 | 0.9455 | +0.00001 | 0.9395, 0.9463 |

Selection flatters Brick's val IoU by 0.007; epoch 87 is Brick's maximum over the last thirty
epochs. The others move by less than 0.001.

### U-Net against the colour rule, val

Same run. U-Net `best.pt`, logit > 0 (sigmoid 0.5), float16 autocast as in training; colour rule
`2B - R - G >= 24` from `configs/seg_baseline.json`. Intervals: percentile 95% bootstrap over the
subset's tiles, 10,000 draws, seed 0, the difference paired (same resampled tiles). "Groups" are the
connected components of the subset's tiles at name-coordinate distance <= 4.5. The script reproduces
the training log's epoch-87 U-Net IoUs and the baseline's val IoUs exactly (difference 0.0 in all
three subsets).

| Subset (val) | tiles | groups | U-Net IoU | colour rule IoU | U-Net minus colour rule |
|---|---:|---:|---|---|---|
| Brick | 21 | 8 | 0.9074 [0.8553, 0.9367] | 0.3967 [0.2538, 0.5461] | +0.5107 [0.3815, 0.6203] |
| FlatConcrete | 60 | 1 | 0.9272 [0.9192, 0.9337] | 0.6611 [0.6176, 0.7021] | +0.2660 [0.2269, 0.3079] |
| SteelTile | 14 | 1 | 0.9455 [0.9381, 0.9516] | 0.7313 [0.6757, 0.7681] | +0.2143 [0.1808, 0.2645] |

U-Net val Dice / precision / recall: Brick 0.9514 / 0.9323 / 0.9713, FlatConcrete 0.9622 / 0.9382 /
0.9874, SteelTile 0.9720 / 0.9657 / 0.9784. Pixel accuracy, U-Net / colour rule / null: Brick 0.9936 /
0.9137 / 0.9353, FlatConcrete 0.9680 / 0.8305 / 0.5877, SteelTile 0.9679 / 0.8120 / 0.4308. The U-Net
beats the colour rule in every subset; every paired interval excludes 0.

### D3 segmentation: THE TEST EVALUATION, once, and not again

**Config commit: `configs/seg_eval.yaml` [`49c44d9`](../../../commit/49c44d9)**, before
`scripts/eval_seg.py test` read the test split. The script refuses unless that config and
`seg_baseline.json` (`fbeb0f2`), `seg_unet.yaml` (`bc7df9d`) and `d3_split.json` (`f2b0465`) are
tracked and committed and the tree is clean, and printed all four SHAs; refuses a checkpoint whose
hash differs from the config's (it matched); and refuses if any `runs/seg_eval_test_*` directory
exists, which it creates before reading test. Run `seg_eval_test_20261023T153146Z` from `49c44d9`,
clean tree, one pass. Its predicted masks are saved in the run directory; nothing after it runs the
model on test.

| Subset (test) | tiles | groups | panel share | **U-Net IoU** | colour rule IoU | U-Net minus colour rule | null: IoU, pixel accuracy |
|---|---:|---:|---:|---|---|---|---|
| Brick | 19 | 8 | 0.0578 | **0.9101** [0.8593, 0.9385] | 0.3068 [0.1876, 0.4177] | +0.6033 [0.5073, 0.6977] | 0, 0.9422 |
| FlatConcrete | 60 | 3 | 0.4391 | **0.7910** [0.7350, 0.8430] | 0.4350 [0.3790, 0.4895] | +0.3560 [0.3232, 0.3944] | 0, 0.5609 |
| SteelTile | 14 | 2 | 0.2547 | **0.9778** [0.9620, 0.9851] | 0.3912 [0.2232, 0.5849] | +0.5867 [0.3980, 0.7537] | 0, 0.7453 |

| Subset (test) | U-Net Dice | U-Net precision | U-Net recall | colour rule Dice / precision / recall |
|---|---|---|---|---|
| Brick | 0.9529 [0.9243, 0.9683] | 0.9601 [0.9297, 0.9769] | 0.9459 [0.9017, 0.9706] | 0.4696 / 0.3317 / 0.8039 |
| FlatConcrete | 0.8833 [0.8472, 0.9148] | 0.9703 [0.9654, 0.9750] | 0.8106 [0.7512, 0.8664] | 0.6062 / 0.7336 / 0.5165 |
| SteelTile | 0.9888 [0.9806, 0.9925] | 0.9879 [0.9756, 0.9940] | 0.9897 [0.9826, 0.9929] | 0.5623 / 0.5933 / 0.5345 |

Pixel accuracy, test, U-Net / colour rule / null: Brick 0.9946 / **0.8950** / 0.9422, FlatConcrete
0.9060 / 0.7054 / 0.5609, SteelTile 0.9943 / 0.7881 / 0.7453. On Brick the colour rule, at IoU
0.3068, has lower pixel accuracy than the model that finds nothing.

Tiles within a scene are correlated, so the tile-bootstrap intervals are narrower than the truth.
SteelTile test is two groups (the 5-tile scene and the 9-tile block) at 25% panel pixels against 53%
in train. FlatConcrete test is below its val (0.7910 against 0.9272), through recall (0.8106 against
0.9874); precision is not lower (0.9703 against 0.9382).

**Overlay tiles**, by the rule in the config (per subset, the lower-median and the lowest U-Net IoU,
ties by name), drawn by `scripts/seg_overlays.py` (run `seg_overlays_20261023T153449Z` at `c1655f6`,
clean) from the saved masks; per-tile counts recomputed from them equal `eval.json`'s on all 93
tiles:

| Subset | median tile | IoU | lowest tile | IoU |
|---|---|---:|---|---:|
| Brick | `PV01_325472_1204667` | 0.8984 | `PV01_325424_1204680` | 0.0000 |
| FlatConcrete | `PV01_325416_1205362` | 0.8845 | `PV01_325419_1205364` | 0.0000 |
| SteelTile | `PV01_325148_1204246` | 0.9707 | `PV01_325143_1204248` | 0.2919 |

The two IoU-0 tiles hold only a sliver of panel cut by the tile edge (Brick: at the top edge;
FlatConcrete: a 1 to 2 px strip at the bottom edge), missed. SteelTile's lowest is a small panel
corner at the tile edge plus three false-positive blobs on bare ground. Figures stay under `runs/`.

### D3 segmentation: post-hoc error diagnostic of the test evaluation

**Post-hoc: measured after the single test evaluation, from its saved predicted masks, no model
pass; it changes no number above and decides nothing.** Same run `seg_overlays_20261023T153449Z`.
A pixel is within 3 px of a true boundary when the Euclidean distance from its centre to the nearest
pixel of the other class is at most 3 (the tile edge is not a boundary). The 3 px radius was chosen
after the evaluation.

| Subset (test) | FP px | FN px | errors within 3 px | FP within | FN within | all pixels within 3 px |
|---|---:|---:|---:|---:|---:|---:|
| Brick | 2,832 | 3,895 | 0.753 | 0.712 | 0.782 | 0.027 |
| FlatConcrete | 42,858 | 326,924 | 0.427 | 0.967 | 0.357 | 0.222 |
| SteelTile | 2,838 | 2,406 | 0.877 | 0.879 | 0.875 | 0.035 |

U-Net per group of test tiles (counts summed over the group):

| Subset | group (tiles, first tile) | IoU | recall | FN px |
|---|---|---:|---:|---:|
| Brick | 2, `PV01_325279_1203778` | 0.9005 | 0.9286 | 224 |
| Brick | 2, `PV01_325290_1203775` | 0.9152 | 0.9444 | 410 |
| Brick | 3, `PV01_325327_1203764` | 0.9475 | 0.9637 | 424 |
| Brick | 5, `PV01_325346_1203791` | 0.9303 | 0.9918 | 153 |
| Brick | 2, `PV01_325424_1204680` | 0.6214 | 0.6331 | 1,114 |
| Brick | 1, `PV01_325472_1204667` | 0.8984 | 0.9059 | 379 |
| Brick | 3, `PV01_325558_1204596` | 0.8897 | 0.9508 | 641 |
| Brick | 1, `PV01_325573_1204537` | 0.9436 | 0.9500 | 550 |
| FlatConcrete | 8, `PV01_325314_1205503` | 0.9150 | 0.9903 | 602 |
| FlatConcrete | **45, `PV01_325407_1205351`** | **0.7808** | **0.7981** | **325,020** |
| FlatConcrete | 7, `PV01_325572_1204537` | 0.9393 | 0.9761 | 1,302 |
| SteelTile | 9, `PV01_325141_1204246` | 0.9820 | 0.9894 | 2,167 |
| SteelTile | 5, `PV01_325206_1204230` | 0.9506 | 0.9920 | 239 |

On Brick and SteelTile the error left is at array edges. On FlatConcrete the false positives are
edges, but two thirds of the missed pixels are more than 3 px inside an array, and one 45-tile roof
holds 99.4% of them. Looked at directly: on that roof large stretches of the panels render bright,
close to white, unlike the blue-grey panels elsewhere, and those stretches are what is missed. Why
they look like that (glare, a different panel type, exposure) is not established here.

### D3 U-Net on D2 thermal frames: the transfer check

Rules committed first in `configs/seg_thermal_negative.yaml` [`0826cca`](../../../commit/0826cca);
`scripts/seg_thermal_negative.py` [`4a98605`](../../../commit/4a98605), run
`seg_thermal_20261023T153719Z` at `4a98605`, clean (a development run of the uncommitted script,
`seg_thermal_20261023T153643Z`, gave identical numbers). The pinned `best.pt`, unchanged and not
fine-tuned; D2 **val** frames only (sortie 2, 51 frames, 640 x 512, 4,929 boxes); D2 test is not
read. Each frame read as greyscale, replicated to three channels, standardised with the PV01 train
mean and std from the checkpoint; no padding needed (640 and 512 are multiples of 16); threshold
0.5. Reference: filled-box masks (pixel centres inside any ground-truth box), an approximation that
includes the gaps a box spans.

| D2 val, 51 frames, 16,711,680 px | pooled | per frame |
|---|---:|---|
| predicted positive fraction | 6e-8 (1 px) | 0 on 50 frames, 1 px on `DJI_20230213154012_0082` |
| filled-box coverage | 0.4621 | median 0.4758, min 0.1690, max 0.6926 |
| IoU with the filled-box mask | 0.0000 | 0 on every frame |
| predicted positives outside every box | 1.0 | defined on 1 frame |

**It collapses: the model predicts no panel on thermal frames.** Domain differences, all at once:
modality (PV01 RGB reflectance, three colour channels; D2 one rendered 8-bit thermal channel,
replicated); ground sampling distance (PV01 0.1 m; D2's is not documented by the dataset, and
`docs/DATA.md` estimates about 4 cm per pixel from the sensor's pitch and focal length at an assumed
30 m, consistent with the median box of 56 x 37 px); viewpoint (both near-nadir drone views, but
PV01 tiles are crops of orthophotos and D2 frames whole camera frames); content (rooftops in Jiangsu
with blue-grey panels against brick, concrete or steel; a ground-mounted plant in Greece, rows bright
against vegetation in the overlay frames). Which difference causes the collapse is not separated; a
control on greyscaled PV01 tiles was not declared and not run.

### Receptive field

`tests/test_unet.py::test_receptive_field_of_depth_4_is_252_px` [`4ddffc5`](../../../commit/4ddffc5):
one output pixel of the depth-4 U-Net depends on a 252 x 252 px input window, measured by
back-propagating from the centre output pixel through a linearised copy (ReLU removed, max pooling
made average pooling). The trained base-32 net, linearised or not, measures the same.
