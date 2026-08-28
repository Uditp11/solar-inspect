# Explainer 04 — Classification and distillation

Module 4 of the pipeline. Twelve-class defect classification on D1, plus the
transfer-learning baseline and the distillation experiment built on it.

Every number here traces to `docs/EXPERIMENTS.md` and a commit. The test split was
read once, at config commit `1a0fdfa`, and is not read again.

---

## What it is

A classifier that takes one 40×24 single-channel crop of a photovoltaic module and
assigns it to one of twelve classes: **No-Anomaly** plus eleven defect types —
Cell, Cell-Multi, Cracking, Diode, Diode-Multi, Hot-Spot, Hot-Spot-Multi,
Offline-Module, Shadowing, Soiling, Vegetation.

The input is the last stage of a chain, not a raw image. Something upstream has
already found the modules in a frame and cut them out; this stage says what is
wrong with each one. The crop is tiny by the standards of image classification —
960 pixels, about a two-hundredth of an ImageNet image — and that size is the
single fact that shapes everything downstream of it.

**The data is 8-bit rendered thermal imagery, not radiometric.** There is no
temperature anywhere in this module. A pixel is a digital number produced by
whatever render settings the source camera applied, and D1 pools crops from mixed
midwave and longwave sensors at 3–15 cm/px, so a digital number is not even
comparable between two crops in the same dataset. Nothing here is a ΔT, a kelvin,
or a degree, and no threshold in this module has physical units.

Two models were built.

- **A small CNN written for this input size**: three conv/BN/ReLU/pool stages,
  115,948 parameters. Three is the ceiling — 40×24 → 20×12 → 10×6 → 5×3, and a
  fourth pool leaves 2×1.
- **An ImageNet-pretrained ResNet-18, fine-tuned**, 11,176,396 parameters, with two
  adaptations to make a 224×224 RGB backbone read a 40×24 greyscale crop.

The ResNet-18 is the reported model: **test macro-F1 0.6956**, accuracy 0.8251
against a null model at 0.4988, on 3,007 images.

**D1 ships no official train/test split.** The split used here was drawn for this
project, so no number in this module is comparable to a published number on this
dataset. That is stated before anyone asks, because a macro-F1 quoted against a
paper's would be comparing two different problems.

---

## Why this stage exists in a PV inspection pipeline

Detection answers *where the modules are*. Classification answers *what is wrong
with this one* — and those are separate questions because the actions they trigger
are separate.

A utility-scale site has hundreds of thousands of modules. A drone survey produces
one thermal frame every second or two across a flight, and detection turns each
frame into 50–600 module boxes. Nobody looks at those boxes. What has to come out
the other end is a work order: a list short enough that a two-person O&M crew can
drive to each entry in a day, and reliable enough that they do not stop trusting
it after the third false alarm.

The classes exist because the responses differ, and this is the part that makes
the twelve-way problem worth solving rather than a binary one:

- **Vegetation and Soiling** are groundskeeping. A mower, or a wash schedule. No
  electrical work, no parts, and often no urgency — a soiled module recovers on its
  own after rain.
- **Shadowing** may not be a fault at all. It can be a permanent obstruction that
  belongs in a design review rather than a repair queue.
- **Cell, Cell-Multi, Hot-Spot, Hot-Spot-Multi and Cracking** are the module
  itself degrading. These drive warranty claims and replacement decisions, and the
  distinction between one affected cell and several is the distinction between
  monitoring and replacing.
- **Diode and Diode-Multi** are a bypass diode failure — a specific, cheap,
  fixable part, and a different technician with a different kit.
- **Offline-Module** is a whole module producing nothing, which is a string-level
  or connection problem more often than a module problem, and is usually the
  largest single production loss on the list.

Collapse those into "anomaly / no anomaly" and every one of them becomes the same
ticket, which is the same as having no triage at all.

### The base rate is what makes this hard, and the dataset hides it

D1 is 49.9% No-Anomaly. **A real site is about 2.2% anomalous.** The dataset was
built for training, so it was balanced for training, and every metric computed on
it inherits that balance.

That gap is not a detail; it is the difference between a deployable model and a
demo. Recompute per-class precision at 2.2% prevalence and the numbers move like
this:

| class | precision as measured | precision at 2.2% |
|---|---:|---:|
| Offline-Module | 0.8200 | **0.2022** |
| Hot-Spot | 0.8462 | **0.1941** |
| Shadowing | 0.8047 | 0.1977 |
| Cell | 0.6231 | 0.3022 |

Four out of five modules the model calls **Offline-Module** would not be one. A
crew dispatched on that list stops reading it. Meanwhile *recall does not move at
all* — recall is a property of the classifier given the true class, and the
prevalence shift does not touch it. Optimising macro-F1 on a balanced dataset and
reporting it as a field result is, precisely, reporting the one number the base
rate leaves alone.

The useful thing survives, and it is worth being clear about which half it is.
**Whether a module is faulty** still triages well at field prevalence: 50%
precision at 90% recall, flagging 39 modules per 1,000. **Which fault it has** does
not. That is the honest summary of what this module would be good for.

### Why a small model at all

The site classifier is not the expensive part of a real inspection pipeline — the
detector is, by a wide margin. The small CNN and the distillation experiment here
are not a claim that defect inference has to run on a robot. They exist because a
40×24 classifier trains in twenty seconds, which makes it the one model in this
project where a real ablation programme — twelve runs, three seeds, a pre-declared
noise floor — fits inside an afternoon. The methodology is the point; the model is
the thing cheap enough to apply it to.

---

## How it works

### The split, and what had to be fixed before any of it

Three data problems were found and dealt with before a single model was trained,
and they are the reason the numbers above mean anything.

**Exact duplicates.** Byte-hashing the JPEGs finds 22 identical pairs. Byte-hashing
is not sufficient and the reason is specific: a JPEG **re-encode** of the same crop
has different bytes and near-identical pixels. Searching the pixel criterion
directly — max |a − b| over all 199,990,000 pairs — finds 30 pairs at ≤ 4 DN, and
four more of them carry **contradictory labels**. Those 40 images are excluded by
connected component before the split is drawn, because deduplicating *after*
splitting leaves a near-identical pair straddling train and test.
[ADR 0003](../adr/0003-d1-is-deduplicated-at-4-dn-by-component.md).

**A measured label-noise floor.** Ten of those thirty components carry two
different labels on what is the same image to within 4 DN — No-Anomaly against
Cell, Cracking against Diode. Which label is right is not knowable from the pixels.
Both members are dropped rather than one kept, because keeping one records a coin
flip as ground truth in the split every later number is measured against.

**Near-duplicate leakage, which is a different claim.** 1,350 pairs score ≥ 0.98
cosine, and ~4% of val and test have a ≥ 0.98 neighbour in train. This was measured
rather than assumed, and it is causal: removing the 232 train neighbours costs 4.9
points of accuracy on those images, while removing the *same number of random
same-class images* costs **0.0000**. The exact zero in that control is what
separates "the model has seen these" from "these images are easy". Diluted across
the whole split the effect is 0.005 macro-F1, inside a seed spread of 0.012–0.019,
so it is not measurable on the headline number.

**This bound was measured on the 116k-parameter from-scratch CNN, and it overstates
the effect on the model that actually ships.** Both figures — the 4.9-point subgroup
cost and the 0.005 whole-split cost — come from the small CNN's removal experiment.
Measured on the same val images, the fine-tuned ResNet-18 shows about a third of the
small CNN's leaky advantage (+0.033, z = 1.80, against +0.105, z = 5.09), and on test
its class-matched leaky-minus-clean difference is +0.020 ± 0.026, z = 0.79. A stronger
model has less headroom left for a memorised near-duplicate to occupy. So 4.9 points is
the conservative number, not the shipped one, and it is quoted because a bound that
overstates the problem is the right way round for a bound to be wrong.
[ADR 0004](../adr/0004-d1-near-duplicate-leakage-is-bounded-not-removed.md).

The rule across all three, in one sentence: **exact and near-exact duplicates are
removed; statistical near-neighbours are measured and bounded, not removed.**

The split is 13,965 / 2,988 / 3,007, stratified, seed 0, pinned by SHA-256. The
data loader recomputes that hash on every load and refuses to run if it differs.

### The small CNN, and why three stages

Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2, three times, at widths 32/64/128, then
flatten and a linear layer. 115,948 parameters.

**You get three downsamples on a 40×24 input and no more.** 40×24 → 20×12 → 10×6 →
5×3; a fourth pool leaves 2×1 and a fifth leaves nothing. Every ImageNet backbone
assumes a 224² input and downsamples five times, so dropping one in unchanged
either destroys the spatial extent or forces an upsample to a resolution the data
never had. That constraint is the entire architectural story of the module, and it
is the reason the small model is written rather than borrowed.

D1 is loaded as **one uint8 tensor on the GPU** — 20,000 × 40 × 24 × 1 byte is
19.2 MB — so an epoch is a permutation and some slicing. There is no `DataLoader`
anywhere in this module. Reading 20,000 files per epoch to feed a model that trains
in twenty seconds is the wrong trade, and on Windows `num_workers > 0` brings a
process-spawn trap with it that is easier to not have than to guard against.
Normalisation statistics are computed **over the train split only**; over all
20,000 they would leak val and test pixel intensities into the normalisation.

### The imbalance ablation, and its pre-declared noise floor

D1 runs 10,000 : 175 between its largest and smallest class — **57 : 1**. Four arms
were compared: plain cross-entropy, inverse-frequency class weights, focal loss at
γ = 2, and class-balanced resampling. Three seeds each, twelve runs.

**The threshold was written down before the first arm ran.** From three earlier
control arms on this split, the seed-to-seed standard deviation of val macro-F1 is
0.012–0.019, so `configs/cls_ablation.yaml` and `docs/EXPERIMENTS.md` both declare,
at commit `93af4b5`: *any two arms differing by less than 0.02 macro-F1 are
indistinguishable and are not ranked.*

| arm | val macro-F1, mean ± std |
|---|---:|
| baseline, plain CE | 0.5980 ± 0.0085 |
| focal, γ = 2 | 0.5855 ± 0.0046 |
| resampling, balanced | 0.5804 ± 0.0039 |
| class weights | 0.5615 ± 0.0065 |

Applying the declared rule, four of the six pairwise comparisons are
indistinguishable. The two that resolve both say a treatment **lost**: baseline
beats class weights by 0.0364, and focal beats class weights by 0.0240.

**None of the three imbalance treatments beats plain cross-entropy at a learning
rate held fixed across all arms and tuned for none of them.** Baseline has the
highest mean and is not thereby the winner — it sits inside the floor of both focal
and resampling. The learning-rate qualification belongs in the sentence rather than
in a limitations paragraph below it, because the treatments being compared change
the effective gradient scale and a fixed LR is therefore not neutral between them.

**What would have flipped under the observed spread.** The measured per-arm std came
in at 0.004–0.009, tighter than the 0.012–0.019 the 0.02 floor was built from. A floor
re-derived from the tighter spread the same way lands near 0.010–0.015, and at that
threshold both comparisons this finding rests on resolve in baseline's favour:
`baseline − resampling` at +0.0175 under either end of the range, `baseline − focal`
at +0.0125 under the lower end. That would have licensed the stronger claim that plain
cross-entropy beat two of the three. **The floor declared before the first arm ran was
kept, and the weaker claim is the one reported** — a threshold re-derived from the
numbers it is about to judge is not a threshold, and this one would have moved in the
direction that flattered the result.

The ordering is not a total order, and that is correct rather than a defect:
baseline beats class weights, resampling does not, and baseline does not beat
resampling. A noise floor honestly applied produces exactly that mix of resolved
and unresolved comparisons.

The treatments are not inert — they trade *within* the tail. Resampling and class
weighting each lift Soiling, the project's worst class, by 5–6 points of F1, and
each lose 8–14 points on Hot-Spot-Multi and Hot-Spot. Macro-F1 averages that trade
to approximately nothing. At 57 : 1, reweighting changes *which* rare class the
model is willing to over-predict; it did not make the tail as a whole easier.

Two things were held fixed on purpose and are limitations rather than oversights.
**The budget** — 30 epochs, cosine decay to zero, no early stopping, selection by
best val macro-F1 — is identical across all four arms and all three seeds, because
at constant LR the baseline reaches train loss 0.14 while val macro-F1 oscillates
from about epoch 22, and four differently-overfit models would compare stopping
rules rather than losses. **The learning rate** is one value across all arms, which
is a named limitation: class weighting and focal loss both change the effective
gradient scale, so a per-arm LR sweep would be the honest comparison — and would
also confound this one, by comparing tuning effort as much as method.

### The transfer-learning baseline

ImageNet-pretrained ResNet-18, fine-tuned end to end at 3e-4. **0.6770 ± 0.0074**
on val — **+0.079 over the best from-scratch arm, four times the declared floor.**
It is the one comparison in this module that is not close, and it is the reported
model.

Two adaptations are needed, and both are choices rather than defaults.

**Aspect ratio.** 40×24 is 5:3, and a square resize to 96×96 stretches the width by
1.67×. *Taken:* the distorting resize. It is a fixed transform applied identically
to train, val and test, so it is a change of basis rather than a corruption, and
ResNet-18's own features were trained under RandomResizedCrop, which jitters aspect
ratio over 3/4 to 4/3 by construction. *Not taken:* pad to 5:3 inside the square,
then resize. It preserves the geometry but spends 40% of the canvas on constant
fill and introduces a straight, maximal-contrast synthetic edge at a fixed position
in every image — with no counterpart in ImageNet, sitting exactly where conv1's 7×7
filters respond hardest. Not ablated; picked and stated.

**One channel into a three-channel stem.** The two standard fixes are to replicate
the grey channel three times or to sum conv1's weights across the input channel,
and they are **the same function at initialisation — but only under a single shared
normalisation**, which is the condition that makes the claim true and is worth
stating rather than leaving implicit. Feeding *x* replicated three times computes
Σ*c* W[:, c] · x, which is exactly the 1-channel convolution whose weight is that
sum; the step that gets you there is that all three input channels carry the same
number. **Normalise per-channel with ImageNet's three constants and they no longer
do — the three channels become three different values, `sum_c W[:, c] * x_c` is not
`sum_c W[:, c] * x`, and the equivalence is simply false.** This project normalises with
D1's own train-split mean and std, one constant applied to the one real channel, the
same statistics `data.py` computes for every other model here — so the condition
holds and the claim stands. The module's self-check verifies it to 10⁻⁶ rather than
asserting it, and it feeds the reference stem a replicated tensor, so what it checks
is exactly the identical-channel case.

Given that condition, the two choices still diverge later in three ways: replication
keeps 9,408 weights in conv1 where 3,136 suffice, and the extra 6,272 are three copies
of one filter free to drift apart while fine-tuning on 13,965 images; replication is
what invites those ImageNet per-channel constants in the first place; and summing ties
the three filters together for the rest of training. *Taken:* sum the weights, and use
D1's own train-split mean and std.

### Calibration

A softmax output is not a probability, and this model demonstrates it: fitting a
single temperature on **val** by NLL gives **T = 2.0463**, meaning the raw model was
overconfident by a factor of two. T is fitted on val and never on test — fitting a
free parameter on test and then scoring on test measures the fit.

| | global ECE | class-balanced ECE |
|---|---:|---:|
| test, raw | 0.1215 | 0.2645 |
| test, T-scaled | **0.0211** | **0.1835** |

**The two columns are a factor of nine apart, and that gap is the finding.** Global
ECE reports a nearly perfect model. It is half No-Anomaly by construction, and
No-Anomaly is the class the model is most confident and most often right about, so
it pulls the global number toward zero and takes eleven defect classes with it.
Class-balanced ECE — the mean of the per-class ECEs, every class counting once —
says 0.184.

The per-class table says something sharper still: **one global temperature made
No-Anomaly worse in order to make everything else better**, 0.0153 → 0.0681 on the
class carrying half the data. A single scalar can only move confidence one way for
twelve differently-miscalibrated heads. And Soiling sits at 0.41 after scaling,
because the problem there is that the model is *wrong* about Soiling, not that it
is confidently wrong — temperature cannot fix that.

### Distillation

Teacher: the ResNet-18 above. Student: the 116k-parameter CNN. The loss is
α·CE(hard) + (1 − α)·T²·KL(soft), at α = 0.5 and T = 4, declared before the run and
not swept.

| | val macro-F1 | params | CPU ms / batch of 64 |
|---|---:|---:|---:|
| teacher | 0.6735 | 11,176,396 | 706.08 |
| student, from scratch | 0.5980 ± 0.0085 | 115,948 | 228.65 |
| student, distilled | 0.5987 ± 0.0075 | 115,948 | 222.86 |

**KD gain: +0.0007 against a declared floor of 0.02. Indistinguishable.** The
teacher–student gap it was meant to close is +0.0756 and is untouched.

The middle row is what makes the top and bottom rows mean anything. A distilled
student at 0.5987 is uninterpretable alone; it is only against the *same student on
hard labels at the same budget* that the gain can be read at all.

**"It did not help" and "I wrote the loss wrong" produce identical numbers here**,
and two of the three ways this loss fails do so silently. `F.kl_div` takes
log-probabilities as its input and probabilities as its target; `reduction='mean'`
divides by every element rather than by the batch, scaling the KD term by exactly
1/12 at twelve classes; and the T² factor exists because soft-target gradients
scale as 1/T², so without it α stops meaning the same thing at T = 1 and T = 4.
All three are pinned by hand-computed tests in `tests/test_cls_metrics.py`, and the
distilled arm selects different epochs from the from-scratch arm — so the term is
carrying real gradient weight, and the null result is a result.

Task 6's confusion matrix predicted where soft targets should act if anywhere:
**Cell ↔ Cell-Multi**, the largest off-diagonal pair, and exactly the kind of
similarity structure "dark knowledge" is supposed to carry. It did not act. The
distilled student moved 3.7 errors from one direction of the pair to the other and
traded 1.5 points of Cell F1 for 0.5 of Cell-Multi. The teacher's own pattern is
qualitatively different from either student's and none of it transferred. The
suspect is capacity rather than the loss — telling one hot cell from several is a
counting problem, and a model that cannot represent the distinction will not learn
it from being told the teacher's uncertainty about it — but that is a hypothesis,
and the capacity sweep that would test it was not run.

**The latency column measures the framework, not the model.** 96× fewer parameters
buys 3.1× less wall time. At 40×24 a forward pass is a few hundred microseconds of
arithmetic wrapped in Python attribute lookups and PyTorch dispatch, so per-op
overhead dominates and the student's op count, not its FLOP count, is what that
ratio reflects. Conditions: CPU, one thread, batch 64, 50 warm-up passes discarded,
median of 200. The levers that would make it a FLOP-bound measurement — INT8,
structured pruning, ONNX Runtime or TensorRT, operator fusion, batch-1 on an
embedded target — are exactly the ones this project has not pulled.

### What was written by hand, and what was not

The confusion matrix, macro-F1, per-class recall and precision, class weights,
focal loss, the KD loss, ECE and the near-duplicate search are written here and
unit-tested against hand-computed values. Macro-F1 is cross-checked against
`sklearn.metrics.f1_score` on every training run — if they disagree, the
hand-rolled one is wrong. `torch.nn` layers, `F.kl_div`, `F.cross_entropy`,
torchvision's ResNet-18 and its ImageNet weights, and
`scipy.optimize.minimize_scalar` for the temperature fit are libraries and are
named as such. The full seam is in the README table, in the same wording.

---

## What I'd say if asked

*Udit writes this section by hand. Not drafted here.*

---

## What I don't know yet

*Udit writes this section by hand. Not drafted here.*
