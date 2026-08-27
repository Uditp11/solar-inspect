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
