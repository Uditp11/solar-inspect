# ADR 0004 — D1's near-duplicate leakage is measured and bounded, not removed

**Date:** 2026-08-27
**Status:** Accepted
**Measured by:** `scripts/near_dup_d1.py`, `scripts/leakage_check_d1.py`
**Applies to split:** `af8781b1` (19,960 images, 13,965 / 2,988 / 3,007), which this
decision leaves unchanged.

## Context

ADR 0003 excluded 40 crops that are *the same image* to within 4 DN. It says nothing
about crops that are merely very similar, and there are far more of those: over all
199,990,000 pairs, **1,350 score ≥ 0.98 cosine** after per-image zero-meaning and L2
normalisation. Turned into the number that matters — held-out images with a near
neighbour **in train** — **4.3% of val (129 images) and 4.4% of test (133)**.

The first version of this reasoning inferred that the leakage was inert and left the
split alone on that basis. The inference was wrong, and the decision needed re-deriving
from a measurement rather than being defended.

### What was measured

**1. The effect is real, and it is causal.** Class-matched accuracy on the 129 leaky val
images against clean val images reweighted to the leaky class mix — matching on class,
because the leaky subset is 80% No-Anomaly and 14% Offline-Module — gives **+0.105 ±
0.021 (z = +5.09)**. Class-matching alone does not settle it: a ≥ 0.98 threshold
preferentially selects the most prototypical crop of a class, and prototypical crops are
easier for any model, including one that has never seen them. So the mechanism was tested
directly, three seeds per arm:

| arm | leaky-val accuracy | leaky − clean-matched | val macro-F1 |
|---|---:|---:|---:|
| full train (13,965) | 0.9509 ± 0.0045 | +0.0837 ± 0.0260 | 0.5686 ± 0.0116 |
| 232 neighbours removed (13,733) | **0.9018 ± 0.0161** | +0.0292 ± 0.0041 | 0.5636 ± 0.0187 |
| same number of random same-class images removed (13,733) | 0.9509 ± 0.0090 | +0.0779 ± 0.0092 | 0.5754 ± 0.0191 |

Removing the neighbours costs **4.9 points** on those images. Removing the same number of
random images of the same classes costs **0.0000**. **The exact zero in the control arm is
the evidence**, and it is what makes this a finding rather than an observation: it rules
out both "the leaky images are just easy" and "the smaller training set did it". Without
that arm the 4.9 points would have been ambiguous between three explanations.

**Every arm above is the 116k-parameter from-scratch CNN, so this bound overstates the
effect on the shipped model.** The removal experiment was run on the small CNN and not
repeated on the fine-tuned ResNet-18 that produces the headline number; on the same val
images the ResNet shows roughly a third of the leaky advantage, and on test the
class-matched difference is +0.020 ± 0.026 (z = 0.79). See the correction at the end of
this ADR, which is where that was measured. The 4.9 points is therefore the conservative
figure rather than the shipped one — the direction an honest bound should err in, but it
should not be quoted as if it described the ResNet.

**2. It does not move the headline number.** The whole-split cost is
0.5686 − 0.5636 = **0.005 macro-F1**, against a seed-to-seed standard deviation of
**0.012–0.019** on the same arms. By this project's own rule — a difference inside the
noise floor is not a result — the leakage is not measurable on val macro-F1 at this
budget. It is a real effect on 4.3% of the split that does not survive dilution across
the other 95.7%.

**3. The ≥ 0.98 graph does not percolate, but its components are chains.** 276
non-singleton components over the split, covering 1,054 images, largest **133**; 328
images sit in components larger than 10. So grouping was mechanically available — the
largest component is 0.67% of the split, not a giant component swallowing the dataset.
But only **1,323 of the 16,341 within-component pairs are themselves ≥ 0.98 (8.1%)**.
Cosine ≥ 0.98 is not transitive; A–B and B–C at 0.98 put A and C in one component however
dissimilar they are. The least similar pair inside the 108-image component scores
**0.6500**, against an all-pairs median of 0.4765 and p90 of 0.7405.

## Decision

**Leave the split alone. Do not drop the near-duplicates. Publish the measured bound in
the README beside the classification numbers.**

The rule, stated once so it can be quoted verbatim: **exact and near-exact duplicates are
removed; statistical near-neighbours are measured and bounded, not removed.**

## Rationale — the three options, in the order they were ruled out

### Dropping the near-duplicates is the worst option, and is off the table permanently

The ≥ 0.98 pairs concentrate in **No-Anomaly (916 within-class pairs) and Offline-Module
(209)** — the prototypical crops of the two commonest classes, and the modal field case.
Deleting them makes the benchmark quietly harder in a way no reader can see, and removes
precisely the images that represent what a real inspection mostly photographs. Curating an
evaluation set by removing the images that turned out to be easy is far harder to defend
than leaving them in and saying so. The five classes macro-F1 actually turns on —
Cell-Multi, Cracking, Hot-Spot-Multi, Soiling, Diode-Multi — have **zero** within-class
pairs at this threshold, so dropping would cost the easy classes and change nothing about
the hard ones.

### Grouping the components into single splits buys 0.005 macro-F1 and costs the finding

It was available: the largest component is 133 images, 0.67% of the split. It was declined
on two grounds.

**The size of the prize.** The whole-split effect is 0.005 macro-F1 against a seed spread
of 0.012–0.019. By §7's own noise-floor rule that is not a result. Acting on it would mean
a third split hash and a third re-measurement of every baseline, converting a controlled
causal experiment with a clean zero in its control arm into one line reading "I grouped
near-duplicates."

**The unit being grouped is not a duplicate set.** Grouping by connected component would
move the 108-image component into one split as a unit — 52 No-Anomaly, 36 Offline-Module,
7 Cell and 5 Shadowing crops, whose least similar pair scores 0.6500, a more ordinary pair
than the 90th percentile of two crops drawn at random. That would be acting decisively on
an edge set the same script already shows is a weak identity test: adjacent-by-file-id
pairs separate from the all-pairs null by only 1.2× on D1, where D2's genuine adjacency
effect was 4×. At 40×24 after zero-meaning, the dominant component of a crop is its
**layout** — a dark rectangle on a lighter surround — not its identity.

### Leaving the split alone and publishing the bound

Chosen. It keeps the benchmark as the data ships it, keeps the controlled experiment
intact as the artifact, and states the cost in the same table as the metric it qualifies.

## Consistency with ADR 0002 and ADR 0003 — the sentence that has to survive being asked

D2 was re-split because 34 of its 35 test images were **byte-identical** to a train or val
image. D1 was deduplicated at ≤ 4 DN because a JPEG re-encode of one crop is the same
image with different bytes. Neither is what is happening here. The rule is the same one in
all three places, and it is a rule about *what a pair is*, not about how large the effect
is: **exact and near-exact duplicates are removed; statistical near-neighbours are
measured and bounded, not removed.** A pair at 0.98 cosine is not asserted to be two copies
of one image, and section G of `scripts/near_dup_d1.py` is the evidence that it often is
not.

## Consequences

- **The split `af8781b1` is final** and is not touched again for the rest of the project.
- **The README's classification rows carry the bound**, in the results table rather than in
  a footnote, including the exact 0.0000 of the control arm. Rounding that to "negligible"
  would delete the evidence.
- **The test leaky/clean subgroup breakdown is computed from the single test evaluation's
  per-image predictions**, not from a separate run — one evaluation, two views of it, so
  the test-once guarantee is intact.
- **Val is now doing four jobs**: epoch selection, config selection, the temperature fit
  for calibration, and this leakage diagnostic. Acceptable at this budget, and better said
  here than discovered by a reader.
- **This bound is specific to this model at this budget.** ~~A model with more capacity to
  memorise could turn 0.005 into something larger.~~ The 4.9-point subgroup effect is the
  part that generalises; the 0.005 is not a property of the dataset.

  **Correction, 2026-08-27, after the test evaluation at `1a0fdfa`.** The struck sentence
  was a guess and it was wrong in direction. Measured on the *same* val images, swapping
  the 116k-parameter CNN for the fine-tuned ResNet-18 takes the class-matched leaky
  advantage from **+0.105 (z = +5.09) to +0.033 (z = +1.80)**; on test the same ResNet-18
  gives **+0.020 ± 0.026 (z = +0.79)**, inside its own error bar. The larger model shows
  *less* leaky advantage, not more.

  The mechanism appears to be a ceiling rather than anything about memorisation capacity.
  The leaky subset is 70–80% No-Anomaly, and the ResNet-18 scores 1.0000 on leaky
  No-Anomaly against 0.9814 on clean No-Anomaly — under two points of headroom for an
  advantage to occupy, where the small CNN left eight. A better model closes the gap the
  leakage was exploiting.

  **What this does not touch is the causal claim above**, which rests on the controlled
  removal experiment — 4.9 points under neighbour removal against 0.0000 under the
  size- and class-matched random control — and not on the size of the class-matched gap.
  What is model-specific is how much the leakage is *worth*, which this ADR always said.
  The correction is left in place rather than edited away; the same discipline as ADR 0002
  and as the retracted inference at the top of this file. Full numbers in
  `docs/EXPERIMENTS.md`.
