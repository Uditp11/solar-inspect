# ADR 0003 — D1 is deduplicated at ≤ 4 DN by connected component, before it is split

**Date:** 2026-08-27
**Status:** Accepted
**Implemented by:** `scripts/dedup_d1.py`, `scripts/split_d1.py`, `configs/d1_dedup.json`

## Context

D2 shipped 353 files that were 252 unique images (ADR 0002). D1 was checked for the
same defect. Hashing the JPEG bytes finds **22 identical pairs** in 20,000 crops — a
small number, and the tempting conclusion is that byte-hashing is therefore
sufficient.

It is not, and the reason is specific: a JPEG **re-encode** of the same crop has
different bytes and near-identical pixels. Byte-hashing cannot see it by
construction. Searching the pixel criterion instead — max |a − b| over the 960
pixels, exhaustively, not as a subset of anything — finds **30 pairs at ≤ 4 DN**,
eight more than byte equality, and **four more pairs carrying contradictory labels**.
Three of those four put No-Anomaly against an anomaly class and two straddle a split
boundary in the split that existed before this decision.

An earlier draft of this decision asserted that byte-level and pixel-level dedup
"agreed at 22 pairs, so the cheaper one is sufficient." They agree at *exact
equality*, which is a tautology. The generalisation from it was never tested at the
boundary, and it is false.

## Decision

Exclude images before splitting, by connected component over the graph of pairs
within **4 DN**:

- a component whose members carry **one** label → keep the lexicographically first
  filename, drop the rest;
- a component carrying **more than one** label → **drop the whole component.**

**20,000 → 19,960 images.** Train/val/test become 13,965 / 2,988 / 3,007, split
sha256 `af8781b1…`.

## Rationale

### Why 4 DN, and not 0, 8 or 16

Measured, not preferred. `scripts/near_dup_d1.py` searches the criterion
exhaustively at each cap:

| max &#124;a − b&#124; | pairs | contradictory label | straddle a split |
|---:|---:|---:|---:|
| 0 DN | 22 | 6 | 10 |
| 2 DN | 25 | 9 | 11 |
| 4 DN | 30 | 10 | 15 |
| 8 DN | 234 | 13 | 111 |
| 16 DN | 10,271 | 915 | 4,928 |

The straddle column is measured against the pre-dedup split `4cbb0c3d`, which is the
state this decision was taken in and the only state in which every crop is in a split.
`scripts/near_dup_d1.py` cannot reproduce it now — 40 images are in no split — so it
reports two columns instead; `docs/DATA.md` carries both and explains the difference.

4 DN is the end of the plateau. The pair count is still flat and the contradiction
count has stopped moving; by 8 DN the population has begun to change character and
by 16 DN it is dominated by genuinely different crops that happen to be close, which
is what 915 contradictory pairs at that cap means. A threshold with a measured knee
behind it survives "why 4 and not 8"; one picked by taste does not.

### Why components and not pairs

Three copies of one image are three pairs and one group, so a dropped count derived
from a pair count double-counts. Here all 30 components turn out to be size 2 and the
two counts coincide — but that is a measurement, and it is the step whose absence
produced two different totals (19,974 and 19,964) for the same operation in an
earlier report. Neither was right; the answer is 19,960, and the pair count they both
rested on had been screened through a cosine shortlist first.

### Why a contradictory component is dropped whole

Both images are real and one of the labels is right. Which one is not knowable from
the pixels, the metadata carries nothing else, and keeping a representative would
record a coin flip as ground truth — in the split that every later number in this
project is measured against. Dropping 20 images out of 20,000 costs nothing that
matters.

### Why before the split and not after

Deduplicating after splitting leaves a near-identical pair straddling train and test.
That is D2's contamination in a new costume, and it fails invisibly.

## Consequences

- **A measured label-noise floor.** The 10 contradictory components are listed by
  filename and label in `docs/DATA.md`. This is a floor observed in the data, not an
  estimate: it says only that at least this much of D1's labelling is
  self-inconsistent among crops that are the same image to within 4 DN.
- **Every number measured before `4b65b1d` is on a different dataset.**
  `docs/EXPERIMENTS.md` gained a `split` column, retrofitted onto the existing rows,
  so a pre-dedup and a post-dedup run cannot be compared by accident.
- **D1 numbers were already not comparable to published ones** — the dataset ships no
  official split — and are now not comparable for a second reason.
- **This does not address the near-duplicate leakage at cosine ≥ 0.98**, which is a
  much larger and different population (1,350 pairs) and is not the same claim. See
  `docs/DATA.md`, "Near-duplicate leakage".
