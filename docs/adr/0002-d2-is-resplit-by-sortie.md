# ADR 0002 — D2's published splits are discarded and replaced by a sortie-grouped split

**Date:** 2026-08-26
**Status:** Accepted
**Implemented by:** `scripts/split_d2.py`, `configs/d2.yaml`

## Context

D2 ships 353 image files in published `train`/`val`/`test` folders. Hashing the JPG
bytes shows they are only **252 unique images**:

- **28 of the 35 test images (80%) are byte-identical to a train image.**
- 6 more (17%) are byte-identical to a val image.
- **Exactly one test image is genuinely held out.**
- 67 of 83 val images (81%) are byte-identical to a train image.
- Duplicated pairs carry identical label files, so this is duplication rather than
  re-annotation.

Evaluating on the published test split measures memorisation. It also defeats the
project's "evaluate the test split exactly once" rule invisibly: the procedure would be
followed to the letter and the number would still be meaningless.

The published box count of **26,678 counts duplicates**. Over the 252 unique frames
there are **19,525 boxes**, and that is the denominator for every detection number from
here on.

## What we are *not* claiming

An earlier draft of this decision argued for grouping because "consecutive frames are
near-duplicates even when not byte-identical." **That was asserted, not measured, and
it is false.** 32×32 normalised correlation between temporally adjacent unique frames
(≤20 s apart) against random pairs:

| | mean | median | p90 |
|---|---:|---:|---:|
| adjacent (≤20 s) | 0.127 | 0.080 | 0.696 |
| random pairs | 0.032 | 0.012 | 0.250 |

Only 15 of 213 adjacent pairs exceed 0.8 and 3 exceed 0.9. This is consistent with the
dataset paper's statement that frames are non-overlapping. **Visual overlap is not the
reason to group-split**, and stating that it is would be a claim that does not survive
being checked.

## Decision

Deduplicate by content hash, then segment the 252 unique frames into **sorties** by
splitting the acquisition timestamps at gaps longer than 300 s, and assign whole
sorties to splits.

There are five sorties, and they are very unequal:

| Sortie | Window | Images | Boxes | Mean/frame | Split |
|---:|---|---:|---:|---:|---|
| 1 | 15:29–15:30 | 7 | 1,515 | 216.4 | train |
| 2 | 15:35–15:47 | 51 | 4,929 | 96.6 | **val** |
| 3 | 15:57–16:21 | 129 | 9,097 | 70.5 | train |
| 4 | 16:34–16:53 | 63 | 3,984 | 63.2 | **test** |
| 5 | 17:02 | 2 | 0 | 0.0 | train |

Resulting split — **138 / 51 / 63 images**, **10,612 / 4,929 / 3,984 boxes**, ≈55/20/25.

## Rationale

- **The real reason to group is acquisition-condition shift, not overlap.** The flight
  spans 15:29–17:02. Sun angle and thermal loading move materially across ninety
  minutes of a February afternoon, and that is a genuine distribution shift for a
  thermal detector. Holding out a whole acquisition window tests the thing that
  actually differs between splits. It is also nearly free, which is why it is worth
  doing even though the leakage argument does not apply.
- **Test is a contiguous, temporally later block.** This is the "next flight" framing,
  which is the honest version of what a held-out set is for.
- **Sortie 1 goes to train** because 216 boxes/frame is a density outlier and it should
  not be allowed to distort an evaluation.
- **Sortie 5 goes to train** as background images. The two 17:02 frames are the ones
  with no label file; they are tail-end transit shots with no panels in view, not an
  annotation bug. Ultralytics reads a missing label file as a background image, which
  is exactly the right treatment.
- **A 63-image test set beats the ~26 that a plain 70/20/10 over 252 frames would give.**
  "What's your uncertainty on that?" is easier to answer at 63.
- **70/20/10 is unreachable group-wise** given sortie sizes of 7/51/129/63/2. 55/20/25
  is what the data actually permits, and forcing the target ratio would mean splitting
  a sortie, which would give up the only property this decision is buying.

### Deliberate side benefit

3 of the 8 frames carrying more than 300 boxes land in **test**. Ultralytics' default
`max_det=300` would silently truncate them. `configs/det_yolo.yaml` sets
**`max_det: 1000`**, and the truncation becomes a measured effect in the headline
evaluation rather than a footnote about a risk.

## Consequences

- **Detection numbers are not comparable to any published number on D2**, because they
  are not the published split. This is stated wherever a detection number appears.
- The split is defined by committed code (`scripts/split_d2.py`) rather than by a
  committed file list, since `data/` is gitignored. The script is deterministic.
- The split is **materialised as a directory tree** rather than as path lists.
  Ultralytics derives its `labels.cache` path from the first image's label directory,
  so lists spanning several source folders make two splits collide on one cache file.
  Verified after the change: three distinct caches, 138/51/63 images and
  10,612/4,929/3,984 boxes loaded, 2 background frames in train.
- Files are hardlinked, so the tree costs ~0 bytes on NTFS.
- Effective training set is 138 images. That is small, and it strengthens rather than
  weakens the argument for fine-tuning a pretrained detector instead of training one.
