# Datasets

Everything here was measured on **2026-08-26**. No dataset is committed.

## Reproducing the data — four steps, all required

```bash
pip install -r requirements.txt     # 0. torch comes from PyTorch's index, not PyPI
python scripts/download_data.py     # 1. fetch, checksum and extract all four datasets
python scripts/split_d2.py          # 2. rebuild D2's split and regenerate configs/d2.yaml
python scripts/dedup_d1.py          # 3. exclude D1's near-identical crops -> configs/d1_dedup.json
python scripts/split_d1.py          # 4. rebuild D1's split -> data/d1_split.json
```

Two more are measurement-only and change nothing the models read — run them to
reproduce the D1 duplicate and leakage numbers below, or skip them:
`scripts/near_dup_d1.py` (needs step 4 first, since it reports leakage per split)
and `scripts/leakage_check_d1.py`.

**Step 0 is not optional either.** `requirements.txt` carries an
`--extra-index-url` line for `https://download.pytorch.org/whl/cu128`, because the
pinned `torch==2.11.0+cu128` does not exist on PyPI's default index. Installing
without it resolves to the CPU-only wheel and every GPU run in this repo then runs
on the CPU instead of failing.

**Steps 3 and 4 are not optional either, and 3 comes before 4.** Splitting first and
deduplicating afterwards leaves a near-identical pair straddling train and test, which
is D2's contamination problem in a new costume. `configs/d1_dedup.json` lists the 40
images excluded and why; `configs/d1_split.json` is committed but only *pins* the split
— it holds the seed, the ratios, the counts and a SHA-256. The 19,960-entry assignment
lives in `data/d1_split.json`, which is gitignored.
`src/solar_inspect/classification/data.py` recomputes the hash on load and refuses to
run if it differs, so a clone that skips this step gets a missing-file error rather
than a quietly different validation set.

**Step 2 is not optional.** D2's published splits are contaminated (see below) and are
not used by anything in this project. `configs/d2.yaml` points at
`data/d2_split/`, which only exists after step 2 — so a clone that runs only step 1 has
the raw data on disk and no usable detection dataset.

`python scripts/download_data.py --verify` re-checks the archives against the checksums
below without touching the network. `scripts/split_d2.py` is deterministic and
idempotent; re-running it rebuilds the same tree and leaves the working tree clean.

Total on disk: ~250 MB extracted, ~140 MB of archives, plus a hardlinked split tree that
costs effectively nothing.

| # | Name | Source | Licence (as published) | Archive SHA-256 |
|---|---|---|---|---|
| D1 | InfraredSolarModules | [github.com/RaptorMaps/InfraredSolarModules](https://github.com/RaptorMaps/InfraredSolarModules) | MIT | `b82c706b…5db05b5e` |
| D2 | Thermal PV Panel Detection (UAV) | [zenodo.org/records/16420123](https://zenodo.org/records/16420123) | CC-BY-4.0 | `8a7ed5ee…223f3474` |
| D3 | PV segmentation — PV01 only | [zenodo.org/records/5171712](https://zenodo.org/records/5171712) | CC-BY-4.0 | `01cfb64e…a753a93b` |
| D4 | Solar Power Generation Data | Kaggle `anikannal/solar-power-generation-data` | `copyright-authors` — **not an open licence** | `fecbfdd2…55690c32` (unstable, see D4) |

---

## D1 — InfraredSolarModules

- 20,000 JPGs + `module_metadata.json` (20,000 entries, all parse).
- **Orientation: 40 rows × 24 columns.** PIL reports `im.size == (24, 40)`, which is
  (W, H) — so the crops are **24 px wide by 40 px tall**, portrait. Verified across
  2,000 images with zero exceptions. The dataset README's "24 by 40" is W×H; a conv
  stack written against `(H, W) = (24, 40)` would be transposed.
- Mode `L` (single channel), `uint8`. No temperature values anywhere.

### Class counts

| Class | Count | Share |
|---|---:|---:|
| No-Anomaly | 10,000 | 50.00% |
| Cell | 1,877 | 9.38% |
| Vegetation | 1,639 | 8.20% |
| Diode | 1,499 | 7.50% |
| Cell-Multi | 1,288 | 6.44% |
| Shadowing | 1,056 | 5.28% |
| Cracking | 940 | 4.70% |
| Offline-Module | 827 | 4.13% |
| Hot-Spot | 249 | 1.25% |
| Hot-Spot-Multi | 246 | 1.23% |
| Soiling | 204 | 1.02% |
| Diode-Multi | 175 | 0.88% |

**Imbalance 10,000 : 175 = 57.1 : 1.**

**D1 has no official train/test split.** Any comparison against a published number on
this dataset is invalid, because it is not the same split. Say so before being asked.

### Duplicates — byte-level hashing is not enough here

Hashing the JPEG bytes finds **22 identical pairs**, and the tempting conclusion is that
this is a clean dataset and the cheap check settles it. It does not. A JPEG **re-encode**
of the same crop has different bytes and near-identical pixels, so byte-hashing cannot
see it by construction. Searching the pixel criterion instead — max |a − b| over the 960
pixels, exhaustively over all 199,990,000 pairs, not as a subset of a similarity
shortlist — gives:

| max &#124;a − b&#124; | pairs | contradictory label | straddle a split (live) | a member excluded | byte-identical |
|---:|---:|---:|---:|---:|---:|
| 0 DN | 22 | 6 | 0 | 22 | 22 |
| 2 DN | 25 | 9 | 0 | 25 | 22 |
| 4 DN | 30 | 10 | 0 | 30 | 22 |
| 8 DN | 234 | 13 | 88 | 47 | 22 |
| 16 DN | 10,271 | 915 | 4,770 | 183 | 22 |

**The straddle column is two columns, and the reason is a measurement that changed under
its own feet.** When the threshold was chosen, all 20,000 crops were in a split and one
column counted pairs whose two members landed in different ones: **10 / 11 / 15 / 111 /
4,928** at 0 / 2 / 4 / 8 / 16 DN. Those figures are what the 4 DN choice was argued
against and they are no longer reproducible, because 40 images are now in no split at
all. Counting an excluded image as its own bucket makes the exclusion itself look like
straddling and reports 20 pairs at 4 DN rather than 15 — a number that measures the fix,
not the problem. So: **straddle (live)** counts pairs both of whose members are still in
a split, which is 0 at ≤ 4 DN by construction and is the point of the dedup; **a member
excluded** counts the rest. Re-running `scripts/near_dup_d1.py` reproduces this table.

Eight more pairs at 4 DN than byte equality finds, and **four more contradictory ones** —
three of those pitting No-Anomaly against an anomaly class, two straddling a split
boundary in the split that existed before this exclusion. `scripts/dedup_d1.py` therefore deduplicates at **≤ 4 DN**, by connected
component, before the split is drawn:
[ADR 0003](adr/0003-d1-is-deduplicated-at-4-dn-by-component.md) carries the reasoning for
the threshold and for the drop rules. **20,000 → 19,960 images**, 13,965 / 2,988 / 3,007.

All 30 components at 4 DN turn out to be **size 2**, so here the pair count and the group
count coincide. That is a measurement and not an assumption: at 16 DN they would not.

### The measured label-noise floor

Ten of the thirty components carry **two different labels on what is the same image to
within 4 DN**. Both images are real; which label is right is not knowable from the pixels,
and the metadata carries nothing else. The whole component is dropped rather than one
member kept, because keeping one would record a coin flip as ground truth in the split
every later number is measured against.

| Files | Labels |
|---|---|
| `10147.jpg` + `4616.jpg` | Cell-Multi / No-Anomaly |
| `1124.jpg` + `7387.jpg` | Cracking / Diode |
| `13046.jpg` + `1594.jpg` | Diode / No-Anomaly |
| `1448.jpg` + `16999.jpg` | Diode / No-Anomaly |
| `15571.jpg` + `5687.jpg` | Cell / No-Anomaly |
| `161.jpg` + `4518.jpg` | Cell-Multi / Offline-Module |
| `3915.jpg` + `7027.jpg` | Cell-Multi / Cracking |
| `450.jpg` + `7484.jpg` | Cracking / Offline-Module |
| `483.jpg` + `6888.jpg` | Hot-Spot / Offline-Module |
| `5171.jpg` + `543.jpg` | Cell / Offline-Module |

This is a **measured** floor on D1's label noise, not an estimated one, and it is a floor
in a narrow sense: it says only that at least this much of the labelling is
self-inconsistent *among crops that are the same image*. It says nothing about how often
two genuinely different crops are labelled inconsistently, which is the larger and
unmeasurable quantity. Seven of the ten involve No-Anomaly or Offline-Module.

### Near-duplicate leakage — measured, and live

Deduplicating at 4 DN removes crops that are *the same image*. It says nothing about crops
that are merely very similar, and there are far more of those. Zero-meaning and
L2-normalising each crop and taking the cosine over all 199,990,000 pairs:

| | mean | median | p90 | p99 | p99.99 |
|---|---:|---:|---:|---:|---:|
| all pairs | 0.4335 | 0.4765 | 0.7405 | 0.8685 | 0.9585 |

**680,604 pairs at ≥ 0.90, 40,438 at ≥ 0.95, 1,350 at ≥ 0.98, 161 at ≥ 0.99.**

Turned into the number that matters — held-out images that have a near neighbour **in
train** — this is the **leakage ceiling**:

| split | n | ≥ 0.90 | ≥ 0.95 | ≥ 0.98 | ≥ 0.99 |
|---|---:|---:|---:|---:|---:|
| val | 2,988 | 76.5% | 36.4% | **4.3%** (129) | 0.7% |
| test | 3,007 | 78.0% | 36.3% | **4.4%** (133) | 0.5% |

#### Where the similar pairs are, and where they are not

The 1,350 pairs at ≥ 0.98 are not spread across the classes. **916 are
No-Anomaly ↔ No-Anomaly and 209 are Offline-Module ↔ Offline-Module.** Counting pairs
whose two members share a class — the only ones that can inflate a class's recall by
memorisation rather than merely confuse two classes:

| Class | n | within-class pairs ≥ 0.98 |
|---|---:|---:|
| No-Anomaly | 10,000 | 916 |
| Offline-Module | 827 | 209 |
| Shadowing | 1,056 | 32 |
| Diode | 1,499 | 14 |
| Vegetation | 1,639 | 12 |
| Cell | 1,877 | 4 |
| Hot-Spot | 249 | 1 |
| **Cell-Multi, Cracking, Hot-Spot-Multi, Soiling, Diode-Multi** | 940–1,288 / 175–246 | **0** |

The five classes with **zero** within-class near-duplicates are the low-support classes
macro-F1 actually turns on.

#### It is not consecutive-frame duplication

The obvious mechanism — the same physical module photographed in adjacent frames — does
not fit. Only **136** of the 1,350 pairs have file ids within 1 of each other; **886 are
more than 100 apart**. And ordering the whole dataset by file id, adjacent pairs score a
mean cosine of **0.5162** against the all-pairs **0.4335** — a separation of 1.2×, where
D2's genuine adjacency effect was 4× (0.127 vs 0.032, ADR 0002). At 40×24, after per-image
zero-meaning, the dominant component of a crop is its **layout** — a dark rectangle on a
lighter surround — not its identity. Raw-pixel cosine is a weak identity test on
near-uniform imagery, which is itself the finding.

#### The leakage is live, and it does not move the headline number

Both halves of that are measured, on the committed split, by
`scripts/leakage_check_d1.py`.

**It is live.** Scoring the baseline on the 129 leaky val images against the clean 2,859
*reweighted to the leaky class mix* — matching on class, because the leaky subset is
80% No-Anomaly and 14% Offline-Module and a raw comparison would be meaningless —
gives **0.9535 against 0.8484, a difference of +0.105 ± 0.021 (z = +5.09)**.

Class-matching is not enough on its own: a ≥ 0.98 threshold preferentially selects the
most prototypical crop of a class, and prototypical crops are easier for any model,
including one that has never seen them. So the mechanism was tested directly. Retraining
with the 232 train images that are ≥ 0.98 neighbours of those 129 deleted, three seeds
per arm, against a control that deletes the **same number of random images of the same
classes**:

| arm | leaky-val accuracy | leaky − clean-matched | val macro-F1 |
|---|---:|---:|---:|
| full train (13,965) | 0.9509 ± 0.0045 | +0.0837 ± 0.0260 | 0.5686 ± 0.0116 |
| neighbours removed (13,733) | **0.9018 ± 0.0161** | +0.0292 ± 0.0041 | 0.5636 ± 0.0187 |
| random removed (13,733) | 0.9509 ± 0.0090 | +0.0779 ± 0.0092 | 0.5754 ± 0.0191 |

Removing the neighbours costs **4.9 points** on those images. Removing the same number of
random same-class images costs **0.0**. The advantage is caused by the near-duplicates in
train, not by the images being easy and not by the smaller training set.

**And it does not move the headline number.** The whole-split cost is
0.5686 − 0.5636 = **0.005 macro-F1**, against a seed-to-seed standard deviation of
0.012–0.019 on the same arms. By this project's own rule that a difference inside the
noise floor is not a result, the leakage is **not measurable on val macro-F1 at this
budget** — it is a real effect on 4.3% of the split that does not survive being diluted
across the other 95.7%.

**Both sentences are needed.** "There is near-duplicate leakage in D1" and "it is worth
0.005 macro-F1" are different claims, and only the second one bounds what it costs.

#### The graph does not percolate, and its components are chains rather than clusters

Grouping each near-duplicate cluster into a single split is only an option if the clusters
are small. 1,323 edges over 19,960 nodes can percolate — cosine ≥ 0.98 is **not
transitive**, so A–B and B–C at 0.98 put A and C in one component however dissimilar A and
C are — and with 916 of the pairs inside No-Anomaly alone a single giant component was
plausible enough that the decision below was not allowed to assume otherwise.

Measured (`scripts/near_dup_d1.py`, section G): **276 non-singleton components covering
1,054 images, largest 133.**

| size | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 13 | 16 | 34 | 108 | 133 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| components | 192 | 32 | 18 | 7 | 10 | 4 | 4 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 |

**670 images sit in components larger than 2, and 328 in components larger than 10** — 7
components carry those 328. 38 of the 129 leaky val images and 47 of the 133 leaky test
images are in a component larger than 10.

So the graph does **not** collapse into one giant component: the largest is 0.67% of the
split, and grouping is mechanically available. What the components are *not* is clusters of
duplicates. Only **1,323 of the 16,341 within-component pairs are themselves ≥ 0.98
(8.1%)** — a component is a chain, and transitivity did most of the work:

| size | least similar pair inside | mean | splits | classes |
|---:|---:|---:|---|---|
| 133 | **0.7704** | 0.9288 | 87 / 24 / 22 | No-Anomaly 122, Diode 5, Hot-Spot 3, Shadowing 1 |
| 108 | **0.6500** | 0.8977 | 78 / 9 / 21 | No-Anomaly 52, Offline-Module 36, Cell 7, Shadowing 5 |
| 34 | 0.8759 | 0.9547 | 26 / 5 / 3 | No-Anomaly 28, Offline-Module 4, Hot-Spot-Multi 2 |
| 16 | 0.8204 | 0.9528 | 14 / 1 / 1 | No-Anomaly 15, Vegetation 1 |
| 13 | 0.9478 | 0.9764 | 10 / 1 / 2 | No-Anomaly 13 |

(splits are train / val / test.)

The least similar pair inside the 108-image component scores **0.6500**, against an
all-pairs **median of 0.4765** and a **p90 of 0.7405** — it is a more ordinary pair than
the 90th percentile of two crops drawn at random. Grouping by connected component would
move those two images into the same split together, along with 36 Offline-Module and 7
Cell crops that are in there because a No-Anomaly chain reached them. That is the same
finding as the adjacent-frame result above, seen from the other side: **raw-pixel cosine
is a weak identity test on near-uniform 40×24 imagery**, and a threshold on it does not
define a set of duplicates.

#### And on test, from the single test evaluation, it is smaller still

The equivalent class-matched breakdown on **test** comes from the per-image predictions of
the one test pass (`c1df507`, run `20260827T205903Z`), not from a second evaluation.

| | model | n leaky / clean | leaky acc | clean, class-matched | difference | z |
|---|---|---:|---:|---:|---:|---:|
| val | small CNN | 129 / 2,859 | 0.9535 ± 0.0187 | 0.8484 ± 0.0097 | +0.105 ± 0.021 | +5.09 |
| val | ResNet-18 | 129 / 2,859 | 0.9612 ± 0.0170 | 0.9282 ± 0.0068 | +0.033 ± 0.018 | +1.80 |
| **test** | **ResNet-18** | **133 / 2,874** | **0.9098 ± 0.0248** | **0.8893 ± 0.0076** | **+0.020 ± 0.026** | **+0.79** |

The two val rows are the same images, so the difference between them is the **model**, not
the split: the fine-tuned ResNet-18 shows a third of the small CNN's leaky advantage, and
on test it is inside its own error bar. The likely reason is a ceiling — the leaky subset
is 70–80% No-Anomaly, and the ResNet-18 scores 1.0000 there against 0.9814 on clean
No-Anomaly, under two points of room for any advantage to sit in.

This qualifies the bound and leaves the causal claim alone: the causal claim rests on the
removal control (4.9 points lost, 0.0000 under random removal), not on the size of the
class-matched gap. ADR 0004 predicted the opposite of this and carries the correction.

#### The decision

**The split is not changed. The near-duplicates are not dropped. The bound is published.**
See [ADR 0004](adr/0004-d1-near-duplicate-leakage-is-bounded-not-removed.md) for the three
options and why this one. The rule, in one line: **exact and near-exact duplicates are
removed; statistical near-neighbours are measured and bounded, not removed.**

The earlier reasoning for leaving the split alone, which rested on the leakage being inert,
was **wrong** — it was inferred, then measured, and the measurement contradicted it. The
decision survives; the argument for it does not, and the replacement is above. The same
failure ADR 0002 exists to record.

---

## D2 — Thermal PV Panel Detection (UAV)

Archive is **15,172,606 bytes (15.2 MB)**. The "19.5 GB" shown on the Zenodo page is a
download-statistics field, not the archive size.

### Annotation format: already YOLO. No conversion needed.

This was the project's biggest unknown and it is closed. The archive is a **Roboflow
export** — every filename carries a `.rf.<hash>` infix. Layout:

```
d2/{train,val,test}/images/*.jpg
d2/{train,val,test}/labels/*.txt
```

Label files are one box per line, exactly five whitespace-separated tokens:

```
0 0.478078125 0.2775390625 0.18737499999999999 0.11710937499999999
```

`class_id cx cy w h`, normalised to [0, 1]. Verified over all 26,678 boxes:
**every line has exactly 5 tokens** (so no polygons/segmentation), **every box is
class id `0`** (single class), and **zero boxes fall outside [0,1] or extend past the
image edge**. The 1.5 h conversion budget is not needed.

**Missing from the archive:** there is no `data.yaml` and no README — only images and
labels. Ultralytics needs one, so `scripts/split_d2.py` generates `configs/d2.yaml`
alongside the split it defines. Single class, `0: panel`.

### Images

- 353 files, all `640 × 512`, PIL mode `RGB`, `uint8`.
- **All three channels are byte-identical in all 353 images** — this is a greyscale
  render replicated across RGB, not colour data. Global pixel range across the whole
  set is `0..245`.
- ~43 KB per frame at 640×512 confirms 8-bit rendered imagery. **There is no radiometric
  content here**, which is why the project's field is `delta_dn_uncalibrated` and never
  a temperature.

### Boxes — two counts, and the second is the one that matters

- **26,678 across the 353 published files**, matching the dataset's stated count.
  This number **counts duplicates** (see below).
- **19,525 across the 252 unique images.** Every detection denominator in this project
  uses this one.
- Published per-split: train 18,487 · val 5,828 · test 2,363. These splits are not used
  — see the re-split below.
- Median box **56.2 × 36.6 px**. (For reference: 12 µm pitch × 30 m / 9.1 mm ≈ 4 cm/px,
  so a 2 m × 1 m module lands around 50 × 25 px. The measured median is the right size.)
- Mean **75.6 boxes/image**, median 64 — but **max 584**.

### ⚠ Two problems found

**1. The published splits are contaminated. 34 of the 35 test images also appear in
train or val, byte-for-byte identical.**

| | files | unique by content |
|---|---:|---:|
| train | 235 | 235 |
| val | 83 | 83 |
| test | 35 | 35 |
| **union** | **353** | **252** |

- **28 of 35 test images (80%) are byte-identical to a train image.**
- **6 more (17%) are byte-identical to a val image.**
- **Exactly 1 test image is genuinely held out.**
- 67 of 83 val images (81%) are byte-identical to a train image; 10 val images are
  held out from both.
- Verified by SHA-256 of the JPG bytes. The duplicated pairs also carry identical
  label files (0 of 28 train/test pairs disagree), so it is straight duplication
  rather than a re-annotation.
- The source frame name survives in the filename (`DJI_20230213153545_0028_T_JPG`),
  and 101 source frames appear in two splits each.

Evaluating the published test split measures memorisation, not generalisation.
**The published splits are discarded.** See "The split actually used" below and
ADR 0002.

**2. Eight unique frames carry more than 300 boxes (max 584), against Ultralytics'
`max_det=300` default.** (Nine of the 353 *files*, but two of those are the same image;
8 of the 252 unique frames, 3.2%.) The design note that `max_det` truncation is "fine
at D2's ~75 panels/frame" holds for the mean and not for the tail. This is a measured
bug on this dataset, not a hypothetical one. `configs/det_yolo.yaml` sets
**`max_det: 1000`**.

**Also:** two train images have no label file at all
(`DJI_20230213170209_0001_T_JPG…`, `DJI_20230213170213_0003_T_JPG…`), so there are 233
label files for 235 train images. Ultralytics reads a missing label file as a
background image with zero objects. No label file anywhere is empty-but-present.

Frames are non-overlapping, so a real flyover sequence cannot be reconstructed from them.

---

## D3 — PV01 (segmentation)

**PV01 only, 108,133,319 bytes.** PV03 (7.2 GB) and PV08 (5.8 GB) were not downloaded:
they are spanned archives (`PV03.z01`…`PV03.z10` plus `PV03.zip`) that Python's
`zipfile` cannot open. Confirmed against the Zenodo file listing. See `docs/adr/`.

- 645 image/mask pairs, 1,290 BMP files, in three rooftop subsets.
- Images `256 × 256 × 3` RGB `uint8`; masks `256 × 256`, PIL mode `L`, single channel.

### Mask class codes — the spec's list is wider than what PV01 contains

The distinct pixel values present across all 645 PV01 masks are exactly:

```
{0, 211, 212, 213}
```

The codes `11/12` and `111/121–125` do not appear in PV01 — they belong to the PV03 and
PV08 subsets that were not downloaded. Binarisation for this project is therefore
`mask != 0`, and the unit test should assert against `{0, 211, 212, 213}` rather than
against the full published code list.

### Foreground fraction — 30.00% positive

| Subset | Masks | Positive pixels |
|---|---:|---:|
| PV01_Rooftop_Brick | 138 | **5.20%** |
| PV01_Rooftop_FlatConcrete | 413 | **33.65%** |
| PV01_Rooftop_SteelTile | 94 | **50.41%** |
| **All PV01** | **645** | **30.00%** |

**This kills the "pixel accuracy is misleading here" line as a statement about PV01 as
a whole.** A null all-background model scores 70.0% pixel accuracy on pooled PV01 —
mediocre, not deceptively high — while its IoU is 0. The sentence is only defensible
per subset, and it points in opposite directions:

- On **Rooftop_Brick** (5.20% positive) it is true: the null model scores **94.8%**
  pixel accuracy at 0 IoU.
- On **Rooftop_SteelTile** (50.41% positive) it is false in the other direction: the
  null model scores **49.6%**, worse than a coin flip, because the positive class is
  the majority.

The 10× spread in class balance across three subsets of the same dataset is itself the
argument for the "report per subset, never pooled" rule — pooling hides it.

---

## D4 — Solar Power Generation Data

Four CSVs, 1.9 MB zipped. Kaggle re-zips server-side on each request, so the archive
SHA-256 above is recorded for provenance but is **not** treated as a gate by
`download_data.py`; the CSV contents are what matter.

**Licence is `copyright-authors`** as reported by the Kaggle API — "Data files ©
Original Authors", i.e. not an open licence. The data is not redistributed here (it is
gitignored); only the download script is committed.

| | Plant 1 | Plant 2 |
|---|---|---|
| Generation rows | 68,778 | 67,698 |
| Weather rows | 3,182 | 3,259 |
| Inverters (`SOURCE_KEY`) | 22 | 22 |
| Weather sensors | **1** | **1** |
| Span | 2020-05-15 → 2020-06-17 (34 days) | 2020-05-15 → 2020-06-17 (34 days) |
| Rows with `IRRADIATION > 0` | 38,376 (55.8%) | 38,722 (57.2%) |

### The datetime format split is narrower than expected

Only **one of the four files** deviates. `Plant_1_Generation_Data.csv` uses
`15-05-2020 00:00` (`%d-%m-%Y %H:%M`). The other three — both weather files and
`Plant_2_Generation_Data.csv` — use `2020-05-15 00:00:00` (`%Y-%m-%d %H:%M:%S`).
Parsing without an explicit `format=` will silently swap day and month for the first
12 days of the month.

### Plant 1's DC_POWER is mis-scaled by ~10×

| | Plant 1 | Plant 2 |
|---|---:|---:|
| max DC_POWER | 14,471 | 1,421 |
| max AC_POWER | 1,411 | 1,385 |
| **median AC/DC efficiency** | **9.78%** | **97.84%** |

Plant 2 sits where an inverter should (97.8%). Plant 1 computes to 9.78%, and
14,471 / 1,421 ≈ 10.2 — Plant 1's DC_POWER is out by a factor of about ten. Any
pooled analysis across both plants inherits this.

### Collinearity is real

`corr(IRRADIATION, MODULE_TEMPERATURE)` = **0.962** (Plant 1), **0.947** (Plant 2).
Forming `XᵀX` squares an already-poor condition number, so use `cho_solve`/`lstsq`
rather than an explicit inverse.

One weather sensor per plant means all 22 inverters share an identical `X` at each
timestamp — a pooled OLS residual ranking is a `groupby` in disguise. Fit per-inverter
or use inverter fixed effects.

---

## The split actually used — decided, see ADR 0002

Deduplicate by content hash, then segment the 252 unique frames into **sorties** at
acquisition-time gaps longer than 300 s, and assign whole sorties to splits.
Reproduce with `python scripts/split_d2.py`; the result is `configs/d2.yaml`.

| Sortie | Window | Images | Boxes | Mean/frame | Split |
|---:|---|---:|---:|---:|---|
| 1 | 15:29–15:30 | 7 | 1,515 | 216.4 | train |
| 2 | 15:35–15:47 | 51 | 4,929 | 96.6 | **val** |
| 3 | 15:57–16:21 | 129 | 9,097 | 70.5 | train |
| 4 | 16:34–16:53 | 63 | 3,984 | 63.2 | **test** |
| 5 | 17:02 | 2 | 0 | 0.0 | train |

**138 / 51 / 63 images · 10,612 / 4,929 / 3,984 boxes · ≈55/20/25.**

Verified through Ultralytics' own loader: three distinct `labels.cache` files, image and
box counts as above, 2 background frames in train.

### The reason is acquisition-condition shift, not overlap

An earlier draft justified grouping on the grounds that consecutive frames are
near-duplicates. **That was asserted rather than measured, and it is false.** 32×32
normalised correlation, adjacent unique frames (≤20 s apart) versus random pairs:

| | mean | median | p90 |
|---|---:|---:|---:|
| adjacent (≤20 s) | 0.127 | 0.080 | 0.696 |
| random pairs | 0.032 | 0.012 | 0.250 |

15 of 213 adjacent pairs exceed 0.8; 3 exceed 0.9. Consistent with the dataset paper's
non-overlapping-frames claim. The reason that does survive: the flight spans 15:29–17:02,
so sun angle and thermal loading shift materially across it, and holding out a whole
acquisition window tests the thing that genuinely differs between splits. It is also
nearly free.

**Consequence: detection numbers here are not comparable to any published number on D2**,
because they are not the published split. That has to be stated wherever one appears.

**70/20/10 is unreachable** given sortie sizes of 7/51/129/63/2, and forcing it would
mean splitting a sortie, which gives up the only property the decision buys. 63 test
images is also a better base for "what's your uncertainty on that?" than the ~26 a plain
70/20/10 over 252 frames would have produced.
