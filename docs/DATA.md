# Datasets

Everything here was measured on **2026-08-26**. No dataset is committed.

## Reproducing the data — two steps, both required

```bash
python scripts/download_data.py     # 1. fetch, checksum and extract all four datasets
python scripts/split_d2.py          # 2. rebuild D2's split and regenerate configs/d2.yaml
```

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
