# ADR 0005 — D3 is split by scene, linked jointly across subsets, with SteelTile blocks and a guard band

**Date:** 2026-10-23
**Status:** Accepted
**Measured by:** `scripts/split_d3.py`
**Split:** `b5924638` (645 tiles: 445 train / 95 val / 93 test / 12 guard), pinned in
`configs/d3_split.json`.

## Context

PV01 has no official split. Its 645 tiles are 256 × 256 crops of rooftop orthophotos,
named `PV01_<x>_<y>` with integer coordinates. A random tile split would put two halves
of one roof, and sometimes the same pixels, on both sides of it. ADR 0001 said the three
rooftop subsets were "the natural scene boundary for splitting". That is false as
measured, and this ADR replaces it.

### What the coordinates mean

Nearest-neighbour distance per subset, in name units:

| Subset | tiles | NN median | NN = 2 | NN = 3 | NN = 1 or 1.41 | NN max |
|---|---:|---:|---:|---:|---:|---:|
| Brick | 138 | 2 | 75 | 50 | 4 | 82.2 |
| FlatConcrete | 413 | 2 | 370 | 26 | 12 | 105.3 |
| SteelTile | 94 | 2 | 91 | 3 | 0 | 3 |

Checked against pixels:

- Pairs 2 or 3 units apart along one axis are **edge-adjacent and do not overlap**: the
  best one-dimensional shift match is about 249 px of 256, so only a boundary strip
  agrees (about 40 pairs checked).
- Pairs 1 or 1.41 units apart are **overlapping crops of one orthophoto**: phase
  correlation finds a shift of about (88, 18) or (169, 18) px, and the overlap is
  **pixel-identical** (mean absolute difference 0.0 over 20,706 to 40,391 px) on all 10
  such pairs.

That fits name units of about 10 m: a 256 px tile at 0.1 m is 25.6 m, 2.56 units, which
the integer names round to 2 or 3. This is an inference from the spacing and the overlaps;
the dataset does not document it. Touching tiles therefore sit at most √(3² + 3²) = 4.24
apart (corner contact), and one missing tile in between gives at least 5. **L = 4.5
means "touching, including at a corner"**, and the component structure does not depend
on it: identical from L = 3.0 to 15 for FlatConcrete and SteelTile (Brick: 53 scenes at
3.0, 52 at 4.5).

### Why the linking is joint, not per subset

One U-Net is trained on all three subsets at once, and the subsets are not spatially
apart. Brick `PV01_325573_1204537` overlaps FlatConcrete `PV01_325572_1204537` and
`PV01_325574_1204537` at distance 1.0, and the overlaps are pixel-identical (mean
absolute difference 0.0 over 39,984 px and 20,944 px). FlatConcrete and SteelTile have 27
pairs within 4.5 units (minimum 2.0): a 64-tile FlatConcrete scene and a 43-tile
SteelTile scene are adjacent roofs in one orthophoto. Linking within each subset would let
those tiles land in different splits while a per-subset leakage check still passed.

Linked jointly over all 645 tiles at L = 4.5 there are **64 scenes**. The cross-subset
links merge the two adjacent roofs into one 107-tile scene and attach the Brick tile to a
FlatConcrete scene (all three overlapping tiles are in test).

### Why SteelTile needs blocks

SteelTile has three scenes under either linking: 5, 46, and 43 tiles (the last inside the
107-tile joint scene). Whole-scene assignment can only give 46 / 5 / 43 or 46 / 43 / 5:
a 5-tile val, on which a third of the best-epoch criterion would rest, or a 5-tile test.
A stricter linking distance does not help: any L below 3 cuts physically touching tiles
apart everywhere in the subset (L = 2.5 gives 29 SteelTile scenes by doing exactly that).

## Decision

1. **Joint linking** over all three subsets at **L = 4.5**; scenes are the connected
   components.
2. **SteelTile, spatial blocks** cut along each scene's long axis (the first principal
   axis of the tile coordinates):
   - the 5-tile scene → test;
   - the 46-tile scene: the 14 tiles at one end → val, the rest → train. Both ends cost
     6 guard tiles; the tie goes to the first end in the script's fixed axis orientation;
   - the 43 SteelTile tiles of the 107-tile scene: the 9 tiles at the end **farther from
     FlatConcrete** → test, the rest of the scene, all 64 FlatConcrete tiles included, →
     train. Measured: one end has 4 FlatConcrete tiles within L (minimum distance 2.0),
     the other has none (minimum distance 7.0); test takes the second.
3. **Brick and FlatConcrete, whole scenes.** Every other scene goes to one split whole,
   chosen by a seeded random search (seed 0, 20,000 draws) for tile fractions near
   70 / 15 / 15 within each subset. A scene holding tiles of two subsets goes whole.
4. **Guard band.** Every train tile within L of any val or test tile is relabelled
   `guard` and used for nothing: not training, not normalisation statistics, not the
   baseline search, not scoring.

## Result

| Subset | train | val | test | guard | fg train | fg val | fg test | fg guard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Brick | 98 | 21 | 19 | 0 | 0.0482 | 0.0647 | 0.0578 | – |
| FlatConcrete | 293 | 60 | 60 | 0 | 0.2999 | 0.4123 | 0.4391 | – |
| SteelTile | 54 | 14 | 14 | 12 | 0.5278 | 0.5692 | 0.2547 | 0.6128 |

(fg = foreground fraction, the share of panel pixels. Masks are read only after the
assignment is fixed, to fill this table; no step of the rule reads a mask.)

Leakage check, minimum name-coordinate distance from any val or test tile to any train
tile of **any** subset; every value must exceed 4.5:

| Subset | val | test |
|---|---:|---:|
| Brick | 5.099 | 5.0 |
| FlatConcrete | 18.111 | 7.0 |
| SteelTile | 5.0 | 5.831 |

No tile is in two splits. Every subset has non-empty train, val and test.

## Consequences

- **The cost is 12 SteelTile tiles**, about 13% of the subset, dropped from training so
  that no train tile touches a val or test tile. Whole-scene assignment for Brick and
  FlatConcrete adds no guard tiles, because scenes are components at L = 4.5.
- **SteelTile test is not like SteelTile train.** It is 25% foreground against 53% for
  train, because the 5-tile scene holds little array. SteelTile test numbers are read with
  that beside them, and FlatConcrete's val and test (41% and 44%) are likewise richer in
  panels than its train (30%).
- **What leakage remains.** Val and test share an orthophoto source and a region with
  train, and SteelTile val and test are cut from the same roofs as SteelTile train. No
  pixels are shared and no tiles touch across splits: every held-out tile is at least
  5.0 units from any train tile, where touching tiles are at most 4.24 apart. This is bounded leakage,
  stated, in the sense of ADR 0004, not none.
- **The split `b5924638` is pinned** in `configs/d3_split.json`; the loader recomputes the
  hash and refuses a different assignment.
