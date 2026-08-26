# ADR 0001 — Segmentation uses PV01 only, not PV03 or PV08

**Date:** 2026-08-26
**Status:** Accepted

## Context

The Zenodo record [5171712](https://zenodo.org/records/5171712) publishes three PV
segmentation subsets. Verified against the record's file listing on 2026-08-26:

| Subset | Archive layout | Size |
|---|---|---|
| PV01 | single `PV01.zip` | 108 MB |
| PV03 | `PV03.zip` + `PV03.z01`…`PV03.z10` | ~7.2 GB |
| PV08 | `PV08.zip` + `PV08.z01`, `PV08.z02` | ~1.2 GB |

PV03 and PV08 are **spanned (multi-part) archives**. Python's `zipfile` cannot open a
spanned archive at all, and Windows Explorer refuses them; extracting either would mean
downloading every part and shelling out to a third-party tool.

## Decision

Download and use **PV01 only**.

## Rationale

- **Resolution match.** PV01 is the 0.1 m/px UAV subset — the resolution band this
  project is about. PV03 and PV08 are satellite-resolution subsets, so they are not
  merely more of the same data.
- **Cost.** 108 MB against ~8.4 GB, plus an extraction tool this project would not
  otherwise need, plus a corresponding increase in training time on an 8 GB laptop GPU.
- **645 samples is already the binding constraint**, not 645 versus 5,000. At this
  sample count the interesting question is transfer learning versus from-scratch, and
  more tiles of a different resolution does not change that question.

## Consequences

- Module 1 is scoped to **0.1 m RGB rooftop imagery from a single geography**, and the
  README must say so rather than implying a general PV segmentation result.
- Results are reported **per rooftop subset, never pooled**. Measured foreground
  fraction varies from 5.20% (Brick) to 50.41% (SteelTile) — a 10× spread — so a pooled
  number would hide the thing that most affects the metric. See `docs/DATA.md`.
- The three subsets are the natural scene boundary for splitting. Random tile
  assignment would leak, because adjacent tiles come from the same orthophoto.
- If satellite-resolution generalisation ever matters, this decision is revisited and
  the extraction tooling cost is paid then.
