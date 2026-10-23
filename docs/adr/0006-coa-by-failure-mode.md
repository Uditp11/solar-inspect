# ADR 0006 — Triage assigns the Class of Abnormality by failure mode, from public sources

**Date:** 2026-10-23
**Status:** Accepted
**Implemented in:** `src/solar_inspect/analytics/triage.py` (`COA_TABLE`, checked against the
table below by `tests/test_triage.py`)

## Context

Module 5's triage turns a classified finding into a work order. The usual scheme in PV
thermography is the three Classes of Abnormality (CoA) of IEC TS 62446-3. That standard is
not public and has not been read for this project. Its three classes are described in public,
peer-reviewed work; this project takes them from Glavaš, Žnidarec, Šljivac and Veić,
"Application of Infrared Thermography in an Adequate Reusability Analysis of Photovoltaic
Modules Affected by Hail", *Applied Sciences* 12(2), 745 (2022), doi:10.3390/app12020745
(open access), which states them with their recommended actions:

| CoA | name | recommended action |
|---|---|---|
| 1 | no abnormality | no imminent action |
| 2 | thermal abnormality | check the cause and, if needed, rectify within a reasonable period |
| 3 | safety-relevant thermal abnormality | interrupt operation without delay, check the cause, rectify |

In practice a class is assigned from a pattern and a measured temperature difference. The
data here cannot carry one. D1, the classifier's data, is 8-bit rendered imagery from mixed
sensors (docs/DATA.md); its pixel values are not radiometric and are not comparable between
crops. Any rule keyed on a thermal threshold would be keyed on a number this project does
not have.

## Decision

**The CoA is assigned by failure mode**: each of D1's twelve classes maps to one CoA, one
extent and one recommended action, and nothing about a finding's pixel values changes that.
`coa_basis` on every `Triage` record is the string `"failure_mode"`.

**CoA 3 goes to the modes that can be safety-relevant.** Whether a given Diode or Hot-Spot
finding *is* safety-relevant depends on a temperature difference this data cannot carry, so
the assignment is an **upper bound**: it sends a crew with a radiometric camera to confirm
it, it does not certify a hazard.

The per-mode reasoning cites the IEA-PVPS failure fact sheets: IEA-PVPS T13-30:2025,
"Photovoltaic Failure Fact Sheets (PVFS), Review 2025", published with the Task 13 report
"Degradation and Failure Modes in New Photovoltaic Cell and Module Technologies" (public PDF
on iea-pvps.org, 83 pages). Page numbers below are PDF pages. Each fact sheet rates safety in
three categories (no effect; may cause fire, shock or physical danger after a follow-up or
second failure; can cause one directly, PDF p. 7); those ratings are glyphs, which the text
extraction used here did not carry, so the rows below cite the sheets' prose, not their
glyphs. Class definitions are the dataset's own (InfraredSolarModules README).

| D1 class | D1 definition | CoA | extent | recommended action | source |
|---|---|---:|---|---|---|
| No-Anomaly | nominal module | 1 | none | none | CoA 1 definition (Glavaš et al. 2022) |
| Diode | activated bypass diode, typically 1/3 of module | 3 | substring | inspect bypass diode and junction box; replace module if faulty | PVFS 1-15 (p. 51): a failed bypass diode lets cells reverse-bias into hot spots, burn marks and, at worst, fire, and can deform or burn the junction box; p. 52, example 1: a short-circuited bypass diode costs about a third of the module's power and shows in IRT. PVFS 1-14 (p. 48): junction-box interconnection failures are dangerous because arcing can start a fire. PVFS 1-19 (p. 58) names thermal differences that may indicate damaged bypass diodes or junction-box failures as direct safety risks; p. 59, example 3: one over-heated substring, short-circuited bypass diode, replace module |
| Diode-Multi | multiple activated bypass diodes, typically 2/3 of module | 3 | substring | inspect bypass diode and junction box; replace module if faulty | as Diode |
| Hot-Spot | hot spot on a thin-film module | 3 | cell | inspect on site with a radiometric camera; replace if confirmed | PVFS 1-19 (p. 58): local overheating can melt solder joints, damage encapsulant and backsheet, break glass, and compromise safety up to fire; modules with a direct safety risk are replaced. The sheet is not specific to thin film. The radiometric confirmation is the project's: the data carries no temperature |
| Hot-Spot-Multi | multiple hot spots on a thin-film module | 3 | cell | inspect on site with a radiometric camera; replace if confirmed | as Hot-Spot |
| Offline-Module | entire module is heated | 2 | module | check wiring and connectors | PVFS 1-19 (p. 59), example 1: one module warmer than the others, possibly open-circuited and not connected to the system, the module itself normally functional; remark: check wiring |
| Cell | hot spot with square geometry in a single cell | 2 | cell | inspect; replace if damage confirmed | Action: PVFS 1-19 (p. 59), examples 7 and 8: one cell, or part of one, warmer than the rest; a shaded or defective cell; visual inspection and cleaning, replace the module where a defect is found. **CoA: project's assignment** (see Consequences) |
| Cell-Multi | hot spots with square geometry in multiple cells | 2 | cell | inspect; replace if damage confirmed | as Cell; **CoA: project's assignment** |
| Cracking | cracking on the module surface | 3 | cell | inspect; replace the module if the glass is broken | PVFS 1-8 (p. 33), glass breakage: "Mechanical and electrical safety is thus compromised"; corrective action: "All damaged modules have to be replaced". Reading D1's "cracking on the module surface" as glass breakage is an inference from the dataset's definition, not something the dataset states |
| Soiling | dirt, dust or other debris on the module surface | 2 | external | clean | PVFS 1-20 (p. 61): an optical loss, reversible by cleaning; non-uniform soiling can cause hot spots that in extreme cases damage the module |
| Vegetation | panels blocked by vegetation | 2 | external | cut vegetation | PVFS 3-3 (p. 75): shading is not itself a safety issue, though hot spots from prolonged shading can lead to follow-up failures; vegetation responsible for shading should be cut |
| Shadowing | sunlight obstructed by vegetation, structures or adjacent rows | 2 | external | remove shade source | PVFS 3-3 (p. 75), as Vegetation; for fixed obstacles the sheet's actions are shading analysis and mitigation, so "remove shade source" is the project's wording |

**Extent** orders work within a CoA in the dispatch rule (module, then substring, then cell,
then external): a whole module before part of one, and a fault in the module before a cause
outside it. It is a property of the failure mode as the dataset defines it, not a measured
area.

## Consequences

- **No energy, power-loss or currency figure** is attached to any class, now or later. The
  `Triage` record has no field for one. The fact sheets quote loss ranges for some modes
  (about a third of a module for a short-circuited bypass diode); those are properties of
  the failure mode, not measurements of a finding, and the triage does not carry them.
- **CoA 3 is an upper bound by mode.** Every Diode, Diode-Multi, Hot-Spot, Hot-Spot-Multi
  and Cracking dispatch is CoA 3, whatever its real severity; confirming it needs a
  radiometric measurement on site (for Cracking, a visual inspection of the glass).
- **Two rows are the project's assignment, not the sources': Cell and Cell-Multi at CoA 2.**
  D1's Cell class is a single-cell hot spot, the failure mode PVFS 1-19 describes. That
  sheet says most modules with a single hot cell lose insignificant power (p. 58), but its
  escalation path from overheating to burn marks and fire applies to any hot spot, and
  PVFS 1-1 (p. 16) names a hot-spot and burn-mark risk from inactive cell parts. The line
  between Cell (CoA 2) and Hot-Spot (CoA 3) is drawn here, not in the sources: thin-film
  Hot-Spot is kept at the upper bound, crystalline Cell is not.
- **Amended 2026-10-23:** Cracking moved from CoA 2 to CoA 3 after the first triage run, on
  reading PVFS 1-8 against D1's class definition; the decision did not use the triage
  results. The first run is kept as superseded.
- **Classes, not findings, carry the CoA**, so the triage inherits the classifier's
  confusions. At field prevalence the classifier's per-class precision is low for several of
  the classes the CoA 3 rows rest on (docs/EXPERIMENTS.md, base-rate correction): the
  dispatch decision is better supported than the label that sets its CoA.
- `delta_dn_uncalibrated` (99th-percentile minus median DN of one crop) is filled on each
  finding but never used to assign a CoA or to rank: D1's crops come from mixed sensors, so
  comparing it across crops compares across frames.
