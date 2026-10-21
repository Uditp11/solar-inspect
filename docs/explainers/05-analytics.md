# Explainer 05 — Analytics: regression and per-inverter analysis

Module 5 of the pipeline, regression half. Plant-level regression of inverter AC
output on weather, its diagnostics, and per-inverter availability and performance
rankings on D4, real generation data from two plants over 34 days. Triage and the
SQL layer are the other half and are not built yet.

Every number here traces to a run logged in `docs/EXPERIMENTS.md` (D4 sections),
with the commit that produced it. Nothing in this module is held out or tested
once: it is in-sample analysis of one fixed 34-day window, not prediction.

---

## What it is

D4 is two plants, 22 string inverters each, one weather sensor each, one row per
inverter per 15 minutes from 2020-05-15 to 2020-06-17. Per row: AC and DC power
and two cumulative yield counters per inverter; per timestamp: irradiation,
module temperature and ambient temperature from the plant's sensor. The sensor
temperatures are real measurements, unlike anything in the imagery modules.

The module does three things.

1. **Cleans and merges** the four CSVs, and reports what was wrong with them.
2. **Fits AC_POWER on the three weather columns per plant**, with two solvers
   written here, and diagnoses that fit: which least-squares assumptions the data
   breaks, and what that does to the standard errors.
3. **Ranks the 22 inverters of each plant** on two separate failure modes,
   availability (does it produce at all when the sun is up) and performance (how
   much it produces per unit of irradiance when it does), with intervals, a
   regression-free cross-check, and a measured look at what D4 can and cannot say
   about the causes.

## Why this stage exists in a PV inspection pipeline

Imagery finds defects on modules. It does not say which of them cost energy, and
a plant's monitoring data does the opposite: it shows which inverter is producing
less, not why. An inspection is most useful when the two meet: an underperforming
inverter tells the inspection where to look, and a thermal finding on a string is
worth more when its inverter's output shows the loss.

The analytics stage is the monitoring side of that. Its job is to turn 15-minute
generation data into a short, defensible list: which inverters are unavailable more
often than their sisters, which produce less per unit of sun, how sure that is, and
what the data cannot rule out. The last part matters most, because a ranking with
no stated confounders sends a crew to an inverter that is merely clipping, shaded
at 4 pm, or smaller than its neighbours.

## How it works

### Cleaning comes first, and three findings came out of it

**The date-format trap.** Only one of the four files, Plant 1's generation CSV,
writes dates day-first (`15-05-2020 00:00`); the other three are ISO. A parser
left to infer the format reads 05-06-2020 as 6 May, and for the first 12 days of
each month nothing fails to say so. Every CSV is read with an explicit `format=`,
and a row that does not match raises instead of being guessed.

**Plant 1's DC_POWER is about 10× too large.** Median AC/DC over producing rows is
**9.78%** in Plant 1 against **97.84%** in Plant 2, where a string inverter should
be. `dc_scale_check()` detects it from the data and reports it; nothing rescales it.
DC_POWER is never a model input (AC is DC times an efficiency, so regressing one on
the other gives R² ≈ 1 for free), and a silent fix would hide the defect from every
later reader. The cumulative yield columns are never inputs either.

**Night rows inflate R².** About half of D4 is night with IRRADIATION = 0 and
AC_POWER = 0. Left in, they hand the model "predict zero in the dark" for free. The
fit is on `IRRADIATION > 0` only, and the same model fitted with night included
measures the inflation:

| | Plant 1 | Plant 2 |
|---|---:|---:|
| R², daylight (the fit), n | 0.9602, 38,376 | 0.4354, 38,722 |
| R², night included, n | 0.9790, 68,774 | 0.6136, 67,698 |
| inflation from night rows | **+0.019** | **+0.178** |

The weaker the daylight fit, the more the night rows flatter it.

### The normal equations, and why XᵀX is never inverted

With A the design matrix (a column of ones, then IRRADIATION, MODULE_TEMPERATURE,
AMBIENT_TEMPERATURE) and y the AC power, least squares minimises

    L(β) = ‖y − Aβ‖² = yᵀy − 2βᵀAᵀy + βᵀAᵀAβ.

Its gradient is ∇L = −2Aᵀy + 2AᵀAβ. Setting it to zero gives the normal equations

    AᵀA β = Aᵀy,

and because AᵀA is symmetric positive semi-definite, the stationary point is a
minimum; with A of full column rank it is unique.

The solve uses a Cholesky factorisation of AᵀA (scipy's `cho_factor` /
`cho_solve`), never `inv(AᵀA)`. The reason is conditioning, not cost: at p = 3 the
O(np² + p³) cost is nothing. Forming AᵀA squares the condition number,
cond(AᵀA) = cond(A)², and D4's irradiation and module temperature are correlated at
0.962 (Plant 1) and 0.947 (Plant 2). Measured on the Plant 1 daylight fit:
**cond(A) = 633, cond(AᵀA) = 4.0 × 10⁵** (Plant 2: 641 and 4.1 × 10⁵). Each factor
of ten in the condition number costs about one significant digit of the solution, so
forming AᵀA spends about five and a half of double precision's sixteen before the
solve starts. Here that still leaves enough: the normal-equations coefficients agree
with `np.linalg.lstsq` (which works on A directly, through LAPACK) to **2.6 × 10⁻¹⁰**
at worst. With stronger collinearity it would not, which is why a QR or SVD solve on A
is the default in general and Cholesky on AᵀA is a choice that has to be checked.

**Gradient descent is the second solver.** Batch gradient descent on the mean
squared loss, features standardised, step 1/L with L the largest eigenvalue of the
Hessian ZᵀZ/n, stopping when the gradient norm has fallen by a factor of 10¹⁰.
Coefficients are mapped back to original units. It converged in 1,747 steps (Plant 1)
and 2,346 (Plant 2) and agrees with `lstsq` to **5.9 × 10⁻⁶**. Standardising removes
the difference in scale between columns, not their correlation; the residual
correlation is what keeps the step count in the thousands. Both solvers are checked
against `np.linalg.lstsq` and scikit-learn's `LinearRegression` on every run.

### Gauss-Markov, and the assumption this data breaks

Gauss-Markov says OLS is the best linear unbiased estimator if the model is linear
in its parameters, A has full column rank, the errors have mean zero given A, and
the errors are homoscedastic and uncorrelated. The last pair is what this data
breaks, and the one that matters most is **autocorrelation**: one inverter's
residual at 11:00 says a lot about its residual at 11:15.

Durbin-Watson is about 2 for uncorrelated residuals and falls towards 0 with
positive first-order autocorrelation. One trap had to be handled first: D4 stores
rows timestamp by timestamp, so in file order the next row is the next inverter at
the same moment, not the same inverter 15 minutes later. Every time-lag statistic
is therefore computed within each inverter's own series. Measured on the plant-level
daylight fit, within inverter: **DW = 1.010 (Plant 1) and 0.253 (Plant 2)**; mean
lag-1 residual autocorrelation 0.385 and 0.853. Breusch-Pagan rejects
homoscedasticity in both plants (p = 7.9 × 10⁻¹²⁰ and below floating-point range).

A lag counts rows of an inverter's daylight series. Night is removed and not filled,
so one day is about 51 rows, not 96, and the residual ACF has a bump near lag 51:
the same time of day on the next day.

What autocorrelation does: the coefficients stay unbiased (it does not touch the
mean-zero condition), but the classical standard error σ²(AᵀA)⁻¹ assumes every row
is new information, and correlated rows are not. The standard errors come out too
small, so intervals are too narrow and tests too confident. Two corrections were
measured against the classical one, at L = 15 lags:

- **Newey-West** (written here): the sandwich (AᵀA)⁻¹ S (AᵀA)⁻¹ with
  S = Γ₀ + Σₗ wₗ(Γₗ + Γₗᵀ), Bartlett weights wₗ = 1 − l/(L+1), lag products taken only
  within one inverter's series. L follows the Newey-West rule of thumb
  floor(4 (n/100)^(2/9)), n being the rows of the fit.
- **Driscoll-Kraay** (statsmodels `hac-groupsum`): scores summed over inverters at
  each timestamp, then Newey-West over time. All 22 inverters of a plant share one
  weather sensor, so their residuals at one timestamp share that sensor's error;
  Driscoll-Kraay allows for that, within-inverter Newey-West does not.

| standard error / classical, range over the 4 coefficients | Plant 1 | Plant 2 |
|---|---:|---:|
| Newey-West, within inverter | 1.31 to 1.90 | 1.62 to 3.09 |
| Driscoll-Kraay | 3.53 to 5.33 | 3.50 to 5.17 |

Classical standard errors are too small by up to a factor of three once
within-inverter autocorrelation is allowed for, and by up to five once the shared
sensor is. With 22 inverters under one sensor, about 38,000 rows carry roughly the
information of the ~1,800 timestamps behind them.

R² and adjusted R² differ by 3 × 10⁻⁶ and 4 × 10⁻⁵: at p = 3 and n ≈ 38,000 the
adjustment says nothing, and it is not presented as a finding.

### VIF, and why the temperature slopes are not read

The variance inflation factor of column j is 1/(1 − R²ⱼ), R²ⱼ from regressing that
column on the others. Measured: IRRADIATION 13.9 and 15.8, MODULE_TEMPERATURE
**23.1 and 30.5**, AMBIENT_TEMPERATURE 4.4 and 6.7 (Plant 1, Plant 2). Above about
10, a coefficient's variance is dominated by its overlap with the others rather than
by its own effect. Plant 2's fit shows what that looks like: module temperature gets
−9.3 and ambient temperature +17.0, signs that physics would not choose, because the
model is trading one correlated column against another. The individual temperature
slopes are therefore not interpreted. The irradiation slope is what the
per-inverter analysis uses, and even it moves when module temperature is added
(below).

### Why a pooled regression cannot rank inverters

One weather sensor per plant means all 22 inverters share an identical row of A at
each timestamp. A pooled OLS residual averaged per inverter then reduces to that
inverter's mean power minus the plant mean: a `groupby`, with no regression in it.
The ranking has to come from per-inverter fits, or from a pooled model with
per-inverter parameters (used below as a cross-check).

### Availability and performance are two failure modes

An inverter can be down (producing nothing when it should) or weak (producing, but
less than its sisters). Folding the two into one fit hides both: Plant 2's low
plant-level R² is mostly zero-output intervals under sun. A post-hoc split, made
after the rankings and labelled as such, refits the plant model on four row sets:

| R² (n) | daylight, zeros kept (the declared fit) | daylight, producing only | ≥ 0.1 kW/m², zeros kept | ≥ 0.1 kW/m², producing |
|---|---:|---:|---:|---:|
| Plant 1 | 0.9602 (38,376) | 0.9695 (36,823) | 0.9316 (30,718) | 0.9520 (30,655) |
| Plant 2 | 0.4354 (38,722) | **0.8854** (32,022) | 0.1999 (28,818) | 0.8118 (25,044) |

Dropping only the zero-output rows takes Plant 2 from 0.435 to 0.885. It does not
replace the declared fit; it says where that fit's unexplained variance is.

The rules for both measures were committed in `configs/d4_analysis.yaml` before any
per-inverter number existed:

- **Counted interval**: a generation row with IRRADIATION ≥ 0.1 kW/m², so dawn and
  dusk, when an inverter is legitimately not yet producing, are not outages.
- **Producing**: AC_POWER > 0.
- **Availability** = producing / counted intervals, per inverter. A timestamp with
  no row for an inverter is a data gap, not an outage; gaps are reported beside it.
- **Performance**: per inverter, OLS of AC_POWER on IRRADIATION over its producing
  counted intervals, ranked on the slope, with a 95% interval from Newey-West
  standard errors (L by the same rule on that inverter's n, 6 or 7). An inverter
  whose interval contains the plant's median slope is marked indistinguishable
  from it rather than ranked as if ordered.
- **Robustness model**: the same with MODULE_TEMPERATURE added, compared by Spearman
  rank correlation, never chosen after seeing both.
- **Sister ratio**: no regression. At each counted timestamp, an inverter's AC_POWER
  over the median AC_POWER of the plant's producing inverters; per inverter, the
  median of those ratios. Twenty-two inverters under the same sky are compared with
  each other, so the weather cancels. It answers the question "why regress on
  irradiance at all?", and its agreement with the slope is the check on the
  regression.

### What is written here, and what is a library

| Component | Written here | Library | Cross-checked against |
|---|---|---|---|
| OLS, normal equations | ✓ | linear solve: `scipy.linalg.cho_solve` | `np.linalg.lstsq`, scikit-learn |
| OLS, gradient descent | ✓ | none | `np.linalg.lstsq`, scikit-learn |
| Durbin-Watson, VIF | ✓ | none | hand-computed, `tests/test_diagnostics.py` |
| Classical and Newey-West covariance | ✓ | `scipy.linalg.cho_solve` for (AᵀA)⁻¹ | hand-computed; statsmodels `HAC` and `hac-panel`, to 1e-10 |
| Driscoll-Kraay covariance, ACF, Breusch-Pagan | | statsmodels | |
| Availability, sister ratio, zero-output stretches | ✓ | pandas grouping | hand-computed toy frames, `tests/test_inverters.py` |
| Spearman rank correlation | | `scipy.stats.spearmanr` | |

The statsmodels HAC cross-check is exact because the definition is the same:
statsmodels' default `use_correction=False` applies no small-sample factor, and its
Bartlett weights are 1 − l/(L+1). The tests pass that setting explicitly.

## Results

Plant 1 / Plant 2, 22 inverters each. Ranking run `20261021T180327Z`; plant-level
numbers above.

| | Plant 1 | Plant 2 |
|---|---:|---:|
| producing counted rows | 30,655 | 25,044 |
| availability, median (min) | 1.0000 (0.9857) | **0.8714 (0.7716)** |
| median slope, AC_POWER per kW/m² | 1232.10 | 1169.13 |
| slope intervals containing the median | 13 / 22 | 13 / 22 |
| flagged at 95% / surviving Bonferroni | 9 / 4 | 9 / 5 |
| Spearman, slope vs robustness model | 0.805 | 0.587 |
| Spearman, slope vs sister ratio | 0.740 | 0.907 |
| bottom-3 overlap, slope vs sister ratio | 2 / 3 | 2 / 3 |
| Spearman, availability at 0.1 vs 0.05 / 0.2 | 1.000 / 1.000 | 1.000 / 0.997 |
| Spearman, availability vs common window | 1.000 | 0.914 |

**Availability.** Plant 1's inverters are essentially always on: the worst two,
`bvBOhCH3iADSZry` and `1BY6WEcLGh8j5v7`, are at 0.986. Plant 2's median inverter
is not producing in 13% of counted intervals, the worst at 0.772. Changing the
counting threshold to 0.05 or 0.2 kW/m² barely moves the ordering.

**Common window.** Four Plant 2 inverters (`IQ2d7wF4YD8zU1Q`, `NgDl19wMapZy17u`,
`mqwcsP2rE7J0TFp`, `xMbIugepa2P7lBB`) have no rows from after 2020-05-20 21:45
until 2020-05-29 16:15, a hole in the middle of the period, so under the declared
rule their availability covers fewer intervals than the other eighteen's. As a
robustness check, availability was recomputed only at the 2,355 of 3,259 Plant 2
timestamps where every inverter has a row (Plant 1: 3,057 of 3,157). Spearman with
the declared ranking is 0.914 in Plant 2 (1.000 in Plant 1). The bottom three are the
same three inverters, reordered; the four inverters with the hole keep the top four
places with unchanged values (0.964 to 0.977), so their high availability is not an
artefact of their shorter record. The declared rule stays primary.

**Performance.** In both plants the two lowest slopes are clear of the rest under
every check: Plant 1 `bvBOhCH3iADSZry` (1136.1, interval 1116.2 to 1156.1, sister
ratio 0.922) and `1BY6WEcLGh8j5v7` (1146.2, sister ratio 0.933); Plant 2
`Quc1TzYxW2pYoWX` (879.6, sister ratio 0.799) and `Et9kgGMDl729KT4` (1016.4, 0.895).
Each is the same under the robustness model and the sister ratio, survives
Bonferroni, and survives the shared-sensor cross-check below. The third-lowest in
each plant (`WRmjgnKYAwPKWDb`, `LYwnQax7tkwH5Cb`) is flagged at 95% but not under
Bonferroni, and the sister ratio places Plant 1's elsewhere (8th): it is not a
finding.

**The two failure modes go together in Plant 2, not in Plant 1.** Plant 2's three
lowest slopes are its three lowest availabilities. Plant 1's two least available
inverters are also its two lowest slopes, but its availability spread is so small
(0.986 to 1.000) that this is weak.

**The robustness model moves Plant 2's middle ranks a lot** (Spearman 0.587):
adding module temperature lowers the slopes by a median of 99 and moves single inverters
by up to 14 places. That is the VIF of 30 at work: irradiance and module temperature
cannot be separated well enough for the individual slope to be stable. The ends of
the ranking hold, the middle does not.

### Confounders, measured where D4 allows

Run `20261021T180557Z`. Each check describes structure; none of them names a cause.

**Zero output (Plant 2).** 3,774 counted intervals at zero, in 531 stretches
(median 4 intervals, longest 34, i.e. 8.5 hours). They sit at midday and high
irradiance: 0.6% of counted intervals at 07:00 to 08:59 are at zero, 22% at 10:00,
30% at 11:00 to 12:59, 26% at 13:00, 15% at 14:00, at most 2.4% from 15:00 on. Classed by how
many inverters are at zero at the same moment: 423 plant-wide (more than half),
3,264 partial, 87 single-inverter; typically 8 of the 22 are at zero together.
Partial-simultaneous zeros dominate. That is consistent with something acting on a
group of inverters when output is highest (an export or grid limit, curtailment,
heat-related trips, a shared feeder), and not separable between those in D4. Single
inverter zeros, which would point at one machine, are rare. Plant 1 has 63 zero
intervals in all.

**Clipping.** An inverter that hits its AC limit shows a flat top: many rows at the
same maximum, however much brighter it gets. The share of producing counted rows
within 2% of each inverter's own maximum has a median of 0.0007 (Plant 1) and 0.0016
(Plant 2), and the scatter of AC against irradiance shows no plateau for the bottom 3
or top 3 of either plant. The curve does bend in Plant 1: the slope fitted on all
rows is 0.953 to 0.964 of the slope fitted below the 90th percentile of irradiance,
for low and high inverters alike (Plant 2: median 0.994). A curve that flattens at the
top pulls a straight-line slope down; here it does so by about 4% for every Plant 1
inverter, which shifts levels more than ranks. Consistent with a high-irradiance loss
(module temperature rises with irradiance) or partial limiting, not separable here.

**Time of day.** Fixed-geometry shading shows as a dip in the sister ratio at the
same hours every day. Producing rows only, Plant 1's two lowest inverters sit at 0.91
to 0.95 at every hour from 06:00 to 17:00, a near-constant shortfall with a slight
afternoon decline; Plant 2's three lowest are flat within 0.78 to 0.84, 0.89 to 0.94
and 0.93 to 0.99. The one time-of-day signature is Plant 1 `WRmjgnKYAwPKWDb`: 1.055
at 06:00 falling steadily to 0.962 at 17:00, above its sisters in the morning and
below them in the afternoon. Consistent with afternoon shading or a different
orientation, not separable from either. With zeros included, Plant 2's bottom three
drop to near 0 around 10:00 to 13:00: that is the zero-output block, not a shading
shape.

**The DC side.** Median AC/DC per inverter is the same for every inverter of a plant
(Plant 1 0.0979 for all 22, the 10× DC defect being a constant factor; Plant 2 0.979
to 0.980), and a ranking on the DC slope agrees with the AC slope ranking at Spearman
0.999 and 1.000. The low-AC inverters already receive less DC: their shortfall is
upstream of conversion, on the array side (modules or strings not contributing,
soiling, shading, mismatch, or simply a smaller array), not in the inverter's
efficiency.

**Persistence.** Per 7-day block, Plant 1's two lowest inverters stay at 0.918 to
0.928 and 0.927 to 0.942 from the first week to the fifth: the shortfall predates the
window. Plant 2's lowest moves from 0.780 to 0.831.

**The sensor.** Plant 2's irradiance sensor reads above zero at 64 night timestamps
(before 05:00 or from 19:00) over 20 nights, median 1.0 × 10⁻⁴ kW/m², at most
0.0029 kW/m² against a daylight median of 0.387, with no AC output at any of them.
Plant 1 has 5, all between 19:00 and 19:45. A small positive offset, far below the
counting threshold: it splits night breaks in the daylight series, but it does not
reach availability or the slopes. Where the sensor sits and how it is tilted is not in
the data.

**Multiple comparisons.** With 22 intervals at 95% per plant, about 22 × 0.05 = 1.1
are expected to miss the median by chance alone, so one inverter outside its interval
is not yet evidence. A Bonferroni interval (z = 3.05 for 0.05/22, two-sided) is
reported as an extra column: of the 9 flagged in each plant, 4 survive in Plant 1
(the two lowest and two of the highest) and 5 in Plant 2 (the two lowest and three of
the highest).

**The shared sensor, and what it does to a difference.** Per-inverter intervals treat
each inverter's fit as independent, but all 22 share one sensor's error, which cancels
in a difference between two inverters. A pooled fit with an intercept and an
irradiance slope per inverter (its slopes equal the per-inverter ones to 10⁻¹⁰) and
Driscoll-Kraay errors gives the standard error of "this inverter's slope minus the mean
of the other 21" with the covariance between inverters included. Against the same
contrast computed as if the fits were independent, its standard error is smaller: median
ratio **0.535** (Plant 1) and **0.717** (Plant 2). Per-inverter intervals overstate the
uncertainty of a difference. The two lowest inverters of each plant stay clear under
both (z of −17.1 and −10.7 in Plant 1, −12.3 and −4.5 in Plant 2); the third-lowest is
at −3.6 and −2.7, borderline.

## Limits

- **No performance ratio.** In IEC 61724 terms the performance ratio is final yield
  (kWh per kWp of installed DC capacity) over reference yield (plane-of-array
  irradiation over 1 kW/m²). D4 has neither input: no nameplate DC capacity per
  inverter, and one irradiance sensor of unknown orientation instead of
  plane-of-array irradiance. The slope here is AC kW per kW/m² of that sensor's
  reading, so an inverter with a smaller array has a lower slope while being
  perfectly healthy. Specific yield (kWh/kWp) cannot be computed for the same reason,
  and the DC/AC ratio, which decides how often an inverter clips, is not measurable.
- **Availability is time-based, not energy-based.** It counts intervals, not the
  energy that would have been produced in them, and it uses this project's threshold
  of 0.1 kW/m², not a contractual definition.
- **34 days cannot separate a fault from a season.** Crystalline-silicon modules
  degrade at around 0.5% a year, about 0.05% over this window, far below anything
  measured here. A persistent shortfall over five weeks says the cause predates the
  window, not what it is.
- **Causes are not identified.** Every confounder above is either measured as a
  pattern or stated as not measurable. The data narrows each finding to "on the
  array side" or "acting on a group of inverters at midday"; it does not name it.
- **Shading geometry, sensor placement and the plant layout are not in D4.** Only the
  time-of-day signature of shading is measured.
- **In-sample only.** Nothing is held out. The R² values describe these 34 days.
- **Two plants.** Every plant-level conclusion rests on two units, which differ in
  almost everything measured here.

With per-string current, a plane-of-array sensor, nameplate capacities and a longer
record, the same analysis becomes actionable: per-string sister ratios localise the
shortfall within an inverter, and a year of data separates a fault from a season.

## Triage and SQL

Comes with the triage stage.
