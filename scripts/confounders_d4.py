"""What D4 can and cannot say about the causes behind the per-inverter rankings.

    python scripts/confounders_d4.py

Every check here describes structure in the data. None names a cause: where a
pattern fits more than one cause, the output says which, not which is true.
Rules (counting threshold, producing) come from configs/d4_analysis.yaml; the
rankings themselves are recomputed with scripts/rank_d4.py's plant_tables, so
"bottom 3" and "top 3" are by the primary slope of that script. Per plant:

1. Zero-output structure. Stretches of AC_POWER == 0 over an inverter's
   counted intervals (solar_inspect.analytics.inverters.zero_runs): count,
   median and longest length, and the hour of day of every zero interval. Each
   zero interval is classed by how many inverters with a counted row at that
   timestamp are at zero too: plant-wide (more than half of them), single (this
   inverter only) or partial (in between). A plant-wide zero under sun points
   at something all inverters share (grid, export limit, plant control, the
   sensor itself); a single-inverter zero points at that inverter or its array.
2. Clipping. Share of each inverter's producing counted rows within 2% of its
   own maximum AC_POWER, and its slope fitted only below the plant's 90th
   percentile of counted irradiance against its slope on all rows. A flat top
   pulls a straight-line slope down; a ratio near 1 says the top is not flat.
   Scatter of AC_POWER against IRRADIATION for the bottom 3 and top 3.
3. Time of day. Median sister ratio per hour for the bottom 3, with zeros
   (the declared ratio) and over producing rows only. Fixed-geometry shading
   shows as a dip at the same hours every day.
4. Sensor. Timestamps with IRRADIATION > 0 at night, night being before 05:00
   or from 19:00 (the data's local time): when, and how large.
5. DC side. Per inverter, the median AC/DC over producing counted rows with
   DC_POWER > 0, and the Spearman agreement of a DC_POWER slope ranking with
   the AC slope ranking. If a low-AC inverter converts at the plant's usual
   AC/DC, its shortfall is already in the DC it receives. Plant 1's DC_POWER
   is about 10x too large (docs/DATA.md); a constant factor leaves its ratios
   comparable within the plant, not with Plant 2.
6. Persistence. Median sister ratio per 7-day block for the bottom 3.
7. Shared sensor. One pooled OLS per plant with an intercept and an
   IRRADIATION slope per inverter (its slopes equal the per-inverter fits),
   with Driscoll-Kraay standard errors (statsmodels `hac-groupsum`, scores
   summed per timestamp, lags by the declared rule on the rows of the fit).
   For each inverter, the contrast slope minus the mean slope of the other 21:
   its standard error from Driscoll-Kraay, which carries the covariance between
   inverters, against the one implied by treating the per-inverter Newey-West
   fits as independent.

Not measurable in D4, stated in the output: DC/AC ratio (no nameplate DC
capacity per inverter), shading geometry beyond its time-of-day signature (no
layout), sensor placement and plane-of-array irradiance (one sensor, position
and tilt unknown), and fault against season (34 days).

zero_runs, sister_ratios and OLS are solar_inspect.analytics; Driscoll-Kraay
and Spearman are statsmodels and scipy.

Writes runs/d4_conf_<utc>/ (gitignored): confounders.json and three figures,
and prints the docs/EXPERIMENTS.md row for the run.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402
import statsmodels.api as sm                                             # noqa: E402
import yaml                                                              # noqa: E402
from scipy import stats                                                  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from rank_d4 import plant_tables                                         # noqa: E402
from solar_inspect.analytics import data as d4                          # noqa: E402
from solar_inspect.analytics.covariance import newey_west_lags           # noqa: E402
from solar_inspect.analytics.inverters import (counted, sister_ratios,   # noqa: E402
                                               zero_runs)
from solar_inspect.analytics.ols import normal_equations                 # noqa: E402
from solar_inspect.classification.train import git_state                 # noqa: E402

NEAR_MAX = 0.98
UPPER_Q = 0.90
NIGHT = (5, 19)          # night: hour < 5 or hour >= 19
WEEK = pd.Timedelta(days=7)

NOT_MEASURABLE = {
    "dc_ac_ratio": "no nameplate DC capacity per inverter in D4",
    "shading_geometry": "no array layout; only its time-of-day signature is measured (3)",
    "sensor_placement": "one sensor per plant, position and tilt unknown; no plane-of-array irradiance",
    "fault_vs_season": "34 days; c-Si degradation of ~0.5%/yr is ~0.05% over the window",
}


def zero_structure(df: pd.DataFrame, irr_min: float) -> dict:
    c = counted(df, irr_min)
    runs = zero_runs(df, irr_min)
    zero = c[c[d4.TARGET] == 0]
    share_zero = (c[d4.TARGET] == 0).groupby(c["DATE_TIME"]).mean()
    k_zero = (c[d4.TARGET] == 0).groupby(c["DATE_TIME"]).sum()
    s, k = zero["DATE_TIME"].map(share_zero), zero["DATE_TIME"].map(k_zero)
    kind = np.where(s > 0.5, "plant_wide", np.where(k == 1, "single", "partial"))
    per_inv = runs.groupby("SOURCE_KEY")["intervals"].agg(["count", "median", "max", "sum"])
    per_inv.columns = ["stretches", "median_intervals", "longest_intervals", "zero_intervals"]
    return {
        "zero_intervals": int(len(zero)), "stretches": int(len(runs)),
        "stretch_intervals_median": float(runs["intervals"].median()) if len(runs) else 0.0,
        "stretch_intervals_max": int(runs["intervals"].max()) if len(runs) else 0,
        "stretches_of_one_interval": int((runs["intervals"] == 1).sum()),
        "zero_intervals_by_kind": {x: int((kind == x).sum()) for x in ("plant_wide", "partial", "single")},
        "inverters_at_zero_with_it": {q: float(k.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "days_with_plant_wide_zero": int(zero.loc[kind == "plant_wide", "DATE_TIME"].dt.normalize().nunique()),
        "median_irradiance_by_kind": {x: float(zero.loc[kind == x, "IRRADIATION"].median())
                                      for x in ("plant_wide", "partial", "single") if (kind == x).any()},
        "zero_intervals_by_hour": {int(h): int(n) for h, n in
                                   zero["DATE_TIME"].dt.hour.value_counts().sort_index().items()},
        "counted_intervals_by_hour": {int(h): int(n) for h, n in
                                      c["DATE_TIME"].dt.hour.value_counts().sort_index().items()},
        "per_inverter": json.loads(per_inv.to_json(orient="index")),
    }


def fit_slope(rows: pd.DataFrame, y: str = d4.TARGET) -> float:
    return float(normal_equations(rows[["IRRADIATION"]].to_numpy(float), rows[y].to_numpy(float))[1])


def clipping(prod: pd.DataFrame) -> pd.DataFrame:
    cut = prod["IRRADIATION"].quantile(UPPER_Q)
    out = {}
    for key, g in prod.groupby("SOURCE_KEY"):
        top = g[d4.TARGET].max()
        out[key] = {"ac_max": float(top),
                    "share_within_2pct_of_max": float((g[d4.TARGET] >= NEAR_MAX * top).mean()),
                    "slope_below_p90": fit_slope(g[g["IRRADIATION"] < cut]),
                    "slope_all": fit_slope(g)}
    t = pd.DataFrame(out).T
    t["slope_all_over_below_p90"] = t["slope_all"] / t["slope_below_p90"]
    return t


def by_hour(df: pd.DataFrame, irr_min: float, keys: list[str]) -> dict:
    c = counted(df, irr_min)
    r = sister_ratios(df, irr_min)
    hour = c["DATE_TIME"].dt.hour
    out = {"all_inverters": {int(h): float(v) for h, v in r.groupby(hour).median().items()}}
    for key in keys:
        m = c["SOURCE_KEY"] == key
        out[key] = {
            "with_zeros": {int(h): float(v) for h, v in r[m].groupby(hour[m]).median().items()},
            "producing_only": {int(h): float(v) for h, v in
                               r[m & (c[d4.TARGET] > 0)].groupby(hour[m]).median().items()}}
    return out


def by_week(df: pd.DataFrame, irr_min: float, keys: list[str]) -> dict:
    c = counted(df, irr_min)
    r = sister_ratios(df, irr_min)
    week = ((c["DATE_TIME"] - df["DATE_TIME"].min()) // WEEK).astype(int)
    return {key: {int(w): float(v) for w, v in r[c["SOURCE_KEY"] == key]
                  .groupby(week[c["SOURCE_KEY"] == key]).median().items()} for key in keys}


def night_readings(df: pd.DataFrame) -> dict:
    ts = df.groupby("DATE_TIME").agg(irr=("IRRADIATION", "first"), ac=(d4.TARGET, "sum")).reset_index()
    h = ts["DATE_TIME"].dt.hour
    n = ts[((h < NIGHT[0]) | (h >= NIGHT[1])) & (ts["irr"] > 0)]
    day = ts[(h >= NIGHT[0]) & (h < NIGHT[1]) & (ts["irr"] > 0)]
    return {"timestamps": int(len(n)), "nights": int(n["DATE_TIME"].dt.normalize().nunique()),
            "by_hour": {int(k): int(v) for k, v in n["DATE_TIME"].dt.hour.value_counts().sort_index().items()},
            "irradiance_median": float(n["irr"].median()) if len(n) else 0.0,
            "irradiance_max": float(n["irr"].max()) if len(n) else 0.0,
            "daylight_irradiance_median": float(day["irr"].median()),
            "with_any_ac_output": int((n["ac"] > 0).sum()),
            "first": str(n["DATE_TIME"].min()), "last": str(n["DATE_TIME"].max())}


def dc_side(prod: pd.DataFrame, t: pd.DataFrame) -> dict:
    p = prod[prod["DC_POWER"] > 0]
    ac_dc = (p[d4.TARGET] / p["DC_POWER"]).groupby(p["SOURCE_KEY"]).median()
    dc_slope = pd.Series({k: fit_slope(g, "DC_POWER") for k, g in p.groupby("SOURCE_KEY")})
    return {"ac_dc_median_plant": float(ac_dc.median()),
            "ac_dc_min": float(ac_dc.min()), "ac_dc_max": float(ac_dc.max()),
            "spearman_dc_slope_vs_ac_slope": float(stats.spearmanr(dc_slope[t.index], t["perf_slope"])[0]),
            "per_inverter": {k: {"ac_dc_median": float(ac_dc[k]), "dc_slope": float(dc_slope[k])}
                             for k in t.index}}


def shared_sensor(prod: pd.DataFrame, t: pd.DataFrame) -> dict:
    prod = prod.sort_values(["SOURCE_KEY", "DATE_TIME"])
    keys = list(t.index)
    D = (prod["SOURCE_KEY"].to_numpy()[:, None] == np.array(keys)[None, :]).astype(float)
    A = np.hstack([D, D * prod["IRRADIATION"].to_numpy(float)[:, None]])
    y = prod[d4.TARGET].to_numpy(float)
    time = np.unique(prod["DATE_TIME"].to_numpy(), return_inverse=True)[1]
    lags = newey_west_lags(len(y))
    res = sm.OLS(y, A).fit(cov_type="hac-groupsum",
                           cov_kwds={"time": time, "maxlags": lags, "use_correction": False})
    k = len(keys)
    slopes, V = res.params[k:], res.cov_params()[k:, k:]
    se_nw = t["perf_se_hac"].to_numpy(float)
    out = {}
    for i, key in enumerate(keys):
        c = np.full(k, -1.0 / (k - 1))
        c[i] = 1.0
        diff = float(c @ slopes)
        se_dk = float(np.sqrt(c @ V @ c))
        se_ind = float(np.sqrt((c ** 2) @ (se_nw ** 2)))
        out[key] = {"slope_minus_mean_of_others": diff, "se_driscoll_kraay": se_dk,
                    "se_independent_newey_west": se_ind, "z_driscoll_kraay": diff / se_dk,
                    "z_independent": diff / se_ind}
    return {"rows": len(y), "hac_lags": lags,
            "max_abs_slope_diff_vs_per_inverter": float(np.abs(slopes - t["perf_slope"].to_numpy()).max()),
            "contrasts": out}


def main() -> int:
    sha, dirty = git_state()
    cfg = yaml.safe_load((REPO / "configs" / "d4_analysis.yaml").read_text(encoding="utf-8"))
    irr_min = cfg["counting_irradiance_min"]
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = REPO / "runs" / f"d4_conf_{run}"
    dest.mkdir(parents=True)
    out = {"run": run, "git_sha": sha, "git_dirty": dirty, "not_measurable_in_d4": NOT_MEASURABLE,
           "plants": {}}
    fig_sc, ax_sc = plt.subplots(2, 6, figsize=(18, 6), sharey="row")
    fig_hr, ax_hr = plt.subplots(1, 2, figsize=(11, 4.2))
    fig_zero, ax_zero = plt.subplots(1, 2, figsize=(11, 4.2))

    for row, plant in enumerate((1, 2)):
        df = d4.load(plant)
        t, _ = plant_tables(df, cfg)               # sorted by primary slope, ascending
        t = t.sort_index()
        order = t.sort_values("perf_slope").index
        bottom, top = list(order[:3]), list(order[-3:])
        c = counted(df, irr_min)
        prod = c[c[d4.TARGET] > 0]

        clip = clipping(prod)
        sh = shared_sensor(prod, t)
        res = {
            "bottom3": bottom, "top3": top,
            "zero_output": zero_structure(df, irr_min),
            "clipping": {"share_within_2pct_median": float(clip["share_within_2pct_of_max"].median()),
                         "share_within_2pct_max": float(clip["share_within_2pct_of_max"].max()),
                         "slope_all_over_below_p90_median": float(clip["slope_all_over_below_p90"].median()),
                         "irradiance_p90": float(prod["IRRADIATION"].quantile(UPPER_Q)),
                         "per_inverter": json.loads(clip.to_json(orient="index"))},
            "sister_ratio_by_hour": by_hour(df, irr_min, bottom),
            "sister_ratio_by_week": by_week(df, irr_min, bottom),
            "night_irradiance": night_readings(df),
            "dc_side": dc_side(prod, t),
            "shared_sensor": sh,
        }
        out["plants"][plant] = res

        for col, key in enumerate(bottom + top):
            g = prod[prod["SOURCE_KEY"] == key]
            ax = ax_sc[row, col]
            ax.scatter(g["IRRADIATION"], g[d4.TARGET], s=2, alpha=0.3, rasterized=True)
            ax.axhline(clip.loc[key, "ac_max"] * NEAR_MAX, color="r", lw=0.6)
            ax.set(title=f"P{plant} {'bottom' if key in bottom else 'top'} {key[:4]}",
                   xlabel="IRRADIATION (kW/m²)")
        ax_sc[row, 0].set_ylabel("AC_POWER")

        hr = res["sister_ratio_by_hour"]
        ax = ax_hr[row]
        ax.plot(list(hr["all_inverters"]), list(hr["all_inverters"].values()), "k--", label="all inverters")
        for key in bottom:
            ax.plot(list(hr[key]["with_zeros"]), list(hr[key]["with_zeros"].values()), "o-", label=key[:4])
        ax.set(title=f"Plant {plant}: median sister ratio by hour", xlabel="hour", ylabel="sister ratio")
        ax.legend(fontsize=8)

        z = res["zero_output"]
        hours = sorted(z["counted_intervals_by_hour"])
        ax = ax_zero[row]
        ax.bar(hours, [z["zero_intervals_by_hour"].get(h, 0) / z["counted_intervals_by_hour"][h]
                       for h in hours])
        ax.set(title=f"Plant {plant}: share of counted intervals at zero, by hour",
               xlabel="hour", ylabel="share")

        print(f"\nplant {plant}: bottom3 {bottom}  top3 {top}")
        print(f"  zero output: {z['zero_intervals']} intervals in {z['stretches']} stretches "
              f"(median {z['stretch_intervals_median']}, longest {z['stretch_intervals_max']} intervals); "
              f"by kind {z['zero_intervals_by_kind']}; plant-wide on {z['days_with_plant_wide_zero']} days")
        print(f"  clipping: share within 2% of max, median {res['clipping']['share_within_2pct_median']:.4f}"
              f" (max {res['clipping']['share_within_2pct_max']:.4f}); slope all / below p90, median "
              f"{res['clipping']['slope_all_over_below_p90_median']:.3f}")
        print(f"  night IRRADIATION > 0: {res['night_irradiance']}")
        print(f"  AC/DC median {res['dc_side']['ac_dc_median_plant']:.4f} "
              f"({res['dc_side']['ac_dc_min']:.4f} to {res['dc_side']['ac_dc_max']:.4f}); "
              f"Spearman DC slope vs AC slope {res['dc_side']['spearman_dc_slope_vs_ac_slope']:.3f}")
        print(f"  pooled fit: {sh['rows']:,} rows, L={sh['hac_lags']}, max |slope - per-inverter| "
              f"{sh['max_abs_slope_diff_vs_per_inverter']:.2e}")
        for key in bottom + top:
            ct = sh["contrasts"][key]
            print(f"    {key}: slope - mean(others) {ct['slope_minus_mean_of_others']:8.1f}  "
                  f"z DK {ct['z_driscoll_kraay']:6.2f}  z indep {ct['z_independent']:6.2f}  "
                  f"SE DK/indep {ct['se_driscoll_kraay'] / ct['se_independent_newey_west']:.3f}")

    for name, fig in (("clipping_scatter", fig_sc), ("sister_ratio_by_hour", fig_hr),
                      ("zero_share_by_hour", fig_zero)):
        fig.tight_layout()
        fig.savefig(dest / f"{name}.png", dpi=110)
    (dest / "confounders.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8",
                                           newline="\n")

    def se_ratio(p):
        cs = out["plants"][p]["shared_sensor"]["contrasts"].values()
        return float(np.median([c["se_driscoll_kraay"] / c["se_independent_newey_west"] for c in cs]))
    p1, p2 = out["plants"][1], out["plants"][2]
    kinds = p2["zero_output"]["zero_intervals_by_kind"]
    print(f"\nwrote {dest.relative_to(REPO).as_posix()}/\n\nEXPERIMENTS row:")
    print(f"| {run} | `{sha[:7]}` | {'yes' if dirty else 'no'} | `scripts/confounders_d4.py` "
          f"| {p1['zero_output']['zero_intervals']:,} ({p1['zero_output']['stretches']:,}) / "
          f"{p2['zero_output']['zero_intervals']:,} ({p2['zero_output']['stretches']:,}) "
          f"| {kinds['plant_wide']:,} / {kinds['partial']:,} / {kinds['single']:,} "
          f"| {p1['clipping']['share_within_2pct_median']:.4f} / {p2['clipping']['share_within_2pct_median']:.4f} "
          f"| {p1['clipping']['slope_all_over_below_p90_median']:.3f} / "
          f"{p2['clipping']['slope_all_over_below_p90_median']:.3f} "
          f"| {p1['night_irradiance']['timestamps']} / {p2['night_irradiance']['timestamps']} "
          f"| {p1['dc_side']['spearman_dc_slope_vs_ac_slope']:.3f} / "
          f"{p2['dc_side']['spearman_dc_slope_vs_ac_slope']:.3f} "
          f"| {se_ratio(1):.3f} / {se_ratio(2):.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
