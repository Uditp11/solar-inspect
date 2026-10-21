"""Per-inverter availability, performance and sister-ratio rankings on D4.

    python scripts/rank_d4.py

Every rule comes from configs/d4_analysis.yaml, committed before any number
below existed. Per plant:

- availability per inverter at the declared counting threshold, at the two
  sensitivity thresholds, and on the common window, the timestamps at which
  every inverter of the plant has a row (all three reported, never adopted);
- performance: per inverter, OLS of AC_POWER on IRRADIATION on its producing
  counted intervals, ranked on the slope, with a 95% interval from Newey-West
  standard errors (normal critical value, lags by the declared rule on that
  inverter's n). An inverter whose interval contains the plant's median slope
  is marked indistinguishable from it rather than ranked as if ordered. With 22
  intervals at 95%, about one misses the median by chance alone, so a
  Bonferroni interval (z for 0.05 / 22, two-sided) is reported beside it as an
  extra column, never in its place;
- the robustness model (MODULE_TEMPERATURE added) and the Spearman agreement
  of its slope ranking with the primary one;
- the sister ratio, its Spearman agreement with the slope ranking, and the
  overlap of the bottom three in each.

One post-hoc check, labelled as such everywhere: the plant-level three-feature
model of scripts/ols_d4.py refitted on producing counted intervals only. It
asks whether the zero-output intervals explain a low plant-level R²; it does
not replace that fit.

OLS, Newey-West, availability and the sister ratio are solar_inspect.analytics;
Spearman is scipy.

Writes runs/d4_rank_<utc>/ (gitignored): rank.json and one CSV per plant, and
prints the docs/EXPERIMENTS.md row for the run.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.analytics import data as d4                          # noqa: E402
from solar_inspect.analytics.covariance import hac_cov, newey_west_lags  # noqa: E402
from solar_inspect.analytics.diagnostics import durbin_watson           # noqa: E402
from solar_inspect.analytics.inverters import (availability, counted,   # noqa: E402
                                               sister_ratio)
from solar_inspect.analytics.ols import add_intercept, normal_equations, r2  # noqa: E402
from solar_inspect.classification.train import git_state                 # noqa: E402

CONFIG = REPO / "configs" / "d4_analysis.yaml"
Z = float(stats.norm.ppf(0.975))


def slope(rows: pd.DataFrame, features: list[str]) -> dict:
    """OLS of AC_POWER on `features`; the IRRADIATION slope with its Newey-West SE."""
    X = rows[features].to_numpy(dtype=float)
    y = rows[d4.TARGET].to_numpy(dtype=float)
    A = add_intercept(X)
    beta = normal_equations(X, y)
    u = y - A @ beta
    lags = newey_west_lags(len(y))
    j = 1 + features.index("IRRADIATION")
    se = float(np.sqrt(hac_cov(A, u, lags)[j, j]))
    return {"n": len(y), "slope": float(beta[j]), "se_hac": se, "hac_lags": lags,
            "ci_low": float(beta[j]) - Z * se, "ci_high": float(beta[j]) + Z * se,
            "r2": r2(y, A @ beta), "durbin_watson": durbin_watson(u)}


def plant_tables(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    irr_min = cfg["counting_irradiance_min"]
    primary = cfg["performance_model"]["primary"]["features"]
    robust = cfg["performance_model"]["robustness"]["features"]

    t = availability(df, irr_min)
    for alt in cfg["sensitivity_counting_irradiance_min"]:
        t[f"availability_at_{alt}"] = availability(df, alt)["availability"]
    present = df.groupby("DATE_TIME")["SOURCE_KEY"].nunique()
    common = present.index[present == df["SOURCE_KEY"].nunique()]
    t["availability_common_window"] = availability(
        df[df["DATE_TIME"].isin(common)], irr_min)["availability"]

    rows = counted(df, irr_min)
    rows = rows[rows[d4.TARGET] > 0].sort_values("DATE_TIME")
    perf = {key: slope(g, primary) for key, g in rows.groupby("SOURCE_KEY")}
    t = t.join(pd.DataFrame(perf).T.astype(float).add_prefix("perf_"))
    t["robust_slope"] = pd.Series({key: slope(g, robust)["slope"]
                                   for key, g in rows.groupby("SOURCE_KEY")})
    t = t.join(sister_ratio(df, irr_min).add_prefix("sister_"))

    median_slope = float(t["perf_slope"].median())
    t["ci_contains_plant_median"] = (t["perf_ci_low"] <= median_slope) & (median_slope <= t["perf_ci_high"])
    z_bonf = float(stats.norm.ppf(1 - 0.05 / (2 * len(t))))
    t["bonferroni_contains_plant_median"] = (
        (t["perf_slope"] - z_bonf * t["perf_se_hac"]).le(median_slope)
        & (t["perf_slope"] + z_bonf * t["perf_se_hac"]).ge(median_slope))
    t["rank_availability"] = t["availability"].rank(method="min").astype(int)
    t["rank_availability_common_window"] = t["availability_common_window"].rank(method="min").astype(int)
    t["rank_slope"] = t["perf_slope"].rank(method="min").astype(int)
    t["rank_robust_slope"] = t["robust_slope"].rank(method="min").astype(int)
    t["rank_sister"] = t["sister_sister_ratio"].rank(method="min").astype(int)

    bottom_slope = set(t.nsmallest(3, "perf_slope").index)
    bottom_sister = set(t.nsmallest(3, "sister_sister_ratio").index)
    summary = {
        "median_slope": median_slope,
        "indistinguishable_from_median": int(t["ci_contains_plant_median"].sum()),
        "bonferroni_z": z_bonf,
        "expected_chance_misses_at_95": 0.05 * len(t),
        "flagged_surviving_bonferroni": sorted(
            t.index[~t["ci_contains_plant_median"] & ~t["bonferroni_contains_plant_median"]]),
        "spearman_slope_vs_robust": float(stats.spearmanr(t["perf_slope"], t["robust_slope"])[0]),
        "spearman_slope_vs_sister": float(stats.spearmanr(t["perf_slope"],
                                                          t["sister_sister_ratio"])[0]),
        "bottom3_slope": sorted(bottom_slope), "bottom3_sister": sorted(bottom_sister),
        "bottom3_overlap": len(bottom_slope & bottom_sister),
        "counted_rows": int(t["counted"].sum()), "producing_counted_rows": int(len(rows)),
        "availability_median": float(t["availability"].median()),
        "availability_min": float(t["availability"].min()),
    }
    for alt in cfg["sensitivity_counting_irradiance_min"]:
        summary[f"spearman_availability_vs_at_{alt}"] = float(
            stats.spearmanr(t["availability"], t[f"availability_at_{alt}"])[0])
    summary["common_window"] = {
        "timestamps": int(len(common)), "of_timestamps": int(len(present)),
        "counted_timestamps": int(counted(df[df["DATE_TIME"].isin(common)],
                                          irr_min)["DATE_TIME"].nunique()),
        "availability_median": float(t["availability_common_window"].median()),
        "availability_min": float(t["availability_common_window"].min()),
        "spearman_vs_declared": float(stats.spearmanr(t["availability"],
                                                      t["availability_common_window"])[0])}

    X = rows[list(d4.FEATURES)].to_numpy(dtype=float)
    y = rows[d4.TARGET].to_numpy(dtype=float)
    summary["posthoc_plant_r2_producing_counted"] = {
        "label": "post-hoc check, not a replacement for the daylight fit",
        "n": len(y), "r2": r2(y, add_intercept(X) @ normal_equations(X, y))}
    return t.sort_values("perf_slope"), summary


def main() -> int:
    sha, dirty = git_state()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = REPO / "runs" / f"d4_rank_{run}"
    dest.mkdir(parents=True)
    out = {"run": run, "git_sha": sha, "git_dirty": dirty, "config": cfg, "z": Z, "plants": {}}

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    for plant in (1, 2):
        t, s = plant_tables(d4.load(plant), cfg)
        t.to_csv(dest / f"plant{plant}.csv", float_format="%.6g")
        out["plants"][plant] = {"summary": s, "inverters": json.loads(t.to_json(orient="index"))}
        print(f"\nplant {plant}: median slope {s['median_slope']:.2f}, "
              f"{s['indistinguishable_from_median']}/22 CIs contain it, "
              f"{len(s['flagged_surviving_bonferroni'])} of the flagged survive Bonferroni "
              f"(z {s['bonferroni_z']:.3f}): {s['flagged_surviving_bonferroni']}; "
              f"Spearman slope~robust {s['spearman_slope_vs_robust']:.3f}, "
              f"slope~sister {s['spearman_slope_vs_sister']:.3f}, "
              f"bottom-3 overlap {s['bottom3_overlap']}/3; "
              f"post-hoc plant R2 on producing counted {s['posthoc_plant_r2_producing_counted']['r2']:.4f}")
        cw = s["common_window"]
        print(f"common window {cw['timestamps']:,} of {cw['of_timestamps']:,} timestamps "
              f"({cw['counted_timestamps']:,} counted): availability median "
              f"{cw['availability_median']:.4f} (min {cw['availability_min']:.4f}), "
              f"Spearman vs declared {cw['spearman_vs_declared']:.3f}")
        print(t[["counted", "missing", "availability", "availability_common_window",
                 "perf_slope", "perf_ci_low", "perf_ci_high",
                 "ci_contains_plant_median", "bonferroni_contains_plant_median",
                 "robust_slope", "sister_sister_ratio"]].round(4))

    (dest / "rank.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    p1, p2 = out["plants"][1]["summary"], out["plants"][2]["summary"]
    print(f"\nwrote {dest.relative_to(REPO).as_posix()}/\n\nEXPERIMENTS row:")
    print(f"| {run} | `{sha[:7]}` | {'yes' if dirty else 'no'} | `scripts/rank_d4.py` "
          f"| {p1['producing_counted_rows']:,} / {p2['producing_counted_rows']:,} "
          f"| {p1['availability_median']:.4f} ({p1['availability_min']:.4f}) / "
          f"{p2['availability_median']:.4f} ({p2['availability_min']:.4f}) "
          f"| {p1['indistinguishable_from_median']} / {p2['indistinguishable_from_median']} "
          f"| {p1['spearman_slope_vs_robust']:.3f} / {p2['spearman_slope_vs_robust']:.3f} "
          f"| {p1['spearman_slope_vs_sister']:.3f} / {p2['spearman_slope_vs_sister']:.3f} "
          f"| {p1['bottom3_overlap']} / {p2['bottom3_overlap']} "
          f"| {p1['common_window']['spearman_vs_declared']:.3f} / "
          f"{p2['common_window']['spearman_vs_declared']:.3f} "
          f"| {22 - p1['indistinguishable_from_median']} ({len(p1['flagged_surviving_bonferroni'])}) / "
          f"{22 - p2['indistinguishable_from_median']} ({len(p2['flagged_surviving_bonferroni'])}) |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
