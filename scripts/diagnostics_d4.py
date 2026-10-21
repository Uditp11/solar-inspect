"""Diagnostics for the plant-level daylight fit of scripts/ols_d4.py.

    python scripts/diagnostics_d4.py

Same fit, same rows: AC_POWER on the three weather columns, IRRADIATION > 0,
one model per plant over all 22 inverters. Reported per plant: R² and adjusted
R², VIF per feature, Durbin-Watson, the residual ACF at lags 1 to 96 (one day of
15-minute steps), classical against Newey-West standard errors, and
Breusch-Pagan.

**The fit pools 22 series.** D4 stores one row per inverter per timestamp, so in
file order the next row is the next inverter at the same moment, not the same
inverter 15 minutes later. Durbin-Watson, the ACF and the Newey-West lags are
therefore computed within each inverter's own series (rows sorted by inverter,
then time). Durbin-Watson in file order is printed too, to show what that order
would measure instead. A lag counts rows of an inverter's daylight series: the
night break and any missing timestamp are not filled, so a lag that spans one
is not a multiple of 15 minutes. The gaps are counted and reported.

All inverters of a plant share one weather sensor, so their residuals at one
timestamp share that sensor's error. Within-inverter Newey-West does not see
that; the Driscoll-Kraay covariance (scores summed per timestamp, then
Newey-West over time) does, and is reported as a reference. Driscoll-Kraay,
the ACF and Breusch-Pagan are statsmodels; everything else is
solar_inspect.analytics. Newey-West lags follow configs/d4_analysis.yaml,
n being the rows of the fit.

Writes runs/d4_diag_<utc>/ (gitignored): diagnostics.json and three figures,
residual vs fitted, Q-Q and ACF.
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
from scipy import stats                                                  # noqa: E402
from statsmodels.stats.diagnostic import het_breuschpagan                # noqa: E402
from statsmodels.tsa.stattools import acf                                # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.analytics import data as d4                          # noqa: E402
from solar_inspect.analytics.covariance import (classical_cov, hac_cov,  # noqa: E402
                                                newey_west_lags)
from solar_inspect.analytics.diagnostics import durbin_watson, vif       # noqa: E402
from solar_inspect.analytics.ols import (add_intercept, adjusted_r2,     # noqa: E402
                                         normal_equations, r2)
from solar_inspect.classification.train import git_state                 # noqa: E402

ACF_LAGS = 96
STEP = pd.Timedelta(minutes=15)
NAMES = ["intercept", *d4.FEATURES]


def gaps(day: pd.DataFrame) -> dict:
    """Steps between consecutive rows of one inverter's daylight series that are not 15 minutes.

    Bucketed by length only. Length is not a reliable label: Plant 2's sensor
    reads IRRADIATION > 0 at some night hours, which splits a night break into
    shorter pieces, and some inverters have days with no rows at all.
    """
    dt = day.groupby("SOURCE_KEY")["DATE_TIME"].diff().dropna()
    off = dt[dt != STEP] / pd.Timedelta(hours=1)
    return {"steps": int(len(dt)), "not_15_min": int(len(off)),
            "up_to_1h": int((off <= 1).sum()),
            "over_1h_up_to_6h": int(((off > 1) & (off <= 6)).sum()),
            "over_6h_up_to_24h": int(((off > 6) & (off <= 24)).sum()),
            "over_24h": int((off > 24).sum()),
            "longest_h": float(off.max()) if len(off) else 0.0}


def diagnose(df: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    file_order = d4.daylight(df)
    day = file_order.sort_values(["SOURCE_KEY", "DATE_TIME"]).reset_index(drop=True)
    X = day[list(d4.FEATURES)].to_numpy(dtype=float)
    y = day[d4.TARGET].to_numpy(dtype=float)
    A = add_intercept(X)
    beta = normal_equations(X, y)
    fitted = A @ beta
    u = y - fitted
    n, k = A.shape
    groups = day["SOURCE_KEY"].to_numpy()
    lags = newey_west_lags(n)

    r2_value = r2(y, fitted)
    adj = adjusted_r2(r2_value, n, k - 1)

    per_inv = {key: g for key, g in pd.Series(u).groupby(groups)}
    dw_each = {key: durbin_watson(e.to_numpy()) for key, e in per_inv.items()}
    ss = {key: float((e.to_numpy() ** 2).sum()) for key, e in per_inv.items()}
    dw_within = sum(dw_each[key] * ss[key] for key in per_inv) / float(u @ u)
    u_file = file_order[d4.TARGET].to_numpy(float) - add_intercept(
        file_order[list(d4.FEATURES)].to_numpy(float)) @ beta
    acf_each = np.array([acf(e.to_numpy(), nlags=ACF_LAGS, fft=True) for e in per_inv.values()])
    acf_mean = acf_each.mean(0)

    cl = classical_cov(A, u)
    nw = hac_cov(A, u, lags, groups=groups)
    time = np.unique(day["DATE_TIME"].to_numpy(), return_inverse=True)[1]
    dk = sm.OLS(y, A).fit(cov_type="hac-groupsum", cov_kwds={
        "time": time, "maxlags": lags, "use_correction": False}).cov_params()
    se = {name: np.sqrt(np.diag(c)) for name, c in
          (("classical", cl), ("newey_west_within_inverter", nw), ("driscoll_kraay", dk))}
    lm, lm_p, f, f_p = het_breuschpagan(u, A)

    out = {
        "n": n, "inverters": len(per_inv), "timestamps": int(time.max() + 1),
        "coefficients": dict(zip(NAMES, beta.tolist())),
        "r2": r2_value, "adjusted_r2": adj, "r2_minus_adjusted": r2_value - adj,
        "vif": dict(zip(d4.FEATURES, vif(X).tolist())),
        "durbin_watson": {
            "within_inverter_pooled": dw_within,
            "per_inverter_min": min(dw_each.values()),
            "per_inverter_median": float(np.median(list(dw_each.values()))),
            "per_inverter_max": max(dw_each.values()),
            "file_order_not_a_time_lag": durbin_watson(u_file),
        },
        "acf_mean_over_inverters": {int(l): float(acf_mean[l]) for l in range(1, ACF_LAGS + 1)},
        "acf_note": "statsmodels acf per inverter series (each series demeaned), "
                    "mean over the 22 inverters; lag = rows of the daylight series",
        "gaps": gaps(day),
        "hac_lags": lags,
        "standard_errors": {name: dict(zip(NAMES, s.tolist())) for name, s in se.items()},
        "se_ratio_to_classical": {
            name: dict(zip(NAMES, (s / se["classical"]).tolist()))
            for name, s in se.items() if name != "classical"},
        "breusch_pagan": {"lm": float(lm), "lm_p": float(lm_p), "f": float(f), "f_p": float(f_p)},
    }
    return out, fitted, u, acf_mean


def main() -> int:
    sha, dirty = git_state()
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = REPO / "runs" / f"d4_diag_{run}"
    dest.mkdir(parents=True)
    out = {"run": run, "git_sha": sha, "git_dirty": dirty, "plants": {}}
    figs = {name: plt.subplots(1, 2, figsize=(11, 4.2)) for name in ("resid_vs_fitted", "qq", "acf")}

    for col, plant in enumerate((1, 2)):
        res, fitted, u, acf_mean = diagnose(d4.load(plant))
        out["plants"][plant] = res

        ax = figs["resid_vs_fitted"][1][col]
        ax.scatter(fitted, u, s=1, alpha=0.15, rasterized=True)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(title=f"Plant {plant}", xlabel="fitted AC_POWER", ylabel="residual")

        ax = figs["qq"][1][col]
        (osm, osr), (slope, icpt, _) = stats.probplot(u, dist="norm")
        ax.scatter(osm, osr, s=1, alpha=0.3, rasterized=True)
        ax.plot(osm, slope * osm + icpt, color="k", lw=0.8)
        ax.set(title=f"Plant {plant}", xlabel="normal quantile", ylabel="residual quantile")

        ax = figs["acf"][1][col]
        ax.bar(range(1, ACF_LAGS + 1), acf_mean[1:], width=0.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(title=f"Plant {plant}", xlabel="lag (rows of an inverter's daylight series)",
               ylabel="mean ACF over inverters", ylim=(-0.2, 1))

        dw = res["durbin_watson"]
        print(f"\nplant {plant}: n={res['n']:,}  R2 {res['r2']:.6f}  adj {res['adjusted_r2']:.6f}  "
              f"VIF {', '.join(f'{v:.2f}' for v in res['vif'].values())}")
        print(f"  DW within inverter {dw['within_inverter_pooled']:.4f} "
              f"(per inverter {dw['per_inverter_min']:.4f} to {dw['per_inverter_max']:.4f}); "
              f"file order {dw['file_order_not_a_time_lag']:.4f}")
        print(f"  ACF lag 1 {acf_mean[1]:.3f}  lag 4 {acf_mean[4]:.3f}  lag 16 {acf_mean[16]:.3f}  "
              f"lag 48 {acf_mean[48]:.3f}  lag 96 {acf_mean[96]:.3f}   gaps {res['gaps']}")
        for name, ratios in res["se_ratio_to_classical"].items():
            print(f"  SE ratio {name} / classical (L={res['hac_lags']}): "
                  + ", ".join(f"{k} {v:.2f}" for k, v in ratios.items()))
        print(f"  Breusch-Pagan LM {res['breusch_pagan']['lm']:.1f}  p {res['breusch_pagan']['lm_p']:.3g}")

    for name, (fig, _) in figs.items():
        fig.tight_layout()
        fig.savefig(dest / f"{name}.png", dpi=110)
    (dest / "diagnostics.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8",
                                           newline="\n")
    print(f"\nwrote {dest.relative_to(REPO).as_posix()}/ (git {sha[:7]}, dirty={dirty})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
