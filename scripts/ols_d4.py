"""AC_POWER on irradiation, module temperature and ambient temperature, per plant.

    python scripts/ols_d4.py

Fits each plant twice: on daylight rows only (`IRRADIATION > 0`), which is the
fit, and on every row with a weather reading, night included, which exists only
to measure how much the night rows inflate R². Night is roughly half of D4 with
AC_POWER = 0 at zero irradiation, so including it hands the model "predict zero
in the dark" for free.

Each fit is solved by both solvers in solar_inspect.analytics.ols and checked
against `np.linalg.lstsq` (LAPACK) and scikit-learn's `LinearRegression`; those
two are references, not solvers of this project.

**These are in-sample fit statistics for inference, not a predictive score.**
Nothing is held out. R² here says how much of the variance in these 34 days the
three weather columns account for; it says nothing about the next 34. The
standard errors are not reported at all yet: 15-minute residuals are
autocorrelated, so plain OLS standard errors would be wrong.

Plant 1's DC_POWER defect is reported from the data and left as it is.
DC_POWER is not an input.

Writes runs/ols_d4_<utc>/fit.json (gitignored) and prints the docs/EXPERIMENTS.md
row for the run.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.analytics import data as d4                          # noqa: E402
from solar_inspect.analytics.ols import (add_intercept, adjusted_r2,     # noqa: E402
                                         gradient_descent, normal_equations, r2)
from solar_inspect.classification.train import git_state                 # noqa: E402

NOTE = ("In-sample fit statistics for inference, not a predictive score. Nothing is "
        "held out, and OLS standard errors are not reported because the residuals "
        "are autocorrelated.")


def fit(df) -> dict:
    X = df[list(d4.FEATURES)].to_numpy(dtype=float)
    y = df[d4.TARGET].to_numpy(dtype=float)
    A = add_intercept(X)
    ne = normal_equations(X, y)
    gd, steps = gradient_descent(X, y)
    ls = np.linalg.lstsq(A, y, rcond=None)[0]
    sk = LinearRegression().fit(X, y)
    sk = np.concatenate([[sk.intercept_], sk.coef_])
    r2_value = r2(y, A @ ne)
    names = ["intercept", *d4.FEATURES]
    return {
        "n": len(y),
        "normal_equations": dict(zip(names, ne.tolist())),
        "gradient_descent": dict(zip(names, gd.tolist())),
        "gd_steps": steps,
        "max_abs_diff": {
            "normal_equations_vs_lstsq": float(np.abs(ne - ls).max()),
            "normal_equations_vs_sklearn": float(np.abs(ne - sk).max()),
            "gradient_descent_vs_lstsq": float(np.abs(gd - ls).max()),
            "gradient_descent_vs_sklearn": float(np.abs(gd - sk).max()),
            "gradient_descent_vs_normal_equations": float(np.abs(gd - ne).max()),
        },
        "r2": r2_value,
        "adjusted_r2": adjusted_r2(r2_value, len(y), X.shape[1]),
        "cond_X": float(np.linalg.cond(A)),
        "cond_XtX": float(np.linalg.cond(A.T @ A)),
    }


def main() -> int:
    sha, dirty = git_state()
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = {"run": run, "git_sha": sha, "git_dirty": dirty, "note": NOTE,
           "target": d4.TARGET, "features": list(d4.FEATURES), "plants": {}}

    print(NOTE)
    for plant in (1, 2):
        df = d4.load(plant)
        dc = d4.dc_scale_check(df)
        if not dc["plausible"]:
            print(f"\nWARNING plant {plant}: median AC/DC = {dc['median_ac_dc']:.4f}, outside "
                  f"{d4.PLAUSIBLE_AC_DC}. DC_POWER is mis-scaled; reported, not rescaled.")
        day, every = fit(d4.daylight(df)), fit(df)
        out["plants"][plant] = {"dc_scale_check": dc, "daylight": day,
                                "with_night": every,
                                "r2_inflation_from_night": every["r2"] - day["r2"]}

        print(f"\nplant {plant}: daylight n={day['n']:,}  (with night n={every['n']:,})")
        print(f"  {'':22s}{'normal eq.':>14s}{'grad. descent':>16s}")
        for k in day["normal_equations"]:
            print(f"  {k:22s}{day['normal_equations'][k]:14.4f}{day['gradient_descent'][k]:16.4f}")
        print(f"  GD steps {day['gd_steps']:,}")
        for k, v in day["max_abs_diff"].items():
            print(f"  max |diff| {k:38s} {v:.2e}")
        print(f"  R2 {day['r2']:.4f}  adj R2 {day['adjusted_r2']:.4f}  "
              f"cond(X) {day['cond_X']:.3g}  cond(XtX) {day['cond_XtX']:.3g}")
        print(f"  with night: R2 {every['r2']:.4f}  adj R2 {every['adjusted_r2']:.4f}  "
              f"(+{every['r2'] - day['r2']:.4f} from night rows)")

    dest = REPO / "runs" / f"ols_d4_{run}"
    dest.mkdir(parents=True)
    (dest / "fit.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    p1, p2 = out["plants"][1], out["plants"][2]
    print(f"\nwrote {(dest / 'fit.json').relative_to(REPO).as_posix()}\n\nEXPERIMENTS row:")
    print(f"| {run} | `{sha[:7]}` | {'yes' if dirty else 'no'} | `scripts/ols_d4.py` "
          f"| {p1['daylight']['n']:,} / {p2['daylight']['n']:,} "
          f"| **{p1['daylight']['r2']:.4f}** / **{p2['daylight']['r2']:.4f}** "
          f"| {p1['with_night']['r2']:.4f} / {p2['with_night']['r2']:.4f} "
          f"| {max(p1['daylight']['max_abs_diff']['normal_equations_vs_lstsq'], p2['daylight']['max_abs_diff']['normal_equations_vs_lstsq']):.1e} "
          f"| {max(p1['daylight']['max_abs_diff']['gradient_descent_vs_lstsq'], p2['daylight']['max_abs_diff']['gradient_descent_vs_lstsq']):.1e} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
