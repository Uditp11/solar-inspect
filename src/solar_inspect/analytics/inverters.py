"""Per-inverter availability and the sister ratio, written here in pandas.

Both take generation rows already merged with their plant's weather
(DATE_TIME, SOURCE_KEY, IRRADIATION, AC_POWER) for one plant, and count only
intervals with IRRADIATION at or above `irr_min`, so that dawn and dusk, when
an inverter is legitimately not yet producing, are not outages. The rules are
declared in configs/d4_analysis.yaml. Cross-checked against hand-worked toy
frames in tests/test_inverters.py.

`availability`: producing (AC_POWER > 0) counted intervals / counted intervals.
A timestamp with no row for an inverter is a data gap, not an outage; it is
reported as `missing`, the plant's counted timestamps that inverter has no row
for.

`sister_ratio`: no regression. At each counted timestamp, an inverter's
AC_POWER over the median AC_POWER of the plant's producing inverters at that
timestamp (itself included when producing); a non-producing inverter scores 0.
A timestamp with no producing inverter has no reference and gives no ratio for
anyone. Summary per inverter: the median ratio and `n`, the number of ratios
behind it. `sister_ratios` gives the ratio of every counted row.

`zero_runs`: run lengths of AC_POWER == 0 over an inverter's counted
intervals. A stretch continues only to the next 15-minute step; a missing row
or an interval below `irr_min` ends it. One row per stretch: inverter, start,
length in intervals.
"""
from __future__ import annotations

import pandas as pd


def counted(df: pd.DataFrame, irr_min: float) -> pd.DataFrame:
    return df[(df["IRRADIATION"] > 0) & (df["IRRADIATION"] >= irr_min)]


def availability(df: pd.DataFrame, irr_min: float) -> pd.DataFrame:
    c = counted(df, irr_min)
    g = c.groupby("SOURCE_KEY")["AC_POWER"]
    out = pd.DataFrame({"counted": g.size(), "producing": g.apply(lambda s: int((s > 0).sum()))})
    out["missing"] = c["DATE_TIME"].nunique() - out["counted"]
    out["availability"] = out["producing"] / out["counted"]
    return out


def sister_ratios(df: pd.DataFrame, irr_min: float) -> pd.Series:
    c = counted(df, irr_min)
    ref = c[c["AC_POWER"] > 0].groupby("DATE_TIME")["AC_POWER"].median()
    return c["AC_POWER"] / c["DATE_TIME"].map(ref)    # NaN where nobody is producing


def sister_ratio(df: pd.DataFrame, irr_min: float) -> pd.DataFrame:
    ratio = sister_ratios(df, irr_min)
    g = ratio.groupby(df.loc[ratio.index, "SOURCE_KEY"])
    return pd.DataFrame({"n": g.count(), "sister_ratio": g.median()})


def zero_runs(df: pd.DataFrame, irr_min: float,
              step: pd.Timedelta = pd.Timedelta(minutes=15)) -> pd.DataFrame:
    c = counted(df, irr_min).sort_values(["SOURCE_KEY", "DATE_TIME"])
    z = c[c["AC_POWER"] == 0]
    new = (z["SOURCE_KEY"] != z["SOURCE_KEY"].shift()) | (z["DATE_TIME"].diff() != step)
    g = z.groupby(new.cumsum().to_numpy())
    return pd.DataFrame({"SOURCE_KEY": g["SOURCE_KEY"].first(), "start": g["DATE_TIME"].first(),
                         "intervals": g.size()}).reset_index(drop=True)
