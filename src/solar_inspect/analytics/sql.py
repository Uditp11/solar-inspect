"""The analytics database on SQLite (Python's standard-library sqlite3), and the pandas
equivalent of every query in sql/.

`build` creates the database from sql/schema.sql and fills it from frames the caller
has already computed: D4 generation and weather rows, the per-plant DC check, the
per-inverter ranking, ADR 0006's CoA table and the D1 triage findings. `query` runs
one file of sql/ with named parameters. Each `*_pd` function computes the same rows
in pandas, with the same columns, types and order, so a query and its equivalent
can be compared frame to frame (tests/test_sql.py on a hand-worked fixture,
scripts/sql_d4.py on the real data).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from solar_inspect.analytics.triage import EXTENT_ORDER

REPO = Path(__file__).resolve().parents[3]
SQL_DIR = REPO / "sql"
EXTENT_RANK = {"none": 0, **{e: i + 1 for i, e in enumerate(EXTENT_ORDER)}}
STEP_S = 900                                   # one D4 interval, 15 minutes


def _stamp(s: pd.Series) -> pd.Series:
    return s.dt.strftime("%Y-%m-%d %H:%M:%S")


def _day(s: pd.Series) -> pd.Series:
    return s.dt.strftime("%Y-%m-%d")


def build(path: str | Path, generation: pd.DataFrame, weather: pd.DataFrame,
          plants: pd.DataFrame, inverters: pd.DataFrame,
          coa_table: dict[str, tuple[int, str, str]], findings: pd.DataFrame
          ) -> sqlite3.Connection:
    """A fresh database at `path` (any existing file there is replaced)."""
    path = Path(path)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.executescript((SQL_DIR / "schema.sql").read_text(encoding="utf-8"))

    def put(table: str, cols: list[str], rows: list[list]) -> None:
        con.executemany(f"INSERT INTO {table} ({', '.join(cols)}) "
                        f"VALUES ({', '.join('?' * len(cols))})", rows)

    g = generation.assign(DATE_TIME=_stamp(generation["DATE_TIME"]))
    put("generation", ["plant", "date_time", "source_key", "dc_power", "ac_power",
                       "daily_yield", "total_yield"],
        g[["PLANT_ID", "DATE_TIME", "SOURCE_KEY", "DC_POWER", "AC_POWER", "DAILY_YIELD",
           "TOTAL_YIELD"]].to_numpy(object).tolist())
    w = weather.assign(DATE_TIME=_stamp(weather["DATE_TIME"]))
    put("weather", ["plant", "date_time", "ambient_temperature", "module_temperature",
                    "irradiation"],
        w[["PLANT_ID", "DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
           "IRRADIATION"]].to_numpy(object).tolist())
    for table, df in (("plant", plants), ("inverter", inverters), ("finding", findings)):
        cols = [c for c in df.columns]
        put(table, cols, [[v.item() if isinstance(v, np.generic) else v for v in r]
                          for r in df[cols].to_numpy(object).tolist()])
    put("coa_class", ["defect_class", "coa", "extent", "extent_order", "action"],
        [[c, coa, ext, EXTENT_RANK[ext], act] for c, (coa, ext, act) in coa_table.items()])
    con.commit()
    return con


def query(con: sqlite3.Connection, name: str, **params) -> pd.DataFrame:
    """Run sql/<name>.sql with named parameters (`:name` in the file)."""
    text = (SQL_DIR / f"{name}.sql").read_text(encoding="utf-8")
    return pd.read_sql_query(text, con, params=params)


# ---- the pandas equivalents ---------------------------------------------------------

def daily_pd(generation: pd.DataFrame, weather: pd.DataFrame, min_daylight_rows: int
             ) -> pd.DataFrame:
    w = weather[weather["IRRADIATION"] > 0]
    d = (w.assign(plant=w["PLANT_ID"], day=_day(w["DATE_TIME"]))
         .groupby(["plant", "day"])["IRRADIATION"]
         .agg(daylight_rows="size", median_irradiation="median").reset_index())
    d = d[d["daylight_rows"] >= min_daylight_rows]
    e = (generation.assign(plant=generation["PLANT_ID"], day=_day(generation["DATE_TIME"]))
         .groupby(["plant", "day"])["AC_POWER"].sum().mul(0.25)
         .rename("ac_energy_kwh").reset_index())
    out = d.merge(e, on=["plant", "day"])
    return (out[["plant", "day", "daylight_rows", "ac_energy_kwh", "median_irradiation"]]
            .sort_values(["plant", "day"]).reset_index(drop=True)
            .astype({"plant": "int64", "daylight_rows": "int64"}))


def rank_pd(inverters: pd.DataFrame) -> pd.DataFrame:
    out = inverters[["plant", "source_key", "slope"]].assign(
        slope_rank=inverters.groupby("plant")["slope"].rank(method="min").astype("int64"))
    return (out.sort_values(["plant", "slope_rank", "source_key"]).reset_index(drop=True)
            .astype({"plant": "int64"}))


def gaps_pd(generation: pd.DataFrame) -> pd.DataFrame:
    g = generation.sort_values(["PLANT_ID", "SOURCE_KEY", "DATE_TIME"])
    step = g.groupby(["PLANT_ID", "SOURCE_KEY"])["DATE_TIME"].diff().dt.total_seconds()
    gap = step > STEP_S
    out = pd.DataFrame({"plant": g["PLANT_ID"], "source_key": g["SOURCE_KEY"],
                        "gaps": gap.astype("int64"),
                        "missing_intervals": ((step - STEP_S) // STEP_S).where(gap, 0)
                        .astype("int64")})
    return (out.groupby(["plant", "source_key"], as_index=False)[["gaps", "missing_intervals"]]
            .sum().sort_values(["plant", "source_key"]).reset_index(drop=True)
            .astype({"plant": "int64"}))


def sister_daily_pd(generation: pd.DataFrame) -> pd.DataFrame:
    daily = (generation.assign(plant=generation["PLANT_ID"], source_key=generation["SOURCE_KEY"],
                               day=_day(generation["DATE_TIME"]))
             .groupby(["plant", "source_key", "day"])["AC_POWER"].sum().mul(0.25)
             .rename("kwh").reset_index())
    med = daily.groupby(["plant", "day"])["kwh"].median().rename("median_kwh").reset_index()
    r = daily.merge(med, on=["plant", "day"])
    r = r[r["median_kwh"] > 0]
    r = r.assign(r=r["kwh"] / r["median_kwh"])
    out = (r.groupby(["plant", "source_key"])["r"]
           .agg(days="size", median_daily_ratio="median").reset_index())
    out["ratio_rank"] = out.groupby("plant")["median_daily_ratio"].rank(method="min").astype("int64")
    return (out.sort_values(["plant", "ratio_rank", "source_key"]).reset_index(drop=True)
            .astype({"plant": "int64", "days": "int64"}))


def work_orders_pd(findings: pd.DataFrame, coa_table: dict[str, tuple[int, str, str]],
                   t_val: float) -> pd.DataFrame:
    c = pd.DataFrame([(k, coa, ext, EXTENT_RANK[ext], act)
                      for k, (coa, ext, act) in coa_table.items()],
                     columns=["triage_class", "coa", "extent", "extent_order", "action"])
    f = findings[findings["p_any"] >= t_val].merge(c, on="triage_class")
    f = f.sort_values(["coa", "extent_order", "p_class", "finding_id"],
                      ascending=[False, True, False, True]).reset_index(drop=True)
    f["dispatch_rank"] = np.arange(1, len(f) + 1, dtype="int64")
    f["rank_in_coa"] = (f.groupby("coa").cumcount() + 1).astype("int64")
    return f[["dispatch_rank", "finding_id", "triage_class", "coa", "extent", "action",
              "p_class", "p_any", "rank_in_coa"]].astype({"coa": "int64"})
