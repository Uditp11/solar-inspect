"""The five SQL queries, each against its pandas equivalent and a hand-worked fixture.

Written before analytics/sql.py and the query files. The fixture is small enough
to work every expected row by hand:

weather (irradiation, kW/m2)            generation (AC_POWER, kW; '-' = no row)
plant 1  day 1  06:00 0.0                    06:00 06:15 06:30 06:45 | d2 06:00 06:15
                06:15 0.2               A      0     4     8     4  |      8     8
                06:30 0.4               B      0     2     4     -  |      4     4
                06:45 0.1               C      0     -     -     2  |     12    12
         day 2  06:00 0.3
                06:15 0.5               plant 2, day 1 only:
plant 2  day 1  06:00 0.5               D      1     1   (06:00, 06:15)
                06:15 0.7

1. daily, min 3 daylight rows: plant 1 day 1 only: 3 daylight rows (0.2, 0.4, 0.1),
   median 0.2, energy (16 + 6 + 2) x 0.25 = 6.0 kWh. Day 2 (2 rows) and plant 2
   (2 rows) fall to HAVING. With min 2: also p1 d2 (2, 12.0 kWh, median 0.4) and
   p2 d1 (2, 0.5 kWh, median 0.6).
2. slopes A 10, B 8, C 8, D 5 -> RANK within plant: B 1, C 1, A 3; D 1.
3. gaps (> 15 min between an inverter's consecutive rows), missing 15-min rows:
   A: 06:45 -> d2 06:00 = 23 h 15 min: 1 gap, 92 missing
   B: 06:30 -> d2 06:00 = 23 h 30 min: 1 gap, 93 missing
   C: 06:00 -> 06:45 (2 missing) and 06:45 -> d2 06:00 (92): 2 gaps, 94 missing
   D: none.
4. daily kWh  d1: A 4.0, B 1.5, C 0.5 (median 1.5)  d2: A 4.0, B 2.0, C 6.0 (median 4.0)
   ratios     d1: A 8/3, B 1, C 1/3                d2: A 1, B 0.5, C 1.5
   median over 2 days = mean: A 11/6, B 0.75, C 11/12 -> rank B 1, C 2, A 3; D 1.0, 1.
5. findings, t = 0.5 (p_any >= t): order a, e (Diode 0.9, tie by id), c (Hot-Spot),
   b (Offline-Module, module), f (Cell 0.30, flagged at p_any 0.55), d (Soiling,
   external); g (p_any 0.1) is not dispatched. Rank within CoA: 1 2 3 | 1 2 3.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from solar_inspect.analytics import sql
from solar_inspect.analytics.triage import COA_TABLE


def ts(day: int, hm: str) -> pd.Timestamp:
    return pd.Timestamp(f"2020-05-{14 + day} {hm}")


GEN = pd.DataFrame(
    [(1, ts(1, "06:00"), "A", 0), (1, ts(1, "06:15"), "A", 4), (1, ts(1, "06:30"), "A", 8),
     (1, ts(1, "06:45"), "A", 4), (1, ts(2, "06:00"), "A", 8), (1, ts(2, "06:15"), "A", 8),
     (1, ts(1, "06:00"), "B", 0), (1, ts(1, "06:15"), "B", 2), (1, ts(1, "06:30"), "B", 4),
     (1, ts(2, "06:00"), "B", 4), (1, ts(2, "06:15"), "B", 4),
     (1, ts(1, "06:00"), "C", 0), (1, ts(1, "06:45"), "C", 2), (1, ts(2, "06:00"), "C", 12),
     (1, ts(2, "06:15"), "C", 12),
     (2, ts(1, "06:00"), "D", 1), (2, ts(1, "06:15"), "D", 1)],
    columns=["PLANT_ID", "DATE_TIME", "SOURCE_KEY", "AC_POWER"])
GEN["AC_POWER"] = GEN["AC_POWER"].astype(float)
GEN["DC_POWER"] = GEN["AC_POWER"] * 1.02
GEN["DAILY_YIELD"] = 0.0
GEN["TOTAL_YIELD"] = 0.0
WX = pd.DataFrame(
    [(1, ts(1, "06:00"), 0.0), (1, ts(1, "06:15"), 0.2), (1, ts(1, "06:30"), 0.4),
     (1, ts(1, "06:45"), 0.1), (1, ts(2, "06:00"), 0.3), (1, ts(2, "06:15"), 0.5),
     (2, ts(1, "06:00"), 0.5), (2, ts(1, "06:15"), 0.7)],
    columns=["PLANT_ID", "DATE_TIME", "IRRADIATION"])
WX["AMBIENT_TEMPERATURE"] = 25.0
WX["MODULE_TEMPERATURE"] = 30.0
PLANTS = pd.DataFrame({"plant": [1, 2], "dc_power_misscaled": [0, 0],
                       "median_ac_dc": [0.98, 0.98]})
INV = pd.DataFrame({"plant": [1, 1, 1, 2], "source_key": ["A", "B", "C", "D"],
                    "slope": [10.0, 8.0, 8.0, 5.0], "slope_ci_low": [9.0, 7.0, 7.0, 4.0],
                    "slope_ci_high": [11.0, 9.0, 9.0, 6.0],
                    "sister_ratio": [1.1, 0.9, 1.0, 1.0], "sister_n": [3, 3, 3, 2]})


def finding(fid: str, cls: str, p_class: float, p_any: float) -> dict:
    return {"finding_id": fid, "frame_index": ord(fid[0]) - 97, "source_frame_path": fid,
            "defect_class": cls, "cls_confidence": p_class, "triage_class": cls,
            "p_class": p_class, "p_any": p_any, "delta_dn_uncalibrated": 1.0}


FIND = pd.DataFrame([finding("a.jpg", "Diode", 0.9, 0.9),
                     finding("b.jpg", "Offline-Module", 0.6, 0.6),
                     finding("c.jpg", "Hot-Spot", 0.7, 0.7),
                     finding("d.jpg", "Soiling", 0.8, 0.8),
                     finding("e.jpg", "Diode", 0.9, 0.9),
                     finding("f.jpg", "Cell", 0.30, 0.55),
                     finding("g.jpg", "Diode", 0.1, 0.1)])


@pytest.fixture()
def db(tmp_path):
    con = sql.build(tmp_path / "t.db", GEN, WX, PLANTS, INV, COA_TABLE, FIND)
    yield con
    con.close()


def test_q1_daily_by_hand_and_pandas(db):
    got = sql.query(db, "01_daily_plant", min_daylight_rows=3)
    assert got.to_dict("records") == [{"plant": 1, "day": "2020-05-15", "daylight_rows": 3,
                                       "ac_energy_kwh": 6.0, "median_irradiation": 0.2}]
    got2 = sql.query(db, "01_daily_plant", min_daylight_rows=2)
    assert got2[["plant", "day", "daylight_rows"]].values.tolist() == [
        [1, "2020-05-15", 3], [1, "2020-05-16", 2], [2, "2020-05-15", 2]]
    assert got2["ac_energy_kwh"].tolist() == [6.0, 12.0, 0.5]
    assert got2["median_irradiation"].tolist() == pytest.approx([0.2, 0.4, 0.6])
    for m in (2, 3):
        pd.testing.assert_frame_equal(sql.query(db, "01_daily_plant", min_daylight_rows=m),
                                      sql.daily_pd(GEN, WX, m))


def test_q2_rank_within_plant(db):
    got = sql.query(db, "02_inverter_rank")
    assert got[["plant", "source_key", "slope_rank"]].values.tolist() == [
        [1, "B", 1], [1, "C", 1], [1, "A", 3], [2, "D", 1]]
    pd.testing.assert_frame_equal(got, sql.rank_pd(INV))


def test_q3_gaps(db):
    got = sql.query(db, "03_gaps")
    assert got[["plant", "source_key", "gaps", "missing_intervals"]].values.tolist() == [
        [1, "A", 1, 92], [1, "B", 1, 93], [1, "C", 2, 94], [2, "D", 0, 0]]
    pd.testing.assert_frame_equal(got, sql.gaps_pd(GEN))


def test_q4_sister_daily(db):
    got = sql.query(db, "04_sister_daily")
    assert got[["plant", "source_key", "days", "ratio_rank"]].values.tolist() == [
        [1, "B", 2, 1], [1, "C", 2, 2], [1, "A", 2, 3], [2, "D", 1, 1]]
    assert got["median_daily_ratio"].tolist() == pytest.approx([0.75, 11 / 12, 11 / 6, 1.0])
    pd.testing.assert_frame_equal(got, sql.sister_daily_pd(GEN))


def test_q5_work_orders(db):
    got = sql.query(db, "05_work_orders", t_val=0.5)
    assert got["finding_id"].tolist() == ["a.jpg", "e.jpg", "c.jpg", "b.jpg", "f.jpg", "d.jpg"]
    assert got["dispatch_rank"].tolist() == [1, 2, 3, 4, 5, 6]
    assert got["rank_in_coa"].tolist() == [1, 2, 3, 1, 2, 3]
    assert got["coa"].tolist() == [3, 3, 3, 2, 2, 2]
    pd.testing.assert_frame_equal(got, sql.work_orders_pd(FIND, COA_TABLE, 0.5))


def test_schema_keys_refuse_duplicates_and_bad_coa(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO generation (plant, date_time, source_key, dc_power, ac_power)"
                   " VALUES (1, '2020-05-15 06:00:00', 'A', 0, 0)")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO coa_class VALUES ('X', 4, 'cell', 3, 'none')")
