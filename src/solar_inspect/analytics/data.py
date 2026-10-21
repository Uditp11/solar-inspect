"""D4 (two plants, 34 days, 15-minute rows): load, merge, clean.

Every CSV is parsed with an explicit `format=`. Only `Plant_1_Generation_Data.csv`
is day-first; a parser left to infer the format reads 05-06-2020 as 6 May, and
for the first 12 days of each month nothing fails to say so. A row that does not
match its file's format raises instead of being guessed.

Each plant has one weather sensor, so the generation-to-weather merge is
many-to-one on (DATE_TIME, PLANT_ID) and is validated as such: no generation row
may gain a duplicate. Generation rows whose timestamp has no weather reading are
dropped by `load()`, and the count is visible as NaN in `merge()`.

The night-row filter is deliberately not part of `load()`. `daylight()` is its own
step so the same frame can be fitted with and without it and the R² it inflates
stays measurable.

Plant 1's DC_POWER is about 10x too large (median AC/DC 9.78% against Plant 2's
97.84%, see docs/DATA.md). `dc_scale_check()` detects that from the data and
reports it. Nothing here rescales it: DC_POWER is never a model input, and a
silent fix would hide the defect from every later reader of the frame.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
D4 = REPO / "data" / "d4"

FORMATS = {
    "Plant_1_Generation_Data.csv": "%d-%m-%Y %H:%M",
    "Plant_2_Generation_Data.csv": "%Y-%m-%d %H:%M:%S",
    "Plant_1_Weather_Sensor_Data.csv": "%Y-%m-%d %H:%M:%S",
    "Plant_2_Weather_Sensor_Data.csv": "%Y-%m-%d %H:%M:%S",
}

TARGET = "AC_POWER"
FEATURES = ("IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE")
# DC_POWER is AC_POWER over efficiency (R² ~ 1 for free); the yields are cumulative.
NEVER_INPUTS = ("DC_POWER", "DAILY_YIELD", "TOTAL_YIELD")

# A string inverter converts DC to AC at well above 90%. Outside this band the
# DC column, not the inverter, is the suspect.
PLAUSIBLE_AC_DC = (0.90, 1.0)


def read(src, fmt: str) -> pd.DataFrame:
    df = pd.read_csv(src)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"], format=fmt)
    return df


def merge(plant: int) -> pd.DataFrame:
    """Generation rows with their plant's weather columns. Rows without weather keep NaN."""
    gen_name = f"Plant_{plant}_Generation_Data.csv"
    wx_name = f"Plant_{plant}_Weather_Sensor_Data.csv"
    gen = read(D4 / gen_name, FORMATS[gen_name])
    wx = read(D4 / wx_name, FORMATS[wx_name]).drop(columns="SOURCE_KEY")
    out = gen.merge(wx, on=["DATE_TIME", "PLANT_ID"], how="left", validate="many_to_one")
    assert len(out) == len(gen), "weather merge duplicated generation rows"
    return out


def load(plant: int) -> pd.DataFrame:
    """The merged frame, minus generation rows that have no weather reading."""
    return merge(plant).dropna(subset=list(FEATURES)).reset_index(drop=True)


def daylight(df: pd.DataFrame) -> pd.DataFrame:
    """IRRADIATION > 0. About half of D4 is night with AC_POWER = 0."""
    return df[df["IRRADIATION"] > 0].reset_index(drop=True)


def dc_scale_check(df: pd.DataFrame) -> dict:
    """Median AC/DC over rows with DC_POWER > 0, and whether it is a plausible efficiency."""
    on = df[df["DC_POWER"] > 0]
    ratio = float((on["AC_POWER"] / on["DC_POWER"]).median())
    lo, hi = PLAUSIBLE_AC_DC
    return {"median_ac_dc": ratio, "plausible": lo <= ratio <= hi, "rows": len(on)}
