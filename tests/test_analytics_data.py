"""D4 loading and cleaning.

The parsing tests run on in-memory CSV text and need no data. The count tests run
on the real CSVs and are skipped when `data/d4/` is absent; their expected values
are the measured counts in docs/DATA.md, not numbers produced by this loader.
"""
from __future__ import annotations

import io

import pandas as pd
import pytest

from solar_inspect.analytics import data as d4

HAVE_D4 = all((d4.D4 / f).exists() for f in d4.FORMATS)
needs_d4 = pytest.mark.skipif(not HAVE_D4, reason="data/d4 not downloaded")


def test_plant1_generation_is_day_first():
    # 05-06-2020 is 5 June. A parser left to infer the format reads it as 6 May.
    src = io.StringIO("DATE_TIME,PLANT_ID\n05-06-2020 00:15,4135001\n")
    t = d4.read(src, d4.FORMATS["Plant_1_Generation_Data.csv"])["DATE_TIME"].iloc[0]
    assert t == pd.Timestamp(2020, 6, 5, 0, 15)


@pytest.mark.parametrize("name", ["Plant_2_Generation_Data.csv",
                                  "Plant_1_Weather_Sensor_Data.csv",
                                  "Plant_2_Weather_Sensor_Data.csv"])
def test_the_other_three_files_are_iso(name):
    src = io.StringIO("DATE_TIME,PLANT_ID\n2020-06-05 00:15:00,4136001\n")
    assert d4.read(src, d4.FORMATS[name])["DATE_TIME"].iloc[0] == pd.Timestamp(2020, 6, 5, 0, 15)


def test_a_row_in_the_wrong_format_raises_instead_of_guessing():
    src = io.StringIO("DATE_TIME,PLANT_ID\n2020-06-05 00:15:00,4135001\n")
    with pytest.raises(ValueError):
        d4.read(src, d4.FORMATS["Plant_1_Generation_Data.csv"])


def test_cumulative_and_dc_columns_are_never_features():
    assert not set(d4.FEATURES) & set(d4.NEVER_INPUTS)
    assert d4.TARGET not in d4.FEATURES


@needs_d4
@pytest.mark.parametrize("plant, gen_rows, no_weather, kept", [
    (1, 68_778, 4, 68_774),
    (2, 67_698, 0, 67_698),
])
def test_merge_adds_no_rows_and_drops_only_rows_without_weather(plant, gen_rows, no_weather, kept):
    merged = d4.merge(plant)
    assert len(merged) == gen_rows                       # many-to-one: no duplicates gained
    assert merged["IRRADIATION"].isna().sum() == no_weather
    assert len(d4.load(plant)) == kept


@needs_d4
@pytest.mark.parametrize("plant, rows", [(1, 38_376), (2, 38_722)])
def test_daylight_filter_matches_data_md(plant, rows):
    assert len(d4.daylight(d4.load(plant))) == rows


@needs_d4
def test_plant1_dc_defect_is_detected_and_not_rescaled():
    p1, p2 = d4.dc_scale_check(d4.load(1)), d4.dc_scale_check(d4.load(2))
    assert p1["median_ac_dc"] == pytest.approx(0.0978, abs=5e-5)
    assert p2["median_ac_dc"] == pytest.approx(0.9784, abs=5e-5)
    assert not p1["plausible"] and p2["plausible"]
    raw = d4.read(d4.D4 / "Plant_1_Generation_Data.csv", d4.FORMATS["Plant_1_Generation_Data.csv"])
    assert d4.load(1)["DC_POWER"].max() == raw["DC_POWER"].max()
