-- solar-inspect analytics database, SQLite. Built by solar_inspect.analytics.sql.build
-- from the cleaned D4 frames, the per-inverter ranking, ADR 0006's CoA table and the D1
-- triage findings. Timestamps are 'YYYY-MM-DD HH:MM:SS' text, the plants' local time.

-- D4 generation rows that have a weather reading (analytics.data.load), both plants.
-- AC_POWER and DC_POWER in kW per the dataset's data card. DC_POWER is stored as
-- recorded: Plant 1's is about 10x too large and is reported, not rescaled (see plant).
CREATE TABLE generation (
    plant        INTEGER NOT NULL,
    date_time    TEXT    NOT NULL,
    source_key   TEXT    NOT NULL,      -- the inverter
    dc_power     REAL    NOT NULL,
    ac_power     REAL    NOT NULL,
    daily_yield  REAL,
    total_yield  REAL,
    PRIMARY KEY (plant, source_key, date_time)
);
CREATE INDEX generation_plant_time ON generation (plant, date_time);

-- One weather sensor per plant. The temperature columns are the sensor's readings.
CREATE TABLE weather (
    plant                INTEGER NOT NULL,
    date_time            TEXT    NOT NULL,
    ambient_temperature  REAL    NOT NULL,
    module_temperature   REAL    NOT NULL,
    irradiation          REAL    NOT NULL,   -- kW/m2
    PRIMARY KEY (plant, date_time)
);

-- dc_power_misscaled: 1 where analytics.data.dc_scale_check finds the median AC/DC
-- outside a plausible inverter efficiency (Plant 1). Nothing is rescaled.
CREATE TABLE plant (
    plant               INTEGER PRIMARY KEY,
    dc_power_misscaled  INTEGER NOT NULL CHECK (dc_power_misscaled IN (0, 1)),
    median_ac_dc        REAL    NOT NULL
);

-- Per inverter, recomputed by scripts/rank_d4.py's plant_tables: the irradiance slope
-- of AC_POWER with its 95% Newey-West interval, and the median sister ratio.
CREATE TABLE inverter (
    plant          INTEGER NOT NULL REFERENCES plant (plant),
    source_key     TEXT    NOT NULL,
    slope          REAL    NOT NULL,
    slope_ci_low   REAL    NOT NULL,
    slope_ci_high  REAL    NOT NULL,
    sister_ratio   REAL    NOT NULL,
    sister_n       INTEGER NOT NULL,
    PRIMARY KEY (plant, source_key)
);

-- ADR 0006: Class of Abnormality by failure mode. extent_order is the dispatch order
-- (module 1, substring 2, cell 3, external 4; none 0 for CoA 1).
CREATE TABLE coa_class (
    defect_class  TEXT    PRIMARY KEY,
    coa           INTEGER NOT NULL CHECK (coa IN (1, 2, 3)),
    extent        TEXT    NOT NULL CHECK (extent IN ('none', 'module', 'substring', 'cell', 'external')),
    extent_order  INTEGER NOT NULL,
    action        TEXT    NOT NULL
);

-- One row per D1 test image, from scripts/triage_d1.py's findings.csv. triage_class is
-- the argmax over the eleven anomaly classes; p_any = 1 - P(No-Anomaly), calibrated.
-- delta_dn_uncalibrated is an intra-crop contrast, not comparable between crops; no
-- query ranks on it.
CREATE TABLE finding (
    finding_id             TEXT    PRIMARY KEY,   -- the image file name
    frame_index            INTEGER NOT NULL UNIQUE,
    source_frame_path      TEXT    NOT NULL,
    defect_class           TEXT    NOT NULL REFERENCES coa_class (defect_class),
    cls_confidence         REAL    NOT NULL,
    triage_class           TEXT    NOT NULL REFERENCES coa_class (defect_class),
    p_class                REAL    NOT NULL,
    p_any                  REAL    NOT NULL,
    delta_dn_uncalibrated  REAL    NOT NULL
);
CREATE INDEX finding_triage_class ON finding (triage_class);
CREATE INDEX finding_p_any ON finding (p_any);
