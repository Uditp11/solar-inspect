-- 1. Per plant per day: daylight intervals, AC energy and median irradiance.
--
-- daylight_rows: weather intervals with irradiation > 0 (one sensor per plant).
-- ac_energy_kwh: sum over the plant's generation rows of ac_power (kW) x 0.25 h.
-- median_irradiation: over the day's daylight intervals. SQLite has no median, so it
--   is the middle row (odd n) or the mean of the two middle rows (even n) of
--   ROW_NUMBER() over the ordered values, with COUNT() over the same partition.
-- HAVING drops days with fewer than :min_daylight_rows daylight intervals.
WITH ordered AS (
    SELECT plant,
           date(date_time) AS day,
           irradiation,
           ROW_NUMBER() OVER (PARTITION BY plant, date(date_time) ORDER BY irradiation) AS rn,
           COUNT(*)     OVER (PARTITION BY plant, date(date_time))                      AS n
    FROM weather
    WHERE irradiation > 0
),
daylight AS (
    SELECT plant, day, MAX(n) AS daylight_rows, AVG(irradiation) AS median_irradiation
    FROM ordered
    WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
    GROUP BY plant, day
    HAVING MAX(n) >= :min_daylight_rows
),
energy AS (
    SELECT plant, date(date_time) AS day, SUM(ac_power) * 0.25 AS ac_energy_kwh
    FROM generation
    GROUP BY plant, date(date_time)
)
SELECT d.plant, d.day, d.daylight_rows, e.ac_energy_kwh, d.median_irradiation
FROM daylight AS d
JOIN energy AS e USING (plant, day)
ORDER BY d.plant, d.day;
