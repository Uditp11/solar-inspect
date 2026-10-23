-- 4. Sister comparison on daily energy: each inverter's AC energy on a day divided by
-- the median inverter's energy that day in the same plant, then the per-inverter median
-- of that ratio, ranked within the plant (lowest first). Medians by ROW_NUMBER() and
-- COUNT() as in query 1. A day whose median energy is 0 gives no ratio. An inverter with
-- no row on a day has no energy that day and no ratio; one with only part of a day's
-- rows has a low ratio that day.
WITH daily AS (
    SELECT plant, source_key, date(date_time) AS day, SUM(ac_power) * 0.25 AS kwh
    FROM generation
    GROUP BY plant, source_key, date(date_time)
),
daily_ordered AS (
    SELECT plant, day, kwh,
           ROW_NUMBER() OVER (PARTITION BY plant, day ORDER BY kwh) AS rn,
           COUNT(*)     OVER (PARTITION BY plant, day)              AS n
    FROM daily
),
plant_median AS (
    SELECT plant, day, AVG(kwh) AS median_kwh
    FROM daily_ordered
    WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
    GROUP BY plant, day
),
ratio AS (
    SELECT d.plant, d.source_key, d.kwh / m.median_kwh AS r
    FROM daily AS d
    JOIN plant_median AS m USING (plant, day)
    WHERE m.median_kwh > 0
),
ratio_ordered AS (
    SELECT plant, source_key, r,
           ROW_NUMBER() OVER (PARTITION BY plant, source_key ORDER BY r) AS rn,
           COUNT(*)     OVER (PARTITION BY plant, source_key)            AS n
    FROM ratio
)
SELECT plant,
       source_key,
       MAX(n)  AS days,
       AVG(r)  AS median_daily_ratio,
       RANK() OVER (PARTITION BY plant ORDER BY AVG(r)) AS ratio_rank
FROM ratio_ordered
WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
GROUP BY plant, source_key
ORDER BY plant, ratio_rank, source_key;
