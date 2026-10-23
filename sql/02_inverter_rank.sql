-- 2. Inverters ranked by irradiance slope within their plant, lowest first.
-- RANK() gives tied slopes the same rank and skips the next, as pandas' rank(method="min").
SELECT plant,
       source_key,
       slope,
       RANK() OVER (PARTITION BY plant ORDER BY slope) AS slope_rank
FROM inverter
ORDER BY plant, slope_rank, source_key;
