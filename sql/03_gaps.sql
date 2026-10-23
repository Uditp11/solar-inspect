-- 3. Data gaps per inverter: steps between an inverter's consecutive generation rows
-- longer than 15 minutes, found with LAG(). missing_intervals counts the 15-minute rows
-- such a step skips ((step - 900 s) / 900 s, summed). Time before an inverter's first row
-- or after its last is not counted. Every inverter is listed, with 0 where it has no gap.
WITH steps AS (
    SELECT plant,
           source_key,
           CAST(strftime('%s', date_time) AS INTEGER)
             - CAST(strftime('%s', LAG(date_time) OVER (PARTITION BY plant, source_key
                                                        ORDER BY date_time)) AS INTEGER) AS step_s
    FROM generation
)
SELECT plant,
       source_key,
       SUM(CASE WHEN step_s > 900 THEN 1 ELSE 0 END)                  AS gaps,
       SUM(CASE WHEN step_s > 900 THEN (step_s - 900) / 900 ELSE 0 END) AS missing_intervals
FROM steps
GROUP BY plant, source_key
ORDER BY plant, source_key;
