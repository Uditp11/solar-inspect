-- 5. The work-order list: findings flagged at p_any >= :t_val, joined to ADR 0006's CoA
-- table, numbered by the dispatch order (CoA descending, extent, P(class) descending,
-- image name), with each finding's place within its CoA. The top 5 per CoA are the rows
-- with rank_in_coa <= 5. delta_dn_uncalibrated is not used.
WITH flagged AS (
    SELECT f.finding_id, f.triage_class, c.coa, c.extent, c.extent_order, c.action,
           f.p_class, f.p_any
    FROM finding AS f
    JOIN coa_class AS c ON c.defect_class = f.triage_class
    WHERE f.p_any >= :t_val
),
ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (ORDER BY coa DESC, extent_order, p_class DESC, finding_id)
             AS dispatch_rank
    FROM flagged
)
SELECT dispatch_rank, finding_id, triage_class, coa, extent, action, p_class, p_any,
       ROW_NUMBER() OVER (PARTITION BY coa ORDER BY dispatch_rank) AS rank_in_coa
FROM ranked
ORDER BY dispatch_rank;
