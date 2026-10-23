"""Build the SQLite database from real data and run the five queries in sql/.

    python scripts/sql_d4.py

Each query's rows are compared with its pandas equivalent (analytics/sql.py):
same columns, same order, integer and text columns equal, float columns equal to
a relative 1e-12 (SQLite and pandas may add the same floats in a different order),
and the largest float difference is printed. Beyond its twin:

- query 2's ranks must equal scripts/rank_d4.py's slope ranks exactly;
- query 4 (daily energy) is compared with the existing per-timestamp sister ratio
  by Spearman, per plant, and the difference is reported, not forced;
- query 5 must equal the triage run's work_orders.csv row for row.

Inputs: the cleaned D4 frames (analytics.data), the per-inverter table recomputed
by rank_d4.plant_tables, ADR 0006's CoA table, and the triage run's findings.csv,
at the threshold its config reads. Refuses to run from a dirty tree.

Writes runs/sql_d4_<utc>/ (gitignored): solar_inspect.db and sql.json.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from rank_d4 import plant_tables                                      # noqa: E402
from solar_inspect.analytics import data as d4                       # noqa: E402
from solar_inspect.analytics import sql                              # noqa: E402
from solar_inspect.analytics.triage import COA_TABLE                 # noqa: E402
from solar_inspect.classification.train import git_state             # noqa: E402

TRIAGE_RUN = "runs/triage_d1_20261023T171927Z"
# Query 1's HAVING floor, declared before the first run: 40 daylight intervals is
# 10 hours of the 15-minute grid, most of a mid-May to mid-June day.
MIN_DAYLIGHT_ROWS = 40
RTOL = 1e-12


def same(name: str, a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Compare a query's rows with its twin; stop on any difference beyond float rounding."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        raise SystemExit(f"{name}: shape or columns differ: {a.shape} {list(a.columns)} "
                         f"vs {b.shape} {list(b.columns)}")
    worst = 0.0
    for c in a.columns:
        if pd.api.types.is_float_dtype(a[c]) or pd.api.types.is_float_dtype(b[c]):
            x, y = a[c].to_numpy(float), b[c].to_numpy(float)
            if not np.allclose(x, y, rtol=RTOL, atol=0):
                raise SystemExit(f"{name}: column {c} differs beyond rtol {RTOL}")
            worst = max(worst, float(np.max(np.abs(x - y))) if len(x) else 0.0)
        elif a[c].tolist() != b[c].tolist():
            raise SystemExit(f"{name}: column {c} differs")
    print(f"  {name}: {len(a)} rows, equal to pandas (largest float difference {worst:.3g})")
    return {"rows": len(a), "max_float_abs_diff": worst}


def main() -> int:
    sha, dirty = git_state()
    if dirty:
        raise SystemExit("the working tree is not clean; commit first")
    cfg = yaml.safe_load((REPO / "configs/d4_analysis.yaml").read_text(encoding="utf-8"))
    tcfg = yaml.safe_load((REPO / "configs/triage.yaml").read_text(encoding="utf-8"))
    t_val = json.loads((REPO / tcfg["threshold_from"]).read_text(encoding="utf-8"))["t_val"]

    gens, wxs, plants, invs = [], [], [], []
    for p in (1, 2):
        g = d4.load(p)
        gens.append(g)
        pid = int(g["PLANT_ID"].iloc[0])                 # 4135001 / 4136001, not 1 / 2
        assert (g["PLANT_ID"] == pid).all()
        name = f"Plant_{p}_Weather_Sensor_Data.csv"
        wxs.append(d4.read(d4.D4 / name, d4.FORMATS[name]))
        dc = d4.dc_scale_check(g)
        plants.append({"plant": pid, "dc_power_misscaled": int(not dc["plausible"]),
                       "median_ac_dc": dc["median_ac_dc"]})
        t, _ = plant_tables(g, cfg)
        invs.append(pd.DataFrame({
            "plant": pid, "source_key": t.index, "slope": t["perf_slope"],
            "slope_ci_low": t["perf_ci_low"], "slope_ci_high": t["perf_ci_high"],
            "sister_ratio": t["sister_sister_ratio"], "sister_n": t["sister_n"].astype(int),
            "python_slope_rank": t["rank_slope"]}).reset_index(drop=True))
    gen, wx = pd.concat(gens, ignore_index=True), pd.concat(wxs, ignore_index=True)
    inv = pd.concat(invs, ignore_index=True)
    plants = pd.DataFrame(plants)
    find = pd.read_csv(REPO / TRIAGE_RUN / "findings.csv", dtype={"finding_id": str},
                       float_precision="round_trip").drop(columns="dispatch_rank")

    dest = REPO / "runs" / f"sql_d4_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    dest.mkdir(parents=True)
    con = sql.build(dest / "solar_inspect.db", gen, wx, plants,
                    inv.drop(columns="python_slope_rank"), COA_TABLE, find)
    counts = {tb: con.execute(f"SELECT COUNT(*) FROM {tb}").fetchone()[0]
              for tb in ("generation", "weather", "plant", "inverter", "coa_class", "finding")}
    print("tables:", counts)
    print("plant:", con.execute("SELECT * FROM plant").fetchall())
    out = {"git_sha": sha, "triage_run": TRIAGE_RUN, "t_val": t_val,
           "min_daylight_rows": MIN_DAYLIGHT_ROWS, "tables": counts, "queries": {}}
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    def show(q: pd.DataFrame, n: int = 6) -> None:
        print(q.head(n).to_string(index=False))

    print("\n1. per plant per day")
    q1 = sql.query(con, "01_daily_plant", min_daylight_rows=MIN_DAYLIGHT_ROWS)
    all_days = sql.query(con, "01_daily_plant", min_daylight_rows=0)
    out["queries"]["01_daily_plant"] = {**same("01", q1, sql.daily_pd(gen, wx, MIN_DAYLIGHT_ROWS)),
                                        "days_before_having": len(all_days),
                                        "dropped_by_having": all_days[all_days["daylight_rows"]
                                                                      < MIN_DAYLIGHT_ROWS]
                                        .to_dict("records")}
    show(q1)
    print(f"  {len(all_days)} plant-days before HAVING, {len(q1)} after")

    print("\n2. inverters ranked by slope within plant")
    q2 = sql.query(con, "02_inverter_rank")
    r2 = same("02", q2, sql.rank_pd(inv))
    py = inv.set_index(["plant", "source_key"])["python_slope_rank"]
    if q2.set_index(["plant", "source_key"])["slope_rank"].sort_index().tolist() != \
            py.sort_index().tolist():
        raise SystemExit("02: SQL ranks differ from rank_d4.py's")
    print("  equal to scripts/rank_d4.py's rank_slope for all 44 inverters")
    out["queries"]["02_inverter_rank"] = {**r2, "equals_rank_d4": True}
    show(q2)

    print("\n3. data gaps per inverter")
    q3 = sql.query(con, "03_gaps")
    out["queries"]["03_gaps"] = {**same("03", q3, sql.gaps_pd(gen)),
                                 "inverters_with_gaps": int((q3["gaps"] > 0).sum()),
                                 "per_plant": q3.groupby("plant")[["gaps", "missing_intervals"]]
                                 .sum().to_dict("index")}
    show(q3.sort_values("missing_intervals", ascending=False))

    print("\n4. sister comparison on daily energy")
    q4 = sql.query(con, "04_sister_daily")
    r4 = same("04", q4, sql.sister_daily_pd(gen))
    m = q4.merge(inv[["plant", "source_key", "sister_ratio"]], on=["plant", "source_key"])
    if len(m) != len(inv):
        raise SystemExit(f"04: {len(m)} of {len(inv)} inverters joined to the sister ratios")
    rho = {int(p): float(stats.spearmanr(g["median_daily_ratio"], g["sister_ratio"])[0])
           for p, g in m.groupby("plant")}
    bottom = {int(p): {"daily": sorted(g.nsmallest(3, "median_daily_ratio")["source_key"]),
                       "per_timestamp": sorted(g.nsmallest(3, "sister_ratio")["source_key"])}
              for p, g in m.groupby("plant")}
    print(f"  Spearman with the per-timestamp sister ratio: {rho}")
    print(f"  bottom 3: {bottom}")
    out["queries"]["04_sister_daily"] = {**r4, "spearman_vs_sister_ratio": rho, "bottom3": bottom}
    show(q4)

    print("\n5. work-order list")
    q5 = sql.query(con, "05_work_orders", t_val=t_val)
    r5 = same("05", q5, sql.work_orders_pd(find, COA_TABLE, t_val))
    b5 = pd.read_csv(REPO / TRIAGE_RUN / "work_orders.csv", dtype={"finding_id": str},
                     float_precision="round_trip")
    cols = ["dispatch_rank", "finding_id", "triage_class", "coa", "extent", "action", "p_class",
            "p_any"]
    if not q5[cols].equals(b5[cols]):
        raise SystemExit("05: SQL work orders differ from the triage run's work_orders.csv")
    print(f"  equal to {TRIAGE_RUN}/work_orders.csv row for row ({len(q5)} rows)")
    out["queries"]["05_work_orders"] = {**r5, "equals_triage_csv": True,
                                        "per_coa": q5.groupby("coa").size().to_dict()}
    print("  top 5 per CoA:")
    print(q5[q5["rank_in_coa"] <= 5].to_string(index=False))

    con.close()
    (dest / "sql.json").write_text(json.dumps(out, indent=2, default=str) + "\n",
                                   encoding="utf-8", newline="\n")
    print(f"\nwrote {dest.relative_to(REPO).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
