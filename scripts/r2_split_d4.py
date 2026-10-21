"""Post-hoc check: where Plant 2's low plant-level R² comes from.

    python scripts/r2_split_d4.py

**Post-hoc.** Made after the per-inverter rankings were seen, to explain a
number, not to choose a model. It never replaces the declared daylight fit of
scripts/ols_d4.py, which stays the plant-level result.

Same model as that fit (AC_POWER on the three weather columns, one model per
plant over all 22 inverters), refitted on four row sets that separate two
changes the per-inverter rules make at once:

                        zeros kept            producing only (AC_POWER > 0)
    IRRADIATION > 0     the declared fit      daylight, producing
    IRRADIATION >= 0.1  counted, zeros kept   counted, producing

The threshold is counting_irradiance_min from configs/d4_analysis.yaml. OLS
and R² are solar_inspect.analytics.

Writes runs/d4_r2split_<utc>/r2_split.json (gitignored) and prints the
docs/EXPERIMENTS.md row for the run.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.analytics import data as d4                          # noqa: E402
from solar_inspect.analytics.inverters import counted                    # noqa: E402
from solar_inspect.analytics.ols import add_intercept, normal_equations, r2  # noqa: E402
from solar_inspect.classification.train import git_state                 # noqa: E402

LABEL = "post-hoc check made after the rankings; not a replacement for the declared daylight fit"


def fit_r2(rows) -> dict:
    X = rows[list(d4.FEATURES)].to_numpy(dtype=float)
    y = rows[d4.TARGET].to_numpy(dtype=float)
    return {"n": len(y), "r2": r2(y, add_intercept(X) @ normal_equations(X, y))}


def main() -> int:
    sha, dirty = git_state()
    cfg = yaml.safe_load((REPO / "configs" / "d4_analysis.yaml").read_text(encoding="utf-8"))
    irr_min = cfg["counting_irradiance_min"]
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = REPO / "runs" / f"d4_r2split_{run}"
    dest.mkdir(parents=True)
    out = {"run": run, "git_sha": sha, "git_dirty": dirty, "label": LABEL,
           "counting_irradiance_min": irr_min, "plants": {}}

    print(LABEL)
    for plant in (1, 2):
        df = d4.load(plant)
        sets = {"daylight": d4.daylight(df), "counted": counted(df, irr_min)}
        res = {}
        for name, rows in sets.items():
            res[f"{name}_zeros_kept"] = fit_r2(rows)
            res[f"{name}_producing"] = fit_r2(rows[rows[d4.TARGET] > 0])
        out["plants"][plant] = res
        print(f"plant {plant}: " + "  ".join(f"{k} {v['r2']:.4f} (n {v['n']:,})" for k, v in res.items()))

    (dest / "r2_split.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    p1, p2 = out["plants"][1], out["plants"][2]
    cells = ("daylight_zeros_kept", "daylight_producing", "counted_zeros_kept", "counted_producing")
    print(f"\nwrote {dest.relative_to(REPO).as_posix()}/\n\nEXPERIMENTS row:")
    print(f"| {run} | `{sha[:7]}` | {'yes' if dirty else 'no'} | `scripts/r2_split_d4.py` | "
          + " | ".join(f"{p1[c]['r2']:.4f} ({p1[c]['n']:,}) / {p2[c]['r2']:.4f} ({p2[c]['n']:,})"
                       for c in cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
