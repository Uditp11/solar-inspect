"""D1's triage threshold, chosen on val and applied once to the saved test logits.

    python scripts/threshold_val_d1.py

Rules in configs/cls_triage_threshold.yaml, committed before this script first
read a test logit. It replaces the threshold scripts/posthoc_cls_d1.py chose on
the test scores (t = 0.7151), which made that operating point's test numbers
optimistic. No model is run: the val and test logits were both saved by the one
test pass.

Score: calibrated P(any anomaly) = 1 - P(No-Anomaly), at the temperature fitted
on val. The threshold is the highest-recall one whose precision at 2.2% field
prevalence is >= 0.50 on **val** (calibration.pick_threshold); that t is then
applied to the test logits once. Refuses to run from a dirty tree.

Writes runs/cls_triage_threshold_<utc>/threshold.json (gitignored).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.calibration import (at_threshold,   # noqa: E402
                                                      pick_threshold, softmax)
from solar_inspect.classification.train import git_state              # noqa: E402

CONFIG = "configs/cls_triage_threshold.yaml"
SWEEP = (0.10, 0.25, 0.50, 0.90, 0.95, 0.99)


def row(r: dict) -> str:
    return (f"{r['t']:>8.4f}{r['flags_per_1000_field']:>14.1f}{r['flags']:>8}{r['recall']:>9.4f}"
            f"{r['precision_measured']:>12.4f}{r['precision_field']:>14.4f}")


def main() -> int:
    sha, dirty = git_state()
    if dirty:
        raise SystemExit("the working tree is not clean; commit first")
    cfg = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
    src = REPO / cfg["source"]
    run = src.parent
    z = np.load(src)
    classes = json.loads((run / "manifest.json").read_text(encoding="utf-8"))["classes"]
    na = classes.index(cfg["null_class"])
    T = json.loads((run / "posthoc.json").read_text(encoding="utf-8"))["temperature"]
    if abs(T - cfg["temperature"]) > 5e-5:
        raise SystemExit(f"posthoc.json T = {T}, config says {cfg['temperature']}")
    p, target = cfg["field_prevalence"], cfg["precision_target"]

    def scores(split: str) -> tuple[np.ndarray, np.ndarray]:
        return (1.0 - softmax(z[f"{split}_logits"], T)[:, na],
                z[f"{split}_labels"] != na)

    sv, av = scores("val")
    head = (f"{'t':>8}{'flags/1000':>14}{'flags':>8}{'recall':>9}{'prec meas':>12}"
            f"{'prec @2.2%':>14}")
    print(f"T = {T:.6f} (val NLL fit)   field prevalence {p}   target precision {target}")
    print(f"\n=== VAL sweep, {len(sv)} images, No-Anomaly share {1 - av.mean():.4f} ===")
    print(head)
    sweep = [at_threshold(sv, av, t, p) for t in SWEEP]
    for r in sweep:
        print(row(r))

    t_val = pick_threshold(sv, av, p, target)
    out = {"git_sha": sha, "config_path": CONFIG, "config": cfg, "temperature": T,
           "val_sweep": sweep, "t_val": t_val}
    if t_val is None:
        print(f"\nUNREACHABLE on val: no threshold gives precision >= {target} at {p:.1%}. "
              "No fallback threshold; test is not read.")
    else:
        val = at_threshold(sv, av, t_val, p)
        print(f"\nchosen on val: t_val = {t_val:.6f}")
        print(head)
        print(row(val))

        st, at = scores("test")
        test = at_threshold(st, at, t_val, p)
        print(f"\n=== TEST, t_val applied once, {len(st)} images, No-Anomaly share "
              f"{1 - at.mean():.4f} ===")
        print(head)
        print(row(test))
        out.update(val=val, test=test)

        # Not a choice: the same rule on the test scores must give back the
        # superseded, test-chosen threshold, which shows the rule is August's.
        t_old = pick_threshold(st, at, p, target)
        old = at_threshold(st, at, t_old, p)
        print(f"\nreproduction of the superseded test-chosen threshold (not used): "
              f"t = {t_old:.4f}")
        print(row(old))
        out["superseded_test_chosen"] = old

    dest = REPO / "runs" / f"cls_triage_threshold_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    dest.mkdir(parents=True)
    (dest / "threshold.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8",
                                         newline="\n")
    print(f"\nwrote {dest.relative_to(REPO).as_posix()}/threshold.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
