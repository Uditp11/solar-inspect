"""D1 imbalance ablation: four arms x three seeds, evaluated on VAL.

    python scripts/ablation_d1.py

The arms and the budget are declared in configs/cls_ablation.yaml, which was
committed before this script was first run. So was the noise floor -- 0.02
macro-F1 -- and the rule this script mechanically applies at the end: two arms
whose means differ by less than that are printed as indistinguishable and are
not ranked. Declaring the threshold in advance and then honouring it is worth
more than the ablation, because it is the only thing separating an experiment
from a leaderboard.

Twelve runs share one load of D1 onto the GPU. Every run appends a row to
docs/EXPERIMENTS.md carrying its arm, split hash and git SHA; the whole thing is
also written to runs/cls_ablation_<utc>/ as one JSON with per-epoch histories.

Evaluation is on val, through train.py's EVAL_SPLIT. The test split is not read
here and is not read anywhere until configs/cls_final.yaml is committed.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.data import load_d1                    # noqa: E402
from solar_inspect.classification.train import (EVAL_SPLIT, NULL_CLASS,  # noqa: E402
                                                evaluate, fit, git_state,
                                                log_experiment, metrics)

CONFIG = "configs/cls_ablation.yaml"


def main() -> int:
    sha, dirty = git_state()        # before anything is written; see train.git_state
    cfg = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
    split_sha = json.loads((REPO / "configs" / "d1_split.json")
                           .read_text(encoding="utf-8"))["sha256"]
    floor = float(cfg["noise_floor_macro_f1"])
    arms, seeds = cfg["arms"], cfg["seeds"]

    d = load_d1()
    rows, bs = d.index[EVAL_SPLIT], cfg["batch_size"]
    null_acc = float((d.labels[rows] == d.classes.index(NULL_CLASS)).float().mean())
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"\nablation {run}: {len(arms)} arms x {len(seeds)} seeds, "
          f"declared noise floor {floor} macro-F1\n")

    out: dict[str, dict] = {}
    for arm, overrides in arms.items():
        results = []
        for seed in seeds:
            model, wall, _, info = fit(cfg | overrides | {"seed": seed}, d, quiet=True)
            m = metrics(evaluate(model, d, rows, bs))
            results.append({"seed": seed, "wall_s": wall,
                            "selected_epoch": info["selected_epoch"],
                            "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                            "per_class_f1": m["f1"].tolist(),
                            "per_class_recall": m["recall"].tolist(),
                            "support": m["support"].tolist(),
                            "history": info["history"]})
            log_experiment({"run": f"{run}_{arm}_s{seed}", "sha": sha, "dirty": dirty,
                            "config": f"{CONFIG}#{arm}", "split_sha": split_sha,
                            "seed": seed, "eval_split": EVAL_SPLIT,
                            "epochs": cfg["epochs"], "wall_s": wall,
                            "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                            "null_accuracy": null_acc})
            print(f"  {arm:<14} seed {seed}  epoch {info['selected_epoch']:>2}  "
                  f"macro-F1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}  "
                  f"({wall:.0f} s)")
        f1 = np.array([r["macro_f1"] for r in results])
        out[arm] = {"overrides": overrides, "runs": results,
                    "mean": float(f1.mean()), "std": float(f1.std(ddof=1))}

    print(f"\n{'arm':<16}{'macro-F1 (mean +/- std)':>26}{'seeds':>28}{'epochs':>14}")
    for arm, v in out.items():
        f1 = [r["macro_f1"] for r in v["runs"]]
        ep = [r["selected_epoch"] for r in v["runs"]]
        print(f"{arm:<16}{v['mean']:>17.4f} +/-{v['std']:.4f}"
              f"{'  '.join(f'{x:.4f}' for x in f1):>28}"
              f"{'  '.join(str(e) for e in ep):>14}")

    # The declared rule, applied mechanically rather than by eye.
    print(f"\npairwise, against the noise floor of {floor} declared in {CONFIG}:")
    ranked = 0
    for a, b in combinations(out, 2):
        diff = out[a]["mean"] - out[b]["mean"]
        if abs(diff) < floor:
            print(f"  {a} vs {b}: {diff:+.4f} -- INDISTINGUISHABLE, not ranked")
        else:
            ranked += 1
            hi, lo = (a, b) if diff > 0 else (b, a)
            print(f"  {a} vs {b}: {diff:+.4f} -- {hi} above {lo}")
    if ranked == 0:
        print("\nNo pair of arms separates by more than the declared floor. The result "
              "is that none of\nthe three imbalance treatments is measurably better "
              "than plain cross-entropy at this\nbudget on this split. That is the "
              "finding; it is not a failed experiment, and the arm\nwith the highest "
              "mean is not the winner.")

    # Per-class F1 on the low-support classes, where an imbalance treatment is
    # supposed to act. A treatment can be invisible in macro-F1 and still have
    # moved the tail, and the tail is what the method claims to fix.
    small = np.argsort([int(s) for s in out["baseline"]["runs"][0]["support"]])[:4]
    print(f"\nper-class F1 on the four smallest val classes, mean over seeds:")
    print(f"{'arm':<16}" + "".join(
        f"{d.classes[c]} (n={out['baseline']['runs'][0]['support'][c]})".rjust(26)
        for c in small))
    for arm, v in out.items():
        f1c = np.array([r["per_class_f1"] for r in v["runs"]]).mean(0)
        print(f"{arm:<16}" + "".join(f"{f1c[c]:>26.4f}" for c in small))

    dest = REPO / "runs" / f"cls_ablation_{run}"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "ablation.json").write_text(json.dumps({
        "run": run, "git_sha": sha, "git_dirty": dirty, "config_path": CONFIG,
        "config": cfg, "split_sha256": split_sha, "eval_split": EVAL_SPLIT,
        "noise_floor_macro_f1": floor, "classes": d.classes,
        "null_accuracy": null_acc, "arms": out,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {dest.relative_to(REPO).as_posix()}/ and docs/EXPERIMENTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
