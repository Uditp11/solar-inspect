"""The detection tuning pass: twelve training runs, each scored on val, resumable.

    python scripts/tune_det.py            # the pass; re-run the same command to resume
    python scripts/tune_det.py --smoke    # every run at epochs=1, into runs/det_tuning_smoke/

Arms, seeds, score and the selection rule are declared in `configs/det_tuning.yaml`
and docs/EXPERIMENTS.md, committed before the first arm ran. Runs go in seed order,
seed 0 of all four arms first, so a pass cut off early still has one complete seed
per arm. Each run is `train_det.py` with the arm's overrides, then `eval_val_det.py`
on its `best.pt`, each as its own process, so one run failing (an exception, out of
memory) is logged and the pass moves on to the next.

- Refuses to start on a dirty tree, records HEAD once, and stops if HEAD moves or the
  tree changes before any later run. A summary written at another commit is refused.
- Resumable: a run whose training manifest and val eval.json both exist, as recorded
  in the summary, is skipped; one that trained but was not evaluated is evaluated.
- After every run, rewrites `summary.csv` and `summary.json` in the output directory,
  with each subprocess's output in `logs/`.
- At the end, prints per-arm mean and standard deviation, the noise floor and the arm
  the declared rule selects. It prints; it changes no config.

Reads val only: both child scripts refuse if val resolves to the test tree, and this
script reads no data itself. Training runs are written to `runs/det_<run>/` and
evaluations to `runs/det_val_<run>/` like any other run; the summary names them.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.train import git_state           # noqa: E402
from solar_inspect.detection.tuning import noise_floor, select_arm  # noqa: E402

CONFIG = "configs/det_tuning.yaml"
BASE_CONFIG = "configs/det_yolo.yaml"
# Seconds per epoch including the per-epoch val, for the estimate printed at the start:
# yolov8n from the full run det_20261022T073325Z (7.4 s); yolov8s scaled from it by the
# second-epoch ratio of the two smoke runs (3.6 s / 3.2 s), so a rougher figure.
SEC_PER_EPOCH = {"yolov8n.pt": 7.4, "yolov8s.pt": 8.3}
OVERHEAD_S = 90    # start-up, final validation in training, and the val evaluation
FIELDS = ["arm", "seed", "status", "train_run", "eval_run", "epochs_run", "best_epoch",
          "wall_s", "peak_vram_reserved_mib", "score", "ap50_101_point",
          "count_mae_at_0.25", "error"]


def run_script(script: str, args: list[str], log: Path) -> tuple[int, str | None]:
    """Runs scripts/<script> in its own process; returns (exit code, the run directory
    it reported with its final 'wrote <dir>/' line)."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with log.open("wb") as f:
        rc = subprocess.run([sys.executable, str(REPO / "scripts" / script), *args],
                            cwd=REPO, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    wrote = re.findall(r"^wrote (\S+)/\s*$", log.read_text(encoding="utf-8", errors="replace"),
                       flags=re.M)
    return rc, (wrote[-1] if wrote else None)


def write_summary(out: Path, state: dict) -> None:
    (out / "summary.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8",
                                      newline="\n")
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(state["runs"])


def done(r: dict) -> bool:
    return (r["status"] == "done" and (REPO / r["train_run"] / "manifest.json").exists()
            and (REPO / r["eval_run"] / "eval.json").exists())


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="epochs=1, separate summary")
    a = ap.parse_args(argv)

    sha, dirty = git_state()
    if dirty:
        raise SystemExit("the working tree is dirty; commit first, every run records HEAD")
    cfg = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
    base = yaml.safe_load((REPO / BASE_CONFIG).read_text(encoding="utf-8"))
    arms, seeds, baseline = cfg["arms"], cfg["seeds"], cfg["baseline"]
    epochs = 1 if a.smoke else base["epochs"]
    out = REPO / "runs" / ("det_tuning_smoke" if a.smoke else "det_tuning")
    (out / "logs").mkdir(parents=True, exist_ok=True)

    state = {"git_sha": sha, "config_path": CONFIG, "config": cfg, "smoke": a.smoke,
             "epochs": epochs, "runs": []}
    if (out / "summary.json").exists():
        prev = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        if prev["git_sha"] != sha:
            raise SystemExit(f"{out.relative_to(REPO)}/summary.json was written at "
                             f"{prev['git_sha'][:7]}, HEAD is {sha[:7]}; one pass, one commit")
        state["runs"] = prev["runs"]
    by_key = {(r["arm"], r["seed"]): r for r in state["runs"]}
    order = [(arm, s) for s in seeds for arm in arms]          # seed-major
    for key in order:
        if key not in by_key:
            by_key[key] = {"arm": key[0], "seed": key[1], "status": "pending"}
    state["runs"] = [by_key[k] for k in order]
    write_summary(out, state)

    todo = [r for r in state["runs"] if not done(r)]
    est = sum(epochs * SEC_PER_EPOCH[arms[r["arm"]]["model"]] + OVERHEAD_S for r in todo)
    print(f"tuning pass at {sha[:7]}{' (smoke, epochs=1)' if a.smoke else ''}: "
          f"{len(order)} runs, {len(order) - len(todo)} already complete, {len(todo)} to go; "
          f"estimated {est / 3600:.1f} h ({est / 60:.0f} min), early stopping ignored")
    print(f"summary: {(out / 'summary.csv').relative_to(REPO).as_posix()}\n", flush=True)

    t_pass = time.time()
    for r in todo:
        now_sha, now_dirty = git_state()
        if now_sha != sha or now_dirty:
            write_summary(out, state)
            raise SystemExit(f"HEAD moved or the tree changed ({now_sha[:7]}, dirty="
                             f"{now_dirty}); stopped before {r['arm']} seed {r['seed']}")
        over = {**cfg.get("common", {}), **arms[r["arm"]], "seed": r["seed"]}
        if a.smoke:
            over["epochs"] = 1
        args = [f"{k}={v}" for k, v in over.items()]
        tag = f"{r['arm']}_s{r['seed']}"
        print(f"[{time.strftime('%H:%M:%S')}] {r['arm']} seed {r['seed']}: "
              + " ".join(args), flush=True)
        t0 = time.time()
        r.pop("error", None)
        try:
            if not (r.get("train_run") and (REPO / r["train_run"] / "manifest.json").exists()):
                rc, run_dir = run_script("train_det.py", args, out / "logs" / f"{tag}_train.log")
                if rc != 0 or run_dir is None:
                    raise RuntimeError(f"train_det.py exited {rc}; see logs/{tag}_train.log")
                r["train_run"] = run_dir
            m = json.loads((REPO / r["train_run"] / "manifest.json").read_text(encoding="utf-8"))
            r.update(status="trained", epochs_run=m["epochs_run"], best_epoch=m["best_epoch"],
                     peak_vram_reserved_mib=round(m["peak_vram_reserved_mib"]))
            rc, eval_dir = run_script("eval_val_det.py",
                                      [f"{r['train_run']}/weights/best.pt", *args],
                                      out / "logs" / f"{tag}_eval.log")
            if rc != 0 or eval_dir is None:
                raise RuntimeError(f"eval_val_det.py exited {rc}; see logs/{tag}_eval.log")
            e = json.loads((REPO / eval_dir / "eval.json").read_text(encoding="utf-8"))
            r.update(status="done", eval_run=eval_dir, score=e[cfg["score"]],
                     ap50_101_point=e["ap50_101_point"],
                     **{"count_mae_at_0.25": e["at_conf"]["0.25"]["count_mae"]})
        except Exception as exc:                 # one failed run never ends the pass
            r.update(status="failed", error=str(exc))
        r["wall_s"] = round(r.get("wall_s", 0) + time.time() - t0)
        write_summary(out, state)
        print(f"    {r['status']}, {r['wall_s']} s"
              + (f", score {r['score']:.6f}" if r["status"] == "done" else f": {r.get('error')}"),
              flush=True)

    # The declared rule, printed only
    res = {}
    for arm in arms:
        sc = np.array([r["score"] for r in state["runs"] if r["arm"] == arm and done(r)])
        res[arm] = {"n": len(sc), "scores": sc.tolist(),
                    "mean": float(sc.mean()) if len(sc) else None,
                    "std": float(sc.std(ddof=1)) if len(sc) > 1 else None}
    print(f"\n{'arm':<16}{'n':>3}{'score mean':>12}{'std':>9}   seeds")
    for arm, v in res.items():
        print(f"{arm:<16}{v['n']:>3}"
              + (f"{v['mean']:>12.6f}" if v["mean"] is not None else f"{'-':>12}")
              + (f"{v['std']:>9.6f}" if v["std"] is not None else f"{'-':>9}")
              + "   " + " ".join(f"{s:.6f}" for s in v["scores"]))
    complete = all(v["n"] == len(seeds) for v in res.values())
    state["result"] = {"arms": res, "complete": complete}
    if complete:
        floor = noise_floor([v["std"] for v in res.values()])
        means = {arm: v["mean"] for arm, v in res.items()}
        chosen = select_arm(means, arms, baseline, floor)
        print(f"\nnoise floor {floor:.3f} (largest std {max(v['std'] for v in res.values()):.6f})")
        for x, y in combinations(arms, 2):
            d = means[x] - means[y]
            call = "indistinguishable" if abs(round(d, 9)) < floor else \
                f"{x if d > 0 else y} above"
            print(f"  {x} - {y}: {d:+.6f}  {call}")
        seed0 = next(r for r in state["runs"] if r["arm"] == chosen and r["seed"] == 0)
        print(f"\nselected by the declared rule: {chosen} (baseline {baseline}); "
              f"model taken forward: its seed 0 run, {seed0['train_run']}")
        state["result"].update(noise_floor=floor, selected=chosen,
                               selected_seed0_run=seed0["train_run"])
    else:
        print("\nnot every arm has all its seeds: the selection rule is not applied "
              "(declared), the per-arm results above stand as they are")
    write_summary(out, state)
    print(f"\nwall time this invocation: {(time.time() - t_pass) / 60:.1f} min")
    st = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                        capture_output=True, text=True).stdout
    print(f"git status --porcelain: {st.strip() or '(empty)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
