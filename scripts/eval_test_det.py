"""The single test-split evaluation of D2 detection.

    python scripts/eval_test_det.py runs/det_<run>/weights/best.pt

**This script reads the test split. Nothing else in the project does, and after
this has run once, nothing does again.**

The rule it enforces is the evaluate-once rule: the configs are committed and
pushed *before* the held-out split is read, so the model cannot have been chosen in
response to its test number. That rule fails invisibly, so it is checked here
rather than remembered. `guard()` refuses to run unless `configs/det_yolo.yaml` and
`configs/d2.yaml` are tracked, unmodified in the working tree, and contained in
`origin/main`, and it prints the SHA of the commit that last touched each so they
can be reported beside the number. It also refuses if a completed test evaluation
already exists under `runs/`, and unless the checkpoint's SHA-256 equals the
`checkpoint_sha256` committed in the config (`check_checkpoint`): the checkpoint is not
in git, so the hash is what pins the model chosen on val.

One pass, several views of it: Ultralytics' validator runs once on test, and every
metric is computed from the predictions it saved (`solar_inspect.detection.evaluate`),
which are kept in the run directory so later questions about the result need no
second pass.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.train import git_state   # noqa: E402
from solar_inspect.detection.data import check_checkpoint   # noqa: E402
from solar_inspect.detection.evaluate import evaluate       # noqa: E402

CONFIG = "configs/det_yolo.yaml"
GUARDED = (CONFIG, "configs/d2.yaml")


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def guard() -> dict[str, str]:
    """Refuse to read test unless every GUARDED file is committed and pushed, and no
    test evaluation has completed before. Returns {path: SHA of its last commit}."""
    done = sorted(REPO.glob("runs/det_test_*/eval.json"))
    if done:
        raise SystemExit(f"a test evaluation already exists: {done[0].parent.name}. "
                         "The test split is evaluated once.")
    shas = {}
    for f in GUARDED:
        if git("ls-files", "--error-unmatch", f).returncode != 0:
            raise SystemExit(f"{f} is not tracked. Commit and push it first.")
        if git("status", "--porcelain", "--", f).stdout.strip():
            raise SystemExit(f"{f} is modified in the working tree. The committed "
                             "version is the one this evaluation is allowed to use.")
        sha = git("log", "-1", "--format=%H", "--", f).stdout.strip()
        if not sha:
            raise SystemExit(f"no commit touches {f}")
        if git("merge-base", "--is-ancestor", sha, "origin/main").returncode != 0:
            raise SystemExit(
                f"{f} was last changed in {sha[:7]}, which is not in origin/main. "
                "Push before evaluating -- a config that exists only locally is not a "
                "pre-declaration.")
        print(f"guard: {f} committed in {sha[:7]}, in origin/main, tree clean")
        shas[f] = sha
    return shas


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit(__doc__)
    weights = Path(argv[0]).resolve()
    config_shas = guard()
    sha, dirty = git_state()
    cfg = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
    weights_sha = check_checkpoint(weights, cfg.get("checkpoint_sha256"))
    print(f"guard: {weights.name} SHA-256 {weights_sha[:12]} = the config's checkpoint_sha256")
    train_manifest = weights.parent.parent / "manifest.json"
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    out = evaluate(weights, "test", cfg, REPO / "runs", f"det_test_{run}")
    out.update({"run": run, "git_sha": sha, "git_dirty": dirty, "config_path": CONFIG,
                "config": cfg, "config_commit_shas": config_shas,
                "weights": weights.relative_to(REPO).as_posix(), "weights_sha256": weights_sha,
                "train_manifest": (json.loads(train_manifest.read_text(encoding="utf-8"))
                                   if train_manifest.exists() else None)})
    dest = REPO / out["save_dir"]
    (dest / "eval.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8",
                                    newline="\n")

    u, tm = out["ultralytics"], out["torchmetrics"]
    print(f"\n=== TEST, once ({out['images']} images, {out['boxes']} boxes), configs "
          + ", ".join(f"{f} {s[:7]}" for f, s in config_shas.items()) + " ===")
    print(f"AP@0.5     Ultralytics {u['map50']:.4f} | from scratch all-point "
          f"{out['ap50_all_point']:.4f}, 101-point {out['ap50_101_point']:.4f} | "
          f"torchmetrics {tm['map_50']:.4f}")
    print(f"AP@0.5:0.95 Ultralytics {u['map50_95']:.4f} | from scratch 101-point "
          f"{out['ap50_95_101_point']:.4f} | torchmetrics {tm['map_50_95']:.4f}")
    for c, r in out["at_conf"].items():
        print(f"conf {c}: precision {r['precision']:.4f} recall {r['recall']:.4f} "
              f"count MAE {r['count_mae']:.2f} signed mean {r['count_signed_mean']:+.2f}"
              + ("  (operating)" if float(c) == out.get("conf_operating") else ""))
    print(f"\nwrote {out['save_dir']}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
