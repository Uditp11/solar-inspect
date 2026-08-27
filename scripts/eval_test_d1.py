"""The single test-split evaluation of D1 classification.

    python scripts/eval_test_d1.py

**This script reads the test split. Nothing else in the project does, and after
this has run once, nothing does again.**

The rule it enforces is non-negotiable #4: the config is committed and pushed
*before* the held-out split is read, so the model cannot have been chosen in
response to its test number. That rule fails invisibly -- a config edited after
the fact leaves no trace in the artifact -- so it is checked here rather than
remembered. `guard()` refuses to run unless `configs/cls_final.yaml` is tracked,
unmodified in the working tree, and contained in `origin/main`, and it prints the
SHA of the commit that last touched it so that SHA can be reported beside the
number.

One pass, several views of it. The per-image logits for val and test are written
to the run directory, and the base-rate correction, the calibration fit and the
leaky/clean subgroup breakdown are all computed from those saved arrays. That is
what keeps "evaluated once" true while still answering more than one question
about the result.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.data import load_d1                    # noqa: E402
from solar_inspect.classification.leakage import class_matched, report   # noqa: E402
from solar_inspect.classification.train import (NULL_CLASS, confusion,   # noqa: E402
                                                fit, git_state,
                                                log_experiment, metrics,
                                                save_confusion_figure)

CONFIG = "configs/cls_final.yaml"
NEAR_DUP = REPO / "data" / "d1_near_dup.json"
THR = 0.98


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def guard() -> str:
    """Refuse to read test unless CONFIG is committed and pushed. Returns its SHA."""
    if git("ls-files", "--error-unmatch", CONFIG).returncode != 0:
        raise SystemExit(f"{CONFIG} is not tracked. Commit and push it first.")
    if git("status", "--porcelain", "--", CONFIG).stdout.strip():
        raise SystemExit(f"{CONFIG} is modified in the working tree. The committed "
                         "version is the one this evaluation is allowed to use.")
    sha = git("log", "-1", "--format=%H", "--", CONFIG).stdout.strip()
    if not sha:
        raise SystemExit(f"no commit touches {CONFIG}")
    if git("merge-base", "--is-ancestor", sha, "origin/main").returncode != 0:
        raise SystemExit(
            f"{CONFIG} was last changed in {sha[:7]}, which is not in origin/main. "
            "Push before evaluating -- a config that exists only locally is not a "
            "pre-declaration.")
    print(f"guard: {CONFIG} committed in {sha[:7]}, in origin/main, tree clean")
    return sha


@torch.no_grad()
def logits_for(model, d, rows: torch.Tensor, bs: int) -> np.ndarray:
    model.eval()
    return torch.cat([model(d.batch(rows[i:i + bs]))
                      for i in range(0, len(rows), bs)]).float().cpu().numpy()


def main() -> int:
    config_sha = guard()
    sha, dirty = git_state()
    cfg = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
    split_sha = json.loads((REPO / "configs" / "d1_split.json")
                           .read_text(encoding="utf-8"))["sha256"]

    d = load_d1()
    model, wall, _, info = fit(cfg, d)                  # selection is on val, as always
    bs = cfg["batch_size"]

    # Both splits scored from the one trained model, in one place, and saved. The
    # val logits are needed for the temperature fit; refitting a model later to
    # get them is how a "single evaluation" turns into several.
    out = {s: logits_for(model, d, d.index[s], bs) for s in ("val", "test")}
    y = {s: d.labels[d.index[s]].cpu().numpy() for s in ("val", "test")}
    pred = {s: out[s].argmax(1) for s in ("val", "test")}

    cm = confusion(y["test"], pred["test"], len(d.classes))
    m = metrics(cm)
    from sklearn.metrics import f1_score
    assert abs(f1_score(y["test"], pred["test"], average="macro", zero_division=0)
               - m["macro_f1"]) < 1e-9, "hand-rolled macro-F1 disagrees with sklearn"
    null_acc = float(cm.sum(1)[d.classes.index(NULL_CLASS)] / cm.sum())

    print(f"\n=== TEST, once, config committed in {config_sha[:7]} ===")
    print(f"macro-F1 {m['macro_f1']:.4f}   accuracy {m['accuracy']:.4f}   "
          f"null model (always {NULL_CLASS}) {null_acc:.4f}")
    print(f"val macro-F1 for the same model: "
          f"{metrics(confusion(y['val'], pred['val'], len(d.classes)))['macro_f1']:.4f} "
          f"(selected epoch {info['selected_epoch']})")
    print(f"\n{'class':<16}{'support':>9}{'recall':>9}{'precision':>11}{'F1':>9}")
    for i, c in enumerate(d.classes):
        print(f"{c:<16}{m['support'][i]:>9}{m['recall'][i]:>9.4f}"
              f"{m['precision'][i]:>11.4f}{m['f1'][i]:>9.4f}")

    print(f"\nconfusion matrix, rows true / columns predicted:")
    print(" " * 17 + "".join(f"{c[:6]:>8}" for c in d.classes))
    for i, c in enumerate(d.classes):
        print(f"{c:<17}" + "".join(f"{v:>8}" for v in cm[i]))

    # The leaky/clean subgroup, from THIS run's per-image predictions. Running it
    # as its own evaluation would spend the one test pass on a diagnostic; taking
    # it from the same predictions costs nothing and leaves the guarantee intact.
    nd = json.loads(NEAR_DUP.read_text(encoding="utf-8"))
    assert nd["paths"] == d.paths, "d1_near_dup.json was built against a different order"
    best = np.array(nd["best_train_cosine"])[d.index["test"].cpu().numpy()]
    sub = class_matched(pred["test"] == y["test"], y["test"], best >= THR, d.classes)
    print(f"\n=== test subgroup: near-duplicate leakage, cosine >= {THR} (ADR 0004) ===")
    print(report(sub))

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = REPO / "runs" / f"cls_final_test_{run}"
    dest.mkdir(parents=True, exist_ok=True)
    save_confusion_figure(cm, d.classes, dest / "confusion_test.png",
                          split="test", title="D1 ResNet-18 (cls_final)")
    np.savez_compressed(dest / "logits.npz", **{f"{s}_logits": out[s] for s in out},
                        **{f"{s}_labels": y[s] for s in y},
                        test_best_train_cosine=best)
    (dest / "manifest.json").write_text(json.dumps({
        "run": run, "git_sha": sha, "git_dirty": dirty, "config_path": CONFIG,
        "config_commit_sha": config_sha, "config": cfg, "eval_split": "test",
        "split_sha256": split_sha, "seed": cfg["seed"],
        "selected_epoch": info["selected_epoch"], "history": info["history"],
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "params": sum(p.numel() for p in model.parameters()), "wall_s": wall,
        "macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "null_accuracy": null_acc,
        "per_class": {c: {"support": int(m["support"][i]), "recall": float(m["recall"][i]),
                          "precision": float(m["precision"][i]), "f1": float(m["f1"][i])}
                      for i, c in enumerate(d.classes)},
        "confusion_matrix": cm.tolist(), "classes": d.classes,
        "leakage_subgroup_test": sub,
    }, indent=2, default=float) + "\n", encoding="utf-8", newline="\n")

    torch.save(model.state_dict(), dest / "teacher.pt")     # gitignored; KD needs it
    log_experiment({"run": run, "sha": sha, "dirty": dirty, "config": CONFIG,
                    "split_sha": split_sha, "seed": cfg["seed"], "eval_split": "test",
                    "epochs": cfg["epochs"], "wall_s": wall, "macro_f1": m["macro_f1"],
                    "accuracy": m["accuracy"], "null_accuracy": null_acc})
    print(f"\nwrote {dest.relative_to(REPO).as_posix()}/ and docs/EXPERIMENTS.md")
    print(f"config commit: {config_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
