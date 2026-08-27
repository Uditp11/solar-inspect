"""Distillation: ResNet-18 teacher -> TinyCNN student. Three rows, evaluated on VAL.

    python scripts/distill_d1.py

The teacher is the exact checkpoint behind the reported test number -- the seed-0
ResNet-18 of configs/cls_final.yaml, saved by scripts/eval_test_d1.py. Its logits
over the train split are computed once here and reused for all three seeds, since
they do not change.

**Three rows, and the middle one is what makes the other two mean anything:**
teacher, student trained from scratch at the same budget, student distilled. A
distilled student's macro-F1 on its own says nothing without the same student on
hard labels; and the gap between them is judged against the 0.02 noise floor
declared in configs/cls_ablation.yaml, not against zero.

**Latency is measured on CPU, warmed up, at a stated batch size -- and it measures
the framework more than the model.** At 40x24 a forward pass is a few hundred
microseconds of arithmetic wrapped in Python attribute lookups and PyTorch
dispatch, so the number below is dominated by per-op overhead rather than by
FLOPs. It is reported because the parameter ratio alone would imply a speedup that
does not exist, and the honest version of "4x smaller" has to say what that buys.

Task 6's confusion matrix predicts where KD should act if it acts anywhere:
Cell <-> Cell-Multi was the largest off-diagonal pair, and confusability between
near-identical classes is exactly the structure soft targets are supposed to
carry. That pair is reported separately. If KD does not help there, it is unlikely
to be helping anywhere, and that is a result too.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.data import load_d1                    # noqa: E402
from solar_inspect.classification.train import (EVAL_SPLIT, NULL_CLASS,  # noqa: E402
                                                build_model, confusion,
                                                evaluate, fit, git_state,
                                                log_experiment, metrics)

CONFIG = "configs/cls_kd.yaml"
LATENCY_BATCH = 64
LATENCY_REPS = 200
WARMUP = 50


@torch.no_grad()
def all_logits(model, d, rows: torch.Tensor, bs: int) -> torch.Tensor:
    model.eval()
    return torch.cat([model(d.batch(rows[i:i + bs])) for i in range(0, len(rows), bs)])


def cpu_latency(model: torch.nn.Module, batch: int) -> tuple[float, float]:
    """Warmed-up median ms per batch on CPU, and per image. One thread, stated."""
    model = model.cpu().eval()
    torch.set_num_threads(1)
    x = torch.randn(batch, 1, 40, 24)
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        t = []
        for _ in range(LATENCY_REPS):
            t0 = time.perf_counter()
            model(x)
            t.append((time.perf_counter() - t0) * 1000.0)
    med = float(np.median(t))
    return med, med / batch


def main() -> int:
    sha, dirty = git_state()
    cfg = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
    split_sha = json.loads((REPO / "configs" / "d1_split.json")
                           .read_text(encoding="utf-8"))["sha256"]
    floor, seeds, bs = float(cfg["noise_floor_macro_f1"]), cfg["seeds"], cfg["batch_size"]

    d = load_d1()
    k = len(d.classes)
    rows = d.index[EVAL_SPLIT]
    y_val = d.labels[rows].cpu().numpy()
    null_acc = float((d.labels[rows] == d.classes.index(NULL_CLASS)).float().mean())

    # ---- the teacher ----------------------------------------------------------
    # runs/ is gitignored, so the pinned path records which teacher was actually
    # used and the fallback lets a clone that has re-run eval_test_d1.py get a
    # teacher at all. If they differ it says so rather than substituting silently.
    tr = REPO / cfg["teacher_run"]
    if not tr.exists():
        found = sorted((REPO / "runs").glob("cls_final_test_*"))
        if not found:
            raise SystemExit("no teacher: run scripts/eval_test_d1.py first")
        print(f"NOTE: {cfg['teacher_run']} is absent (runs/ is gitignored). "
              f"Using {found[-1].name} instead -- this is a different teacher from "
              "the one the committed numbers were measured against.")
        tr = found[-1]
    tman = json.loads((tr / "manifest.json").read_text(encoding="utf-8"))
    teacher = build_model(tman["config"], k).to(d.images.device)
    teacher.load_state_dict(torch.load(tr / "teacher.pt", map_location=d.images.device))
    print(f"teacher: {cfg['teacher_run']}, config commit "
          f"{tman['config_commit_sha'][:7]}, {tman['params']:,} params")

    # Teacher logits over EVERY row, indexed globally, so fit() can slice them by
    # the same row indices it slices images with. Only the train rows are used by
    # the loss; filling the rest costs one extra forward pass over 20,000 crops.
    t_logits = torch.zeros(len(d.paths), k, device=d.images.device)
    for s in ("train", "val", "test"):
        t_logits[d.index[s]] = all_logits(teacher, d, d.index[s], bs)
    t_cm = confusion(y_val, t_logits[rows].argmax(1).cpu().numpy(), k)
    t_m = metrics(t_cm)
    print(f"teacher {EVAL_SPLIT} macro-F1 {t_m['macro_f1']:.4f} "
          f"(the reported test number is 0.6956 and is not this)")

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out: dict[str, dict] = {}
    for arm, over in cfg["arms"].items():
        res = []
        for seed in seeds:
            model, wall, _, info = fit(cfg | over | {"seed": seed}, d, quiet=True,
                                       teacher_logits=t_logits)
            cm = evaluate(model, d, rows, bs)
            m = metrics(cm)
            res.append({"seed": seed, "wall_s": wall, "macro_f1": m["macro_f1"],
                        "accuracy": m["accuracy"],
                        "selected_epoch": info["selected_epoch"],
                        "confusion": cm.tolist(),
                        "per_class_f1": m["f1"].tolist()})
            log_experiment({"run": f"{run}_{arm}_s{seed}", "sha": sha, "dirty": dirty,
                            "config": f"{CONFIG}#{arm}", "split_sha": split_sha,
                            "seed": seed, "eval_split": EVAL_SPLIT,
                            "epochs": cfg["epochs"], "wall_s": wall,
                            "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                            "null_accuracy": null_acc})
            print(f"  {arm:<22} seed {seed}  epoch {info['selected_epoch']:>2}  "
                  f"macro-F1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}")
        f1 = np.array([r["macro_f1"] for r in res])
        out[arm] = {"overrides": over, "runs": res,
                    "mean": float(f1.mean()), "std": float(f1.std(ddof=1)),
                    "params": sum(p.numel() for p in model.parameters())}
        out[arm]["latency_ms_batch"], out[arm]["latency_ms_image"] = \
            cpu_latency(model, LATENCY_BATCH)

    t_lat_b, t_lat_i = cpu_latency(teacher, LATENCY_BATCH)
    out["teacher"] = {"runs": [{"seed": tman["seed"], "macro_f1": t_m["macro_f1"],
                                "accuracy": t_m["accuracy"],
                                "confusion": t_cm.tolist(),
                                "per_class_f1": t_m["f1"].tolist()}],
                      "mean": t_m["macro_f1"], "std": float("nan"),
                      "params": tman["params"],
                      "latency_ms_batch": t_lat_b, "latency_ms_image": t_lat_i}

    order = ["teacher", "student-from-scratch", "student-distilled"]
    print(f"\n{'':<24}{'val macro-F1':>22}{'params':>13}"
          f"{f'CPU ms / batch of {LATENCY_BATCH}':>28}{'ms / image':>13}")
    for name in order:
        v = out[name]
        f1 = (f"{v['mean']:.4f} (1 seed)" if name == "teacher"
              else f"{v['mean']:.4f} +/-{v['std']:.4f}")
        print(f"{name:<24}{f1:>22}{v['params']:>13,}"
              f"{v['latency_ms_batch']:>28.2f}{v['latency_ms_image']:>13.4f}")

    a, b = out["student-from-scratch"], out["student-distilled"]
    gap = b["mean"] - a["mean"]
    print(f"\nKD gain, distilled minus from-scratch: {gap:+.4f} macro-F1")
    print(f"declared noise floor: {floor}")
    print("  -> " + ("INDISTINGUISHABLE. Distillation did not measurably help this "
                     "student at this\n     budget with alpha=0.5, T=4. That is the "
                     "result; it is not a bug to be tuned away."
                     if abs(gap) < floor else
                     f"distillation {'helps' if gap > 0 else 'HURTS'} by more than the "
                     "floor."))
    print(f"the teacher-student gap it is trying to close: "
          f"{out['teacher']['mean'] - a['mean']:+.4f}")

    # Where soft targets are supposed to act, if they act anywhere.
    ci, cmi = d.classes.index("Cell"), d.classes.index("Cell-Multi")
    print(f"\nCell <-> Cell-Multi, the largest off-diagonal pair, mean over seeds:")
    print(f"{'':<24}{'Cell->Cell-Multi':>20}{'Cell-Multi->Cell':>20}"
          f"{'Cell F1':>10}{'Cell-Multi F1':>16}")
    for name in order:
        cms = np.array([r["confusion"] for r in out[name]["runs"]], dtype=float).mean(0)
        f1c = np.array([r["per_class_f1"] for r in out[name]["runs"]]).mean(0)
        print(f"{name:<24}{cms[ci, cmi]:>20.1f}{cms[cmi, ci]:>20.1f}"
              f"{f1c[ci]:>10.4f}{f1c[cmi]:>16.4f}")

    dest = REPO / "runs" / f"cls_kd_{run}"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "kd.json").write_text(json.dumps({
        "run": run, "git_sha": sha, "git_dirty": dirty, "config_path": CONFIG,
        "config": cfg, "split_sha256": split_sha, "eval_split": EVAL_SPLIT,
        "teacher_run": cfg["teacher_run"],
        "latency": {"batch": LATENCY_BATCH, "reps": LATENCY_REPS, "warmup": WARMUP,
                    "threads": 1, "device": "cpu"},
        "noise_floor_macro_f1": floor, "classes": d.classes, "arms": out,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {dest.relative_to(REPO).as_posix()}/ and docs/EXPERIMENTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
