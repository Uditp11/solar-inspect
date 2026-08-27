"""Baseline D1 classifier: train, evaluate on val, log the run.

    python src/solar_inspect/classification/train.py configs/cls_baseline.yaml

Evaluation is on **val**, hardcoded. The test split exists in the data object and
is never read here. It is evaluated once, in Task 7, after the final config is
committed -- a validation loop pointed at test burns that guarantee with no
visible symptom, so the guarantee is enforced by there being no switch to flip.

Metrics are computed from a confusion matrix built with np.bincount and
cross-checked against sklearn once per run. Accuracy is reported only next to the
null model that predicts No-Anomaly always, because on D1 that null model scores
50% and accuracy on its own says nothing.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))   # run as a file; the package is not pip-installed

from solar_inspect.classification.data import load_d1          # noqa: E402
from solar_inspect.classification.model import TinyCNN         # noqa: E402

EVAL_SPLIT = "val"          # never "test" -- see the module docstring
NULL_CLASS = "No-Anomaly"


def confusion(true: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    """Rows are true classes, columns predicted. Hand-checkable on a toy input."""
    return np.bincount(true * k + pred, minlength=k * k).reshape(k, k)


def metrics(cm: np.ndarray) -> dict:
    """Per-class recall and precision from the confusion matrix, then macro-F1."""
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1).astype(float), cm.sum(0).astype(float)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {
        "macro_f1": float(f1.mean()),
        "accuracy": float(tp.sum() / cm.sum()),
        "recall": recall, "precision": precision, "f1": f1,
        "support": support.astype(int),
    }


@torch.no_grad()
def evaluate(model: nn.Module, d, rows: torch.Tensor, batch: int) -> np.ndarray:
    model.eval()
    preds = torch.cat([model(d.batch(rows[i:i + batch])).argmax(1)
                       for i in range(0, len(rows), batch)])
    return confusion(d.labels[rows].cpu().numpy(), preds.cpu().numpy(), len(d.classes))


def git_state() -> tuple[str, bool]:
    """Must be called BEFORE the run writes anything.

    The run appends to docs/EXPERIMENTS.md and creates runs/, so asking git
    afterwards reports a tree the run itself dirtied and every run would record
    dirty=yes regardless of what it was launched from.
    """
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())
    return sha, dirty


def save_confusion_figure(cm: np.ndarray, classes: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Row-normalised: raw counts are unreadable when one row is 1,500 and another
    # is 26. The counts are printed beside every recall in the log instead.
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(norm, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)),
                  [f"{c}  (n={n})" for c, n in zip(classes, cm.sum(1))], fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (support)")
    ax.set_title(f"D1 baseline, {EVAL_SPLIT} split - row-normalised confusion")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def log_experiment(row: dict) -> None:
    path = REPO / "docs" / "EXPERIMENTS.md"
    header = (
        "# Experiments\n\n"
        "Every training run, in order. A run with `dirty=yes` was made against a\n"
        "working tree that did not match its commit, so its numbers are not\n"
        "reproducible from that SHA -- treat them as indicative only.\n\n"
        "| run | git SHA | dirty | split | config | seed | eval split | epochs | wall | macro-F1 | acc | null acc |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    if not path.exists():
        path.write_text(header, encoding="utf-8", newline="\n")
    line = (f"| {row['run']} | `{row['sha'][:7]}` | {'yes' if row['dirty'] else 'no'} "
            f"| `{row['split_sha'][:8]}` "
            f"| `{row['config']}` | {row['seed']} | {row['eval_split']} | {row['epochs']} "
            f"| {row['wall_s']:.0f} s | **{row['macro_f1']:.4f}** | {row['accuracy']:.4f} "
            f"| {row['null_accuracy']:.4f} |\n")
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line)


def class_weights(labels: torch.Tensor, k: int) -> torch.Tensor:
    """n / (k * n_c) -- inverse frequency, normalised to average 1.

    Averaging to 1 matters. It keeps the weighted loss on the same scale as
    unweighted CE, so the arms of the ablation see comparable gradient magnitudes
    at the single fixed LR they all share. A class absent from `labels` would
    divide by zero, so its count is clamped to 1; it cannot contribute anyway.
    """
    n_c = torch.bincount(labels, minlength=k).float().clamp_min(1.0)
    return len(labels) / (k * n_c)


def focal_loss(logits: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """-(1 - p_t)^gamma * log p_t, mean over the batch.

    gamma is the whole mechanism. At gamma=0 this is exactly cross-entropy; above
    it, an example's gradient is scaled by (1 - p_t)^gamma, which is near zero for
    anything the model already gets right with confidence. It reweights by
    *difficulty*, which is a different axis from class frequency -- that is why it
    is a separate arm here and not a variant of class weighting.
    """
    logp = torch.log_softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
    return (-((1.0 - logp.exp()) ** gamma) * logp).mean()


def fit(cfg: dict, d, train_rows: torch.Tensor | None = None,
        quiet: bool = False) -> tuple[nn.Module, float, list[float], dict]:
    """Train one model. The defaults are exactly configs/cls_baseline.yaml.

    Extracted from main() so that a diagnostic can reproduce *this* model rather
    than a re-typed copy of this loop that has quietly drifted from it.
    `train_rows` overrides the train split, which is how the leakage control
    retrains with a subset of train removed.

    Four optional keys drive Task 7's ablation, and **every one defaults to the
    baseline's behaviour**, so a config written before they existed trains the
    identical model and ADR 0004's leakage-control numbers still reproduce here:

      loss:      ce (default) | weighted_ce | focal    -- gamma from cfg["gamma"]
      sampler:   shuffle (default) | balanced
      schedule:  constant (default) | cosine
      select:    last (default) | best_val

    `balanced` draws len(train_rows) rows per epoch with replacement at
    probability 1/n_c, so its epoch is the same length as every other arm's: the
    four arms differ in what they weight, not in how many steps they get.
    """
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = d.images.device
    k = len(d.classes)
    model = TinyCNN(k, tuple(cfg["widths"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    train_rows = d.index["train"] if train_rows is None else train_rows
    eval_rows = d.index[EVAL_SPLIT]
    bs = cfg["batch_size"]
    g = torch.Generator(device=device).manual_seed(cfg["seed"])

    kind = cfg.get("loss", "ce")
    y_train = d.labels[train_rows]
    if kind == "ce":
        loss_fn = nn.CrossEntropyLoss()
    elif kind == "weighted_ce":
        loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, k).to(device))
    elif kind == "focal":
        gamma = float(cfg["gamma"])
        loss_fn = lambda logits, y: focal_loss(logits, y, gamma)      # noqa: E731
    else:
        raise ValueError(f"unknown loss {kind!r}")

    sampler = cfg.get("sampler", "shuffle")
    if sampler == "balanced":
        # Sampling probability 1/n_c: an epoch becomes a draw from a uniform class
        # distribution, so Diode-Multi (175 rows) is seen as often as No-Anomaly
        # (10,000) and each of its 175 crops is therefore seen ~57x more often.
        w = (1.0 / torch.bincount(y_train, minlength=k).float().clamp_min(1.0))[y_train]
    elif sampler != "shuffle":
        raise ValueError(f"unknown sampler {sampler!r}")

    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
             if cfg.get("schedule", "constant") == "cosine" else None)
    select = cfg.get("select", "last")
    track = select == "best_val"

    if quiet:
        print(f"  fitting seed {cfg['seed']} on {len(train_rows)} train rows", flush=True)
    else:
        print(f"model: {sum(p.numel() for p in model.parameters()):,} params  "
              f"train={len(train_rows)} {EVAL_SPLIT}={len(eval_rows)}  "
              f"{len(train_rows) // bs + 1} steps/epoch  loss={kind} sampler={sampler} "
              f"schedule={cfg.get('schedule', 'constant')} select={select}")

    t0 = time.perf_counter()
    epoch_times, history = [], []
    best = (-1.0, 0, None)
    for epoch in range(1, cfg["epochs"] + 1):
        te = time.perf_counter()
        model.train()
        if sampler == "balanced":
            perm = train_rows[torch.multinomial(w, len(train_rows), replacement=True,
                                                generator=g)]
        else:
            perm = train_rows[torch.randperm(len(train_rows), generator=g, device=device)]
        total = 0.0
        for i in range(0, len(perm), bs):
            rows = perm[i:i + bs]
            loss = loss_fn(model(d.batch(rows)), d.labels[rows])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss) * len(rows)
        if sched is not None:
            sched.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - te)

        if quiet and not track:
            continue
        # Evaluation is deterministic and consumes no RNG, so running it in quiet
        # mode does not shift the training stream that later epochs draw from.
        m = metrics(evaluate(model, d, eval_rows, bs))
        history.append({"epoch": epoch, "train_loss": total / len(perm),
                        "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"],
                        "lr": opt.param_groups[0]["lr"]})
        if track and m["macro_f1"] > best[0]:
            best = (m["macro_f1"], epoch,
                    {n: t.detach().clone() for n, t in model.state_dict().items()})
        if not quiet:
            print(f"epoch {epoch:>3}  loss {total / len(perm):.4f}  "
                  f"{EVAL_SPLIT} macro-F1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}  "
                  f"({epoch_times[-1]:.2f} s)")

    info = {"select": select, "selected_epoch": cfg["epochs"], "history": history}
    if track:
        model.load_state_dict(best[2])
        info["selected_epoch"], info["best_val_macro_f1"] = best[1], best[0]
        if not quiet:
            print(f"selected epoch {best[1]} by {EVAL_SPLIT} macro-F1 {best[0]:.4f}")
    return model, time.perf_counter() - t0, epoch_times, info


def main(config_path: str) -> int:
    sha, dirty = git_state()        # before the run writes anything; see git_state
    cfg = yaml.safe_load((REPO / config_path).read_text(encoding="utf-8"))

    # Read before the run, so a row can never carry a split hash that a concurrent
    # re-split changed underneath it.
    split_sha = json.loads((REPO / "configs" / "d1_split.json")
                           .read_text(encoding="utf-8"))["sha256"]

    d = load_d1()
    model, wall, epoch_times, info = fit(cfg, d)
    eval_rows, bs = d.index[EVAL_SPLIT], cfg["batch_size"]

    cm = evaluate(model, d, eval_rows, bs)
    m = metrics(cm)

    # One cross-check against a library, so the from-scratch metrics are evidence
    # rather than an assertion. If these disagree, the hand-rolled one is wrong.
    from sklearn.metrics import f1_score
    y_true = np.repeat(np.arange(len(d.classes)), cm.sum(1))
    y_pred = np.concatenate([np.repeat(np.arange(len(d.classes)), r) for r in cm])
    assert abs(f1_score(y_true, y_pred, average="macro", zero_division=0)
               - m["macro_f1"]) < 1e-9, "hand-rolled macro-F1 disagrees with sklearn"

    null_acc = float(cm.sum(1)[d.classes.index(NULL_CLASS)] / cm.sum())
    print(f"\n{EVAL_SPLIT} macro-F1 {m['macro_f1']:.4f}")
    print(f"{'class':<16} {'support':>8} {'recall':>8} {'precision':>10} {'F1':>8}")
    for i, c in enumerate(d.classes):
        print(f"{c:<16} {m['support'][i]:>8} {m['recall'][i]:>8.4f} "
              f"{m['precision'][i]:>10.4f} {m['f1'][i]:>8.4f}")
    print(f"\naccuracy {m['accuracy']:.4f}  vs null model "
          f"(always {NULL_CLASS}) {null_acc:.4f}  "
          f"-- accuracy alone is meaningless at this base rate")
    print(f"wall {wall:.1f} s total, {np.median(epoch_times):.2f} s/epoch (median)")

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO / "runs" / f"cls_baseline_{run}"
    out.mkdir(parents=True, exist_ok=True)
    save_confusion_figure(cm, d.classes, out / f"confusion_{EVAL_SPLIT}.png")
    (out / "manifest.json").write_text(json.dumps({
        "run": run, "git_sha": sha, "git_dirty": dirty, "config_path": config_path,
        "config": cfg, "eval_split": EVAL_SPLIT, "seed": cfg["seed"],
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "split_sha256": split_sha,
        "norm": {"mean": d.mean, "std": d.std},
        "params": sum(p.numel() for p in model.parameters()),
        "wall_s": wall, "median_epoch_s": float(np.median(epoch_times)),
        "select": info["select"], "selected_epoch": info["selected_epoch"],
        "history": info["history"],
        "macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "null_accuracy": null_acc,
        "per_class": {c: {"support": int(m["support"][i]), "recall": float(m["recall"][i]),
                          "precision": float(m["precision"][i]), "f1": float(m["f1"][i])}
                      for i, c in enumerate(d.classes)},
        "confusion_matrix": cm.tolist(), "classes": d.classes,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")

    log_experiment({"run": run, "sha": sha, "dirty": dirty, "config": config_path,
                    "split_sha": split_sha,
                    "seed": cfg["seed"], "eval_split": EVAL_SPLIT, "epochs": cfg["epochs"],
                    "wall_s": wall, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                    "null_accuracy": null_acc})
    print(f"wrote {out.relative_to(REPO).as_posix()}/ and docs/EXPERIMENTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/cls_baseline.yaml"))
