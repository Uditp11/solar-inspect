"""Is D1's near-duplicate leakage live, or inert? Two measurements, VAL only.

    python scripts/near_dup_d1.py          # first: writes data/d1_near_dup.json
    python scripts/leakage_check_d1.py

**Val only.** The equivalent breakdown on test is computed as a subgroup of the
single test evaluation, from that run's per-image predictions -- running it here
would spend the one test evaluation on a diagnostic.

**A. Class-matched accuracy.** Leaky val images against clean val images
reweighted to the leaky class mix (see leakage.py). This is the comparison the
brief asked for, and on its own it is *not sufficient*: matching on class does not
match on within-class difficulty, and a >= 0.98 cosine threshold preferentially
selects the most prototypical crop of a class -- the fully-dark Offline-Module,
the featureless No-Anomaly. Those are easier for any model, including one that has
never seen them.

**B. The control that separates the two.** Retrain with every train image that is
a >= 0.98 neighbour of a leaky val image deleted from train, and re-score the same
val images. If the advantage was memorisation it must fall; if it is intrinsic
difficulty it will not. Three seeds per arm, because the arms differ in train-set
size and therefore in step count and RNG stream, and a single-seed difference here
would be indistinguishable from run-to-run noise.

The control is asymmetric and that is worth saying out loud: removing the
neighbours also shrinks train, which pushes accuracy down on its own. So **no drop
is strong evidence against memorisation**, while a drop is ambiguous between
memorisation and the smaller training set. A size-matched random-removal arm is
reported alongside for exactly that reason.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.data import load_d1                   # noqa: E402
from solar_inspect.classification.leakage import class_matched, report  # noqa: E402
from solar_inspect.classification.train import (confusion, fit,          # noqa: E402
                                                metrics)

NEAR_DUP = REPO / "data" / "d1_near_dup.json"
THR = 0.98
SPLIT = "val"
SEEDS = (0, 1, 2)


@torch.no_grad()
def predict(model, d, rows: torch.Tensor, bs: int) -> np.ndarray:
    model.eval()
    return torch.cat([model(d.batch(rows[i:i + bs])).argmax(1)
                      for i in range(0, len(rows), bs)]).cpu().numpy()


def main() -> int:
    if not NEAR_DUP.exists():
        raise SystemExit("run scripts/near_dup_d1.py first")
    cfg = yaml.safe_load((REPO / "configs" / "cls_baseline.yaml").read_text(encoding="utf-8"))
    d = load_d1()
    nd = json.loads(NEAR_DUP.read_text(encoding="utf-8"))
    assert nd["paths"] == d.paths, "d1_near_dup.json was built against a different row order"

    rows = d.index[SPLIT]
    rows_np = rows.cpu().numpy()
    labels = d.labels[rows].cpu().numpy()
    leaky = np.array(nd["best_train_cosine"])[rows_np] >= THR
    bs = cfg["batch_size"]

    # ---- A. class-matched, on the committed baseline ---------------------------
    model, *_ = fit(cfg | {"seed": SEEDS[0]}, d)
    correct = predict(model, d, rows, bs) == labels
    print(f"\n=== A. {SPLIT}: class-matched leaky vs clean (cosine >= {THR}) ===")
    r = class_matched(correct, labels, leaky, d.classes)
    print(report(r))

    # ---- B. the control -------------------------------------------------------
    # Every train row that is a >= 0.98 neighbour of a leaky val row, from the full
    # pair list rather than from each val image's single best match: an image with
    # four near-duplicates in train is only deduplicated if all four go.
    train_set = set(d.index["train"].tolist())
    leaky_val = set(rows_np[leaky].tolist())
    neigh = {j for i, j, _ in nd["cosine_pairs_0.98"] if i in leaky_val and j in train_set}
    neigh |= {i for i, j, _ in nd["cosine_pairs_0.98"] if j in leaky_val and i in train_set}
    neigh_t = torch.tensor(sorted(neigh), dtype=torch.int64, device=d.images.device)
    keep = torch.tensor(sorted(train_set - neigh), dtype=torch.int64, device=d.images.device)

    # Size-matched random removal, stratified by class so the two removals take the
    # same class mix out of train -- otherwise this arm removes 300 No-Anomaly
    # images where the other removed 300 mostly-Offline-Module ones.
    rng = np.random.default_rng(0)
    tr_np = d.index["train"].cpu().numpy()
    tr_lab = d.labels[d.index["train"]].cpu().numpy()
    neigh_lab = d.labels[neigh_t].cpu().numpy()
    drop_rand: list[int] = []
    for c in range(len(d.classes)):
        k = int((neigh_lab == c).sum())
        pool = tr_np[tr_lab == c]
        drop_rand += rng.choice(pool, size=min(k, len(pool)), replace=False).tolist()
    keep_rand = torch.tensor(sorted(train_set - set(drop_rand)),
                             dtype=torch.int64, device=d.images.device)

    print(f"\n=== B. control: {len(neigh)} train images are >= {THR} neighbours of the "
          f"{int(leaky.sum())} leaky val images ===")
    print(f"arms: full train ({len(train_set)}) | neighbours removed ({len(keep)}) | "
          f"class-matched random removal ({len(keep_rand)})")

    arms = {"full": None, "neighbours-removed": keep, "random-removed": keep_rand}
    out: dict[str, dict[str, list[float]]] = {}
    for name, tr in arms.items():
        accs_l, accs_c, f1s = [], [], []
        for s in SEEDS:
            m, *_ = fit(cfg | {"seed": s}, d, train_rows=tr, quiet=True)
            pred = predict(m, d, rows, bs)
            ok = pred == labels
            rr = class_matched(ok, labels, leaky, d.classes)
            accs_l.append(rr["acc_leaky"])
            accs_c.append(rr["acc_clean_matched"])
            f1s.append(metrics(confusion(labels, pred, len(d.classes)))["macro_f1"])
        out[name] = {"leaky": accs_l, "clean_matched": accs_c, "macro_f1": f1s}

    print(f"\n{'arm':<22}{'leaky acc':>20}{'clean matched':>20}{'leaky - clean':>18}"
          f"{'val macro-F1':>20}")
    for name, v in out.items():
        l, c, f = (np.array(v[k]) for k in ("leaky", "clean_matched", "macro_f1"))
        print(f"{name:<22}{l.mean():>13.4f} +/-{l.std(ddof=1):.4f}"
              f"{c.mean():>13.4f} +/-{c.std(ddof=1):.4f}"
              f"{(l - c).mean():>11.4f} +/-{(l - c).std(ddof=1):.4f}"
              f"{f.mean():>13.4f} +/-{f.std(ddof=1):.4f}")
    print("The macro-F1 column is the size of the problem: the whole-split cost of the "
          "leakage is\nbounded by full minus neighbours-removed, and the random-removed "
          "arm says how much of\nthat is just the smaller training set.")

    full, rem = np.array(out["full"]["leaky"]), np.array(out["neighbours-removed"]["leaky"])
    rnd = np.array(out["random-removed"]["leaky"])
    print(f"\nleaky-val accuracy, neighbours removed minus full: "
          f"{(rem.mean() - full.mean()):+.4f}  "
          f"(size-matched random removal: {(rnd.mean() - full.mean()):+.4f})")
    print("If memorisation drove the gap in A, removing the neighbours must cost more "
          "than removing the same number of random images of the same classes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
