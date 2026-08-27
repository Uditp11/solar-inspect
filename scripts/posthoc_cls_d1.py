"""Calibration and base-rate correction, both from the saved logits of ONE run.

    python scripts/posthoc_cls_d1.py [runs/cls_final_test_<utc>]

Neither section trains anything and neither reads the test split again. They read
`logits.npz` written by scripts/eval_test_d1.py, which is what makes it true that
the test split was evaluated once and then looked at from three angles -- the
headline metrics, the leakage subgroup, and everything here.

**A. Calibration.** Temperature scaling, **T fitted on val by minimising NLL, and
never on test.** Fitting T on test and then reporting test ECE measures how well a
free parameter was fitted to the thing it is being scored on. Global ECE and
class-balanced ECE are both reported, because at 57:1 the global number is 50%
No-Anomaly by construction and says almost nothing about the eleven defect
classes an inspection actually cares about.

**B. Base rate.** D1 is 49.9% No-Anomaly. Field prevalence of anomalies is ~2.2%
(the dataset's own ICLR 2020 paper). Recall is unaffected by that shift, precision
is destroyed by it, and the arithmetic is three lines -- so the content of this
section is the two assumptions, which are printed in the same place as the table:

  A1. Per-class sensitivity AND the whole class-conditional confusion structure
      P(pred = c | true = k) transport unchanged from this dataset to the field.
  A2. Only the anomaly/no-anomaly ratio shifts. The relative mix among the eleven
      anomaly classes is the same in the field as in D1.

Neither is obviously true. A1 assumes a field camera, altitude, season and module
type distribution that make a Cell crop as separable as it is here; D1 is pooled
from mixed midwave and longwave sensors at 3-15 cm/px, so it is already a mixture
and "the field" is a different mixture. A2 is the weaker of the two: Soiling and
Vegetation are seasonal and site-specific in a way Diode faults are not, so the
relative mix almost certainly does not hold. Both are stated rather than defended.

Under A2 the reweighting has exactly two distinct weights -- one for No-Anomaly,
one shared by every anomaly class -- so **recall is invariant** and the ranking of
images by anomaly score is invariant too. Prevalence moves the precision axis and
nothing else. That is why the threshold sweep below can be run on the ordinary
calibrated score and only its precision column reweighted.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.train import confusion, metrics    # noqa: E402

FIELD_PREVALENCE = 0.022        # anomalies, spec 5.2 / the D1 paper
NULL_CLASS = "No-Anomaly"
BINS = 15
PRECISION_TARGET = 0.50


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(z: np.ndarray, y: np.ndarray, T: float) -> float:
    p = softmax(z, T)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = BINS) -> float:
    """Expected calibration error: sum over bins of |accuracy - confidence| * share.

    Equal-width bins on the predicted probability of the predicted class. Hand
    checkable: a model that is right 70% of the time in the bin it calls 0.9
    contributes 0.2 times that bin's share of the data.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability_figure(rows: list[tuple[str, np.ndarray, np.ndarray]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.linspace(0.0, 1.0, BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for label, conf, correct in rows:
        idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, BINS - 1)
        acc = np.array([correct[idx == b].mean() if (idx == b).any() else np.nan
                        for b in range(BINS)])
        ax.plot(centres, acc, "o-", ms=4, label=label)
    ax.set_xlabel("predicted probability of the predicted class")
    ax.set_ylabel("observed accuracy")
    ax.set_title("D1 ResNet-18, reliability - T fitted on val, never on test")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(run_dir: str | None = None) -> int:
    run_dir = run_dir or sorted(glob.glob(str(REPO / "runs" / "cls_final_test_*")))[-1]
    run = Path(run_dir)
    z = np.load(run / "logits.npz")
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    classes: list[str] = man["classes"]
    k, na = len(classes), classes.index(NULL_CLASS)
    L = {s: z[f"{s}_logits"] for s in ("val", "test")}
    Y = {s: z[f"{s}_labels"] for s in ("val", "test")}
    print(f"run {man['run']}  config commit {man['config_commit_sha'][:7]}  "
          f"split {man['split_sha256'][:8]}")

    # ---- A. calibration -------------------------------------------------------
    fitted = minimize_scalar(lambda t: nll(L["val"], Y["val"], t),
                             bounds=(0.05, 10.0), method="bounded")
    T = float(fitted.x)
    print(f"\n=== A. temperature scaling, T fitted on VAL by NLL ===")
    print(f"T = {T:.4f}   (T > 1 means the raw model was overconfident)")
    print(f"val NLL  {nll(L['val'], Y['val'], 1.0):.4f} -> {nll(L['val'], Y['val'], T):.4f}")
    print(f"test NLL {nll(L['test'], Y['test'], 1.0):.4f} -> "
          f"{nll(L['test'], Y['test'], T):.4f}")

    rows, table = [], {}
    for split in ("val", "test"):
        for name, t in (("raw", 1.0), ("T-scaled", T)):
            p = softmax(L[split], t)
            conf, pred = p.max(1), p.argmax(1)
            correct = (pred == Y[split]).astype(float)
            per_class = {c: ece(conf[Y[split] == i], correct[Y[split] == i])
                         for i, c in enumerate(classes) if (Y[split] == i).any()}
            table[(split, name)] = {
                "global": ece(conf, correct),
                "class_balanced": float(np.mean(list(per_class.values()))),
                "per_class": per_class,
                "mean_confidence": float(conf.mean()),
                "accuracy": float(correct.mean()),
            }
            if split == "test":
                rows.append((f"test, {name}", conf, correct))

    print(f"\n{'':<20}{'global ECE':>12}{'class-balanced ECE':>22}"
          f"{'mean conf':>12}{'accuracy':>11}")
    for (split, name), v in table.items():
        print(f"{split + ', ' + name:<20}{v['global']:>12.4f}{v['class_balanced']:>22.4f}"
              f"{v['mean_confidence']:>12.4f}{v['accuracy']:>11.4f}")
    print("\nThe two ECE columns disagree on purpose. Global ECE is half No-Anomaly by\n"
          "construction, and No-Anomaly is the class the model is most confident and\n"
          "most often right about, so it drags the global number toward zero.")

    print(f"\nper-class ECE on test (T-scaled), support beside every number:")
    sup = np.bincount(Y["test"], minlength=k)
    for c in sorted(table[("test", "T-scaled")]["per_class"],
                    key=lambda c: -table[("test", "T-scaled")]["per_class"][c]):
        i = classes.index(c)
        print(f"  {c:<16}{sup[i]:>7}   raw {table[('test', 'raw')]['per_class'][c]:.4f}"
              f"   T-scaled {table[('test', 'T-scaled')]['per_class'][c]:.4f}")

    reliability_figure(rows, run / "reliability_test.png")

    # ---- B. base rate ---------------------------------------------------------
    cm = np.array(man["confusion_matrix"], dtype=float)
    support = cm.sum(1)
    Lc = cm / np.maximum(support[:, None], 1)          # P(pred = c | true = k)
    prior = support / support.sum()
    field = FIELD_PREVALENCE * prior / (1.0 - prior[na])
    field[na] = 1.0 - FIELD_PREVALENCE

    def precision_at(pri: np.ndarray) -> np.ndarray:
        pred_mass = pri @ Lc                            # P(pred = c)
        return np.divide(pri * np.diag(Lc), pred_mass,
                         out=np.zeros(k), where=pred_mass > 0)

    m = metrics(cm.astype(int))
    p_now, p_field = precision_at(prior), precision_at(field)
    print(f"\n=== B. per-class precision at {FIELD_PREVALENCE:.1%} field prevalence ===")
    print("A1: sensitivity and the whole confusion structure P(pred=c | true=k) "
          "transport unchanged.\nA2: only the anomaly/no-anomaly ratio shifts; the "
          "relative mix among the eleven anomaly\n    classes is unchanged. Neither is "
          "obviously true. A2 is the weaker: Soiling and\n    Vegetation are seasonal "
          "and site-specific in a way Diode faults are not.\n")
    print(f"{'class':<16}{'test n':>8}{'prior':>9}{'field prior':>13}{'recall':>9}"
          f"{'precision':>11}{'precision @2.2%':>17}")
    for i, c in enumerate(classes):
        print(f"{c:<16}{int(support[i]):>8}{prior[i]:>9.4f}{field[i]:>13.5f}"
              f"{m['recall'][i]:>9.4f}{p_now[i]:>11.4f}{p_field[i]:>17.4f}")
    print("\nRecall has no column at field prevalence because it does not have one: "
          "under A2 the\nreweighting is a single constant on every anomaly class, so "
          "recall is invariant and\nonly precision moves.")

    # The triage decision an O&M crew actually makes: flag, or do not flag.
    p_test = softmax(L["test"], T)
    score = 1.0 - p_test[:, na]
    is_anom = Y["test"] != na
    w = np.where(is_anom, FIELD_PREVALENCE / (1.0 - prior[na]),
                 (1.0 - FIELD_PREVALENCE) / prior[na])
    order = np.argsort(-score)
    tp = np.cumsum(w[order] * is_anom[order])
    fp = np.cumsum(w[order] * ~is_anom[order])
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / w[is_anom].sum()
    thr = score[order]

    print(f"\n=== B2. the triage threshold: flag if P(any anomaly) >= t ===")
    print("Calibrated probabilities, T from val. The ranking is prevalence-invariant, "
          "so this\nsweep is on the ordinary score and only precision is reweighted.\n")
    print(f"{'t':>8}{'flagged/1000 modules':>23}{'recall':>10}"
          f"{'precision, as measured':>24}{'precision @2.2%':>17}")
    prec_now = np.cumsum(is_anom[order]) / np.arange(1, len(order) + 1)
    for t in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        j = int(np.searchsorted(-thr, -t, side="right")) - 1
        if j < 0:
            continue
        flagged = 1000.0 * (tp[j] + fp[j]) / w.sum()
        print(f"{t:>8.2f}{flagged:>23.1f}{rec[j]:>10.4f}{prec_now[j]:>24.4f}"
              f"{prec[j]:>17.4f}")

    ok = np.where(prec >= PRECISION_TARGET)[0]
    print(f"\nthreshold for a stated precision target of {PRECISION_TARGET:.0%} "
          f"at {FIELD_PREVALENCE:.1%} prevalence:")
    if len(ok) == 0:
        print(f"  UNREACHABLE. The highest precision this model attains at field "
              f"prevalence is\n  {prec.max():.4f}, at t = {thr[int(prec.argmax())]:.4f} "
              f"with recall {rec[int(prec.argmax())]:.4f}. That is the result: at 2.2%\n"
              "  prevalence no operating point on this model gives one real defect per "
              "two dispatches.")
    else:
        j = int(ok[np.argmax(rec[ok])])
        print(f"  t = {thr[j]:.4f}  ->  precision {prec[j]:.4f}, recall {rec[j]:.4f}, "
              f"{1000.0 * (tp[j] + fp[j]) / w.sum():.1f} flags per 1,000 modules")
        print(f"  Chosen for precision, not for macro-F1. The macro-F1-optimal operating "
              f"point is\n  argmax, which sits far to the left of this and would flag "
              "many times as many modules.")

    (run / "posthoc.json").write_text(json.dumps({
        "temperature": T, "bins": BINS,
        "ece": {f"{s}_{n}": v for (s, n), v in table.items()},
        "field_prevalence": FIELD_PREVALENCE,
        "prior": prior.tolist(), "field_prior": field.tolist(),
        "recall": m["recall"].tolist(),
        "precision_as_measured": p_now.tolist(),
        "precision_at_field": p_field.tolist(),
        "precision_target": PRECISION_TARGET,
        "classes": classes,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {(run / 'posthoc.json').relative_to(REPO).as_posix()} and "
          f"{(run / 'reliability_test.png').relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
