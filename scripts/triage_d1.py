"""D1 triage: the work-order list, rebuilt from the saved test pass. No model is run.

    python scripts/triage_d1.py

Rules in configs/triage.yaml, committed before the first run.

1. **The order is proven before anything is used.** logits.npz has no image ids.
   The test rows of the pinned split (load_d1 checks its hash), in its pinned
   order, must give back, row for row: the saved test and val labels, the saved
   per-image near-duplicate cosine (a continuous per-image value, so a shuffled
   order cannot match it by chance), the recorded confusion matrix and the
   recorded macro-F1. If any of these fails the script stops.
2. Calibrated probabilities at the val-fitted temperature; one Finding per test
   image, with delta_dn_uncalibrated from the raw 8-bit crop.
3. analytics.triage.dispatch at t_val, read from the threshold run's output. The
   dispatch count must equal that run's test flag count, or the script stops.
4. Counts per CoA and extent, as counted and reweighted to 2.2% field prevalence;
   per triage class, the precision at 2.2% from the existing base-rate table
   (posthoc.json, argmax predictions, not recomputed); and the post-hoc label
   checks, labelled post-hoc.

Writes runs/triage_d1_<utc>/ (gitignored): findings.csv (all test images),
work_orders.csv (the dispatched ones, in order) and summary.json.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.analytics.triage import COA_TABLE, EXTENT_ORDER, NULL_CLASS, \
    delta_dn_uncalibrated, dispatch                                          # noqa: E402
from solar_inspect.classification.calibration import softmax                # noqa: E402
from solar_inspect.classification.train import confusion, git_state, metrics  # noqa: E402
from solar_inspect.contract import Finding                                  # noqa: E402

CONFIG = "configs/triage.yaml"
NEAR_DUP = REPO / "data" / "d1_near_dup.json"


def prove_order(d, z, man) -> list[str]:
    """Stop unless the pinned test order reproduces everything the test pass saved."""
    test, val = d.index["test"].cpu().numpy(), d.index["val"].cpu().numpy()
    y = d.labels.cpu().numpy()
    checks = {}
    checks["test labels row for row"] = np.array_equal(y[test], z["test_labels"])
    checks["val labels row for row"] = np.array_equal(y[val], z["val_labels"])
    nd = json.loads(NEAR_DUP.read_text(encoding="utf-8"))
    checks["near-dup paths = load order"] = nd["paths"] == d.paths
    checks["near-dup cosine row for row"] = np.array_equal(
        np.array(nd["best_train_cosine"])[test], z["test_best_train_cosine"])
    cm = confusion(z["test_labels"], z["test_logits"].argmax(1), len(d.classes))
    checks["confusion matrix = recorded"] = cm.tolist() == man["confusion_matrix"]
    f1 = float(metrics(cm)["macro_f1"])
    checks["macro-F1 = recorded"] = abs(f1 - man["macro_f1"]) < 1e-12 and round(f1, 4) == 0.6956
    checks["classes = recorded"] = d.classes == man["classes"]
    for k, ok in checks.items():
        print(f"  order proof: {k:<32} {'ok' if ok else 'FAILED'}")
    if not all(checks.values()):
        raise SystemExit("the D1 test order cannot be proven; stopping")
    print(f"  macro-F1 rebuilt {f1:.4f}; {len(test)} test images in pinned order")
    return [d.paths[i] for i in test]


def main() -> int:
    sha, dirty = git_state()
    if dirty:
        raise SystemExit("the working tree is not clean; commit first")
    cfg = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
    src = REPO / cfg["source"]
    run = src.parent
    z = np.load(src)
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    post = json.loads((run / "posthoc.json").read_text(encoding="utf-8"))
    T = post["temperature"]
    if abs(T - cfg["temperature"]) > 5e-5:
        raise SystemExit(f"posthoc.json T = {T}, config says {cfg['temperature']}")
    thr = json.loads((REPO / cfg["threshold_from"]).read_text(encoding="utf-8"))
    if thr["t_val"] is None:
        raise SystemExit("no val threshold reached the target; no dispatch")
    t = thr["t_val"]
    if list(cfg["extent_order"]) != list(EXTENT_ORDER):
        raise SystemExit("extent order in the config differs from the code's")

    from solar_inspect.classification.data import load_d1
    d = load_d1("cpu")
    paths = prove_order(d, z, man)
    classes = man["classes"]
    na = classes.index(NULL_CLASS)
    labels = z["test_labels"]
    probs = softmax(z["test_logits"], T)
    crops = d.images[d.index["test"]].cpu().numpy()[:, 0]           # raw uint8 (n, 40, 24)
    names = [Path(p).name for p in paths]

    findings = [Finding(finding_id=n, frame_id=Path(n).stem, frame_index=i,
                        source_frame_path=f"data/d1/{p}", bbox=(0.0, 0.0, 24.0, 40.0),
                        det_score=float("nan"),           # D1 crops are given, not detected
                        defect_class=classes[int(probs[i].argmax())],
                        cls_confidence=float(probs[i].max()),
                        delta_dn_uncalibrated=delta_dn_uncalibrated(crops[i]))
                for i, (n, p) in enumerate(zip(names, paths))]
    rows = dispatch(names, probs, classes, t)
    by_id = {f.finding_id: f for f in findings}
    sent = [r for r in rows if r["triage"].dispatch_rank is not None]

    # ---- the stop-condition: the dispatch count is B0's test flag count ----------------
    n, n_b0 = len(sent), thr["test"]["flags"]
    is_anom = labels != na
    q = 1.0 - is_anom.mean()
    p = cfg["field_prevalence"]
    w = np.where(is_anom, p / (1.0 - q), (1.0 - p) / q)
    idx = {nm: i for i, nm in enumerate(names)}
    sent_i = np.array([idx[r["image"]] for r in sent])
    per_1000_field = 1000.0 * w[sent_i].sum() / w.sum()
    print(f"\nt_val = {t:.6f}; dispatched {n} of {len(names)}; threshold run flagged {n_b0}")
    print(f"  per 1,000: {1000 * n / len(names):.1f} as counted on D1 test, "
          f"{per_1000_field:.1f} at {p:.1%} prevalence (threshold run: "
          f"{thr['test']['flags_per_1000_field']:.1f})")
    if n != n_b0 or abs(per_1000_field - thr["test"]["flags_per_1000_field"]) > 1e-9:
        raise SystemExit("the dispatch count differs from N_val; stopping")

    def tally(key) -> dict:
        c, wf = Counter(), Counter()
        for r in sent:
            k = key(r)
            c[k] += 1
            wf[k] += w[idx[r["image"]]]
        return {str(k): {"count": c[k], "per_1000_counted": 1000.0 * c[k] / len(names),
                         "per_1000_field": 1000.0 * wf[k] / w.sum()} for k in sorted(c)}

    by_coa = tally(lambda r: r["triage"].coa)
    by_extent = tally(lambda r: r["triage"].extent)
    prec_field = dict(zip(post["classes"], post["precision_at_field"]))
    by_class = tally(lambda r: r["triage_class"])
    for c in by_class:
        by_class[c].update(coa=COA_TABLE[c][0], extent=COA_TABLE[c][1],
                           precision_at_field_argmax=prec_field[c])
    top_null = sum(r["top_is_no_anomaly"] for r in sent)

    # ---- post-hoc, labelled post-hoc: changes no reported number ----------------------
    true_coa = np.array([COA_TABLE[classes[k]][0] for k in labels])
    coa3_sent = [r for r in sent if r["triage"].coa == 3]
    coa3_true_too = sum(true_coa[idx[r["image"]]] == 3 for r in coa3_sent)
    sent_set = {r["image"] for r in sent}
    coa3_true = [nm for nm, c in zip(names, true_coa) if c == 3]
    coa3_missed = sum(nm not in sent_set for nm in coa3_true)
    posthoc = {"coa3_dispatched": len(coa3_sent), "coa3_dispatched_true_coa3": int(coa3_true_too),
               "share": coa3_true_too / len(coa3_sent) if coa3_sent else float("nan"),
               "true_coa3_modules": len(coa3_true), "true_coa3_not_dispatched": int(coa3_missed),
               "true_coa3_dispatched_any_coa": len(coa3_true) - int(coa3_missed)}

    dest = REPO / "runs" / f"triage_d1_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    dest.mkdir(parents=True)
    with open(dest / "findings.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["finding_id", "frame_index", "source_frame_path", "defect_class",
                     "cls_confidence", "triage_class", "p_class", "p_any",
                     "delta_dn_uncalibrated", "dispatch_rank"])
        for r in sorted(rows, key=lambda r: by_id[r["image"]].frame_index):
            f = by_id[r["image"]]
            wr.writerow([f.finding_id, f.frame_index, f.source_frame_path, f.defect_class,
                         repr(f.cls_confidence), r["triage_class"], repr(r["p_class"]),
                         repr(r["p_any"]), repr(f.delta_dn_uncalibrated),
                         r["triage"].dispatch_rank or ""])
    with open(dest / "work_orders.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["dispatch_rank", "finding_id", "triage_class", "coa", "coa_basis",
                     "extent", "action", "p_class", "p_any", "top_is_no_anomaly",
                     "delta_dn_uncalibrated", "class_precision_at_field"])
        for r in sent:
            tr = r["triage"]
            wr.writerow([tr.dispatch_rank, tr.finding_id, r["triage_class"], tr.coa,
                         tr.coa_basis, tr.extent, tr.action, repr(r["p_class"]),
                         repr(r["p_any"]), int(r["top_is_no_anomaly"]),
                         repr(by_id[r["image"]].delta_dn_uncalibrated),
                         repr(prec_field[r["triage_class"]])])
    summary = {"git_sha": sha, "config_path": CONFIG, "config": cfg, "temperature": T,
               "t_val": t, "test_images": len(names), "dispatched": n,
               "per_1000_counted": 1000.0 * n / len(names), "per_1000_field": per_1000_field,
               "flagged_top_class_no_anomaly": top_null, "by_coa": by_coa,
               "by_extent": by_extent, "by_class": by_class, "posthoc": posthoc}
    (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8",
                                       newline="\n")

    print(f"  flagged modules whose top class is No-Anomaly: {top_null}")
    for title, tab in (("CoA", by_coa), ("extent", by_extent)):
        print(f"\nby {title}:")
        for k, v in tab.items():
            print(f"  {k:<10}{v['count']:>6}  {v['per_1000_counted']:>7.1f} per 1,000 counted"
                  f"  {v['per_1000_field']:>6.2f} per 1,000 at 2.2%")
    print("\nby triage class (precision at 2.2% from the base-rate table, argmax predictions):")
    for c, v in sorted(by_class.items(), key=lambda kv: (-kv[1]["coa"], -kv[1]["count"])):
        print(f"  {c:<16} CoA {v['coa']}  {v['extent']:<10}{v['count']:>6}"
              f"  {v['per_1000_field']:>6.2f} per 1,000 at 2.2%"
              f"  precision@2.2% {v['precision_at_field_argmax']:.4f}")
    print(f"\nPOST-HOC (labels; changes no reported number): {json.dumps(posthoc)}")
    print("\nfirst 10 work orders:")
    for r in sent[:10]:
        tr = r["triage"]
        print(f"  #{tr.dispatch_rank:<4} {tr.finding_id:<11} {r['triage_class']:<15} CoA {tr.coa} "
              f"{tr.extent:<10} P(class) {r['p_class']:.4f}  ddn "
              f"{by_id[r['image']].delta_dn_uncalibrated:.1f}")
    print(f"\nwrote {dest.relative_to(REPO).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
