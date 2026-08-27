"""Is the near-duplicate leakage in D1 inert, or is the model scoring on it?

4.4% of val and 4.5% of test have a >= 0.98 cosine neighbour in train (measured by
scripts/near_dup_d1.py). Comparing raw accuracy on that subset against the rest is
worthless: the leaky subset is overwhelmingly No-Anomaly and Offline-Module, which
are the two easiest classes, so it would score higher than the clean set even if
the model had never seen any of it.

The comparison here is **class-matched**. Per-class accuracy is measured on the
clean images, then reweighted to the leaky subset's class mix. That is the
expectation of "draw a class-stratified sample of the clean images matched to the
leaky class mix", computed exactly rather than sampled, so it carries no sampling
noise of its own.

Read the output as: if leaky sits materially above class-matched clean, the model
is being helped by images it has effectively already seen, and the decision not to
re-split D1 (ADR 0003) reopens.
"""
from __future__ import annotations

import numpy as np


def class_matched(correct: np.ndarray, labels: np.ndarray, leaky: np.ndarray,
                  classes: list[str]) -> dict:
    """Leaky accuracy vs clean accuracy reweighted to the leaky class mix.

    correct, labels, leaky: parallel arrays over the images of ONE split.
    Standard errors are binomial within each class and combined as an independent
    weighted sum -- so the leaky-minus-clean interval is honest about the fact
    that the leaky side is only ~130 images.
    """
    assert correct.shape == labels.shape == leaky.shape, "arrays must be parallel"
    clean = ~leaky
    n_l = int(leaky.sum())
    if n_l == 0:
        raise ValueError("no leaky images in this split")

    per_class, acc_c, var_c, weights = [], 0.0, 0.0, []
    for c in range(len(classes)):
        n_lc = int((leaky & (labels == c)).sum())
        if n_lc == 0:
            continue
        w = n_lc / n_l
        m_cc = clean & (labels == c)
        n_cc = int(m_cc.sum())
        p_l = float(correct[leaky & (labels == c)].mean())
        p_c = float(correct[m_cc].mean()) if n_cc else float("nan")
        if n_cc == 0:
            raise ValueError(f"class {classes[c]} has leaky images but no clean ones")
        acc_c += w * p_c
        var_c += (w ** 2) * p_c * (1 - p_c) / n_cc
        weights.append(w)
        per_class.append({"class": classes[c], "n_leaky": n_lc, "acc_leaky": p_l,
                          "n_clean": n_cc, "acc_clean": p_c, "weight": w})

    p_l = float(correct[leaky].mean())
    se_l = float(np.sqrt(p_l * (1 - p_l) / n_l))
    se_c = float(np.sqrt(var_c))
    diff = p_l - acc_c
    se_d = float(np.sqrt(se_l ** 2 + se_c ** 2))
    return {
        "n_leaky": n_l, "n_clean": int(clean.sum()),
        "acc_leaky": p_l, "se_leaky": se_l,
        "acc_clean_matched": acc_c, "se_clean_matched": se_c,
        "acc_clean_raw": float(correct[clean].mean()),
        "diff": diff, "se_diff": se_d, "z": diff / se_d if se_d else 0.0,
        "per_class": per_class,
    }


def report(r: dict) -> str:
    lines = [
        f"leaky n={r['n_leaky']}   clean n={r['n_clean']}",
        f"  accuracy, leaky               {r['acc_leaky']:.4f} +/- {r['se_leaky']:.4f}",
        f"  accuracy, clean class-matched {r['acc_clean_matched']:.4f} "
        f"+/- {r['se_clean_matched']:.4f}",
        f"  accuracy, clean unmatched     {r['acc_clean_raw']:.4f}   "
        "(not the comparison -- the class mixes differ)",
        f"  difference                    {r['diff']:+.4f} +/- {r['se_diff']:.4f}  "
        f"(z = {r['z']:+.2f})",
        "",
        f"  {'class':<16}{'n_leaky':>8}{'acc_leaky':>11}{'n_clean':>9}{'acc_clean':>11}"
        f"{'weight':>8}",
    ]
    for p in sorted(r["per_class"], key=lambda p: -p["n_leaky"]):
        lines.append(f"  {p['class']:<16}{p['n_leaky']:>8}{p['acc_leaky']:>11.4f}"
                     f"{p['n_clean']:>9}{p['acc_clean']:>11.4f}{p['weight']:>8.3f}")
    return "\n".join(lines)
