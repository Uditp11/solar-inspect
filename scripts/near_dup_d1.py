"""D1 near-duplicate sweep. Measures only; changes nothing on disk except a cache.

    python scripts/near_dup_d1.py

D2 shipped 353 files that were 252 unique images, and nobody noticed until the
bytes were hashed. This script asks the same question of D1, and asks it two ways,
because the two answers mean different things:

  **Cosine similarity** (zero-mean, L2-normalised, full 200 M pairs) answers "how
  self-similar is this dataset?" On 40x24 crops after per-image zero-meaning the
  dominant component of a crop is its *layout* -- a dark rectangle on a lighter
  surround -- not its identity, so two different offline modules score 0.98
  without being the same panel. High cosine is therefore a **ceiling on possible
  leakage**, not evidence of it. Methodology matches the adjacent-vs-random
  contrast used on D2 in docs/DATA.md.

  **Max absolute pixel difference** (exhaustive, not a subset of the cosine
  shortlist) answers "is this the same image?" A JPEG re-encode of one crop has
  different bytes and near-identical pixels, so byte-hashing cannot see it. The
  search is exact: max|a-b| <= c implies ||a-b||_2 <= c*sqrt(960), so an L2 filter
  with a margin gives a superset of candidates which are then checked exactly.
  Byte-hashing is reported alongside to show what it misses.

Outputs a report to stdout and `data/d1_near_dup.json` (gitignored) holding the
pair lists and each image's nearest train neighbour, so scripts/dedup_d1.py and
the leakage diagnostic do not each recompute the sweep.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.data import load_d1          # noqa: E402

OUT = REPO / "data" / "d1_near_dup.json"
COS_THR = (0.90, 0.95, 0.98, 0.99)
DN_CAPS = (0, 2, 4, 8, 16)
CH = 2000                       # rows of the similarity matrix held at once
DIM = 40 * 24


def cosine_pairs(X: torch.Tensor, thr: float) -> list[tuple[int, int, float]]:
    """All i < j with cosine >= thr, over the full pairwise matrix."""
    n = X.shape[0]
    out = []
    for a in range(0, n, CH):
        S = X[a:a + CH] @ X.T
        rows = torch.arange(a, min(a + CH, n), device=X.device)[:, None]
        S = S.masked_fill(torch.arange(n, device=X.device)[None, :] <= rows, -2.0)
        ii, jj = (S >= thr).nonzero(as_tuple=True)
        out += [(a + int(i), int(j), float(S[i, j])) for i, j in zip(ii, jj)]
    return out


def cosine_histogram(X: torch.Tensor, bins: int = 2000) -> torch.Tensor:
    """Histogram of every i < j cosine, for quantiles without holding 200 M floats."""
    n = X.shape[0]
    hist = torch.zeros(bins, dtype=torch.float64, device=X.device)
    for a in range(0, n, CH):
        S = X[a:a + CH] @ X.T
        rows = torch.arange(a, min(a + CH, n), device=X.device)[:, None]
        keep = torch.arange(n, device=X.device)[None, :] > rows
        hist += torch.histc(S[keep], bins=bins, min=-1.0, max=1.0).double()
    return hist


def near_identical_pairs(Xr: torch.Tensor, cap: int) -> list[tuple[int, int, int]]:
    """All i < j with max|a-b| <= cap, exhaustively. Returns (i, j, max|diff|).

    Two stages. The screen is ||a-b||_2^2 <= cap^2 * 960, which every qualifying
    pair satisfies, widened by 5% + 512 because these are fp32 sums of terms
    around 1.6e7 and the rounding is worth more than the tightness. The second
    stage recomputes max|a-b| exactly on the candidates, so the screen can only
    cost time, never correctness.
    """
    n = Xr.shape[0]
    sq = (Xr * Xr).sum(1)
    thr = (cap ** 2) * DIM * 1.05 + 512
    cand_i, cand_j = [], []
    for a in range(0, n, CH):
        D = sq[a:a + CH, None] + sq[None, :] - 2 * (Xr[a:a + CH] @ Xr.T)
        rows = torch.arange(a, min(a + CH, n), device=Xr.device)[:, None]
        D = D.masked_fill(torch.arange(n, device=Xr.device)[None, :] <= rows, 1e9)
        ii, jj = (D <= thr).nonzero(as_tuple=True)
        cand_i.append(ii + a)
        cand_j.append(jj)
    ci, cj = torch.cat(cand_i), torch.cat(cand_j)

    out = []
    for a in range(0, len(ci), 50_000):
        i, j = ci[a:a + 50_000], cj[a:a + 50_000]
        d = (Xr[i] - Xr[j]).abs().amax(1)
        keep = (d <= cap).nonzero(as_tuple=True)[0]
        out += [(int(i[k]), int(j[k]), int(d[k])) for k in keep]
    return sorted(out)


def main() -> int:
    d = load_d1()
    paths, classes = d.paths, d.classes
    n = len(paths)
    cls = {p: classes[int(d.labels[i])] for i, p in enumerate(paths)}
    split = {p: s for s in ("train", "val", "test")
             for p in (paths[i] for i in d.index[s].tolist())}

    Xr = d.images.reshape(n, -1).float()                    # raw DN, for pixel distance
    X = Xr - Xr.mean(1, keepdim=True)                       # zero-mean per crop
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-8)     # L2-normalise

    # ---- A. the full pairwise cosine distribution -------------------------------
    npairs = n * (n - 1) // 2
    hist = cosine_histogram(X)
    centres = torch.linspace(-1, 1, 2001, device=X.device)
    centres = (centres[:-1] + centres[1:]) / 2
    cum = torch.cumsum(hist, 0)
    quant = lambda p: float(centres[int((cum >= p * npairs).nonzero()[0])])   # noqa: E731
    mean = float((hist * centres.double()).sum() / npairs)

    pairs = cosine_pairs(X, min(COS_THR))
    print(f"\n=== A. pairwise cosine over all {npairs:,} pairs ===")
    print(f"mean {mean:.4f}  median {quant(.5):.4f}  p90 {quant(.9):.4f}  "
          f"p99 {quant(.99):.4f}  p99.99 {quant(.9999):.4f}")
    for t in COS_THR:
        c = sum(1 for _, _, s in pairs if s >= t)
        print(f"  >= {t}: {c:>6} pairs  ({c / npairs:.3e} of all pairs)")

    # ---- B. where the >= 0.98 pairs live ----------------------------------------
    p98 = [(i, j) for i, j, s in pairs if s >= 0.98]
    by_pair = Counter(tuple(sorted((cls[paths[i]], cls[paths[j]]))) for i, j in p98)
    n_of = Counter(cls[p] for p in paths)
    print(f"\n=== B. the {len(p98)} pairs >= 0.98, by class pair (top 12) ===")
    for (a, b), c in by_pair.most_common(12):
        print(f"  {c:>4}  {a} / {b}")
    # Within-class is the column that matters. Macro-F1 turns on the low-support
    # classes, and a pair straddling two classes cannot inflate either one's recall
    # by memorisation -- it can only make them confusable.
    print(f"\n{'class':<16}{'n':>7}{'within-class':>14}{'any pair':>10}")
    for c in sorted(n_of, key=lambda c: -n_of[c]):
        touch = sum(1 for i, j in p98 if cls[paths[i]] == c or cls[paths[j]] == c)
        print(f"  {c:<16}{n_of[c]:>7}{by_pair.get((c, c), 0):>14}{touch:>10}")

    # ---- C. file-id gaps: the "consecutive frames" mechanism, tested -------------
    ids = np.array([int(Path(p).stem) for p in paths])
    gap = np.abs(ids[[i for i, _ in p98]] - ids[[j for _, j in p98]])
    print(f"\n=== C. file-id gap inside those {len(p98)} pairs ===")
    for lo, hi in ((0, 1), (2, 5), (6, 20), (21, 100), (101, 10 ** 9)):
        k = int(((gap >= lo) & (gap <= hi)).sum())
        print(f"  gap {lo}-{'inf' if hi > 10**8 else hi}: {k:>5}")

    # ---- D. the leakage ceiling: held-out images with a train neighbour ---------
    tr = d.index["train"]
    best = torch.full((n,), -2.0, device=X.device)
    arg = torch.zeros(n, dtype=torch.long, device=X.device)
    for a in range(0, n, CH):
        S = X[a:a + CH] @ X[tr].T
        gi = torch.arange(a, min(a + CH, n), device=X.device)[:, None]
        S = S.masked_fill(tr[None, :] == gi, -2.0)          # never match self
        v, k = S.max(1)
        best[a:a + CH], arg[a:a + CH] = v, tr[k]
    best_c, arg_c = best.cpu().numpy(), arg.cpu().numpy()
    sp = np.array([split.get(p, "dropped") for p in paths])

    print("\n=== D. images with a TRAIN near-neighbour (self excluded) ===")
    print(f"{'split':<7}{'n':>7}" + "".join(f"{t:>14}" for t in COS_THR))
    for s in ("val", "test", "train"):
        m = sp == s
        row = "".join(f"{int((best_c[m] >= t).sum()):>7}{(best_c[m] >= t).mean() * 100:>6.1f}%"
                      for t in COS_THR)
        print(f"{s:<7}{int(m.sum()):>7}" + row)
    for s in ("val", "test"):
        m = (sp == s) & (best_c >= 0.98)
        dis = sum(1 for i in np.where(m)[0] if cls[paths[i]] != cls[paths[arg_c[i]]])
        print(f"  of the {int(m.sum())} {s} images >= 0.98, the label DISAGREES "
              f"with the train neighbour in {dis}")

    # ---- E. same image, not just similar: exhaustive max|pixel diff| ------------
    sha = {}
    for i, p in enumerate(paths):
        sha.setdefault(hashlib.sha256((REPO / "data" / "d1" / p).read_bytes()).digest(),
                       []).append(i)
    byte_pairs = {(v[a], v[b]) for v in sha.values() if len(v) > 1
                  for a in range(len(v)) for b in range(a + 1, len(v))}

    dn = near_identical_pairs(Xr, max(DN_CAPS))
    print(f"\n=== E. exhaustive max|pixel diff| (not a subset of the cosine shortlist) ===")
    print(f"byte-identical pairs (sha256 over the JPEG file): {len(byte_pairs)}")
    print(f"{'cap':>6}{'pairs':>8}{'contradictory':>15}{'straddle split':>16}{'byte-identical':>16}")
    for cap in DN_CAPS:
        sel = [(i, j) for i, j, m in dn if m <= cap]
        contra = sum(1 for i, j in sel if cls[paths[i]] != cls[paths[j]])
        cross = sum(1 for i, j in sel if split.get(paths[i], "dropped") != split.get(paths[j], "dropped"))
        bid = sum(1 for i, j in sel if (i, j) in byte_pairs)
        print(f"{cap:>4} DN{len(sel):>8}{contra:>15}{cross:>16}{bid:>16}")

    # This table is over all 20,000 crops as published, which is the population the
    # 4 DN threshold was chosen against (ADR 0003). The dropped images are still in
    # the tensor and still form pairs here; what the dedup guarantees is that no
    # pair has both members inside a split, which is the line below.
    survive = [(i, j) for i, j, m in dn
               if m <= 4 and paths[i] in split and paths[j] in split]
    print(f"\nof the <= 4 DN pairs, {len(survive)} still have both members in a "
          "split after configs/d1_dedup.json is applied (0 is the point of it)")

    print("\ncontradictory pairs at <= 4 DN, over the dataset as published:")
    for i, j, m in dn:
        if m <= 4 and cls[paths[i]] != cls[paths[j]]:
            print(f"  max|d| {m}  {Path(paths[i]).name:>10} [{cls[paths[i]]:<14} "
                  f"{split.get(paths[i], 'DROP'):<5}]  {Path(paths[j]).name:>10} "
                  f"[{cls[paths[j]]:<14} {split.get(paths[j], 'DROP'):<5}]")

    # ---- F. the D2-analogue null: adjacent by file id ---------------------------
    order = np.argsort(ids)
    adj = torch.tensor([[int(order[k]), int(order[k + 1])] for k in range(n - 1)],
                       device=X.device)
    sa = (X[adj[:, 0]] * X[adj[:, 1]]).sum(1).cpu().numpy()
    print(f"\n=== F. adjacent by file id (n={len(sa)}), the nearest thing D1 has "
          "to consecutive frames ===")
    print(f"mean {sa.mean():.4f}  median {np.median(sa):.4f}  "
          f"p90 {np.percentile(sa, 90):.4f}  max {sa.max():.4f}  "
          f">=0.98: {int((sa >= 0.98).sum())}")
    print("(compare with the all-pairs mean in A: if near-duplication came from "
          "consecutive acquisition, adjacent pairs would separate from the null.)")

    OUT.write_text(json.dumps({
        "_": "GENERATED by scripts/near_dup_d1.py. Gitignored. Row indices are into "
             "the sorted image_filepath order that data.py uses.",
        "paths": paths,
        "cosine_pairs_0.98": [[i, j, round(s, 6)] for i, j, s in pairs if s >= 0.98],
        "pixel_pairs": [[i, j, m] for i, j, m in dn],
        "byte_identical_pairs": sorted(map(list, byte_pairs)),
        "best_train_cosine": [round(float(v), 6) for v in best_c],
        "best_train_row": [int(v) for v in arg_c],
    }) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {OUT.relative_to(REPO).as_posix()} (gitignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
