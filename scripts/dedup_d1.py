"""Deduplicate D1 at <= 4 DN, by connected component, before it is split.

    python scripts/dedup_d1.py

Writes `configs/d1_dedup.json` (committed, small): the images to exclude and why.
`scripts/split_d1.py` reads it, so the split is defined over the deduplicated set
and the exclusion list is reviewable without re-running a GPU sweep.

**Why a pixel threshold and not a byte hash.** Hashing the JPEG bytes finds 22
identical pairs. It cannot see a re-encode of the same crop, which has different
bytes and near-identical pixels. Four more contradictory pairs sit outside byte
equality, three of them pitting No-Anomaly against an anomaly class, two of them
straddling a split boundary.

**Why 4 DN.** Measured, not chosen. Exhaustive pair counts by max|a-b| run
22 / 25 / 30 / 234 / 10,271 at 0 / 2 / 4 / 8 / 16 DN, and the count of pairs
carrying contradictory labels runs 6 / 9 / 10 / 13 / 915. Four is the end of the
plateau: the pair count is still flat and the contradiction count has stopped
moving, while by 8 DN the population has started to change character and by 16 DN
it is dominated by genuinely different images that happen to be close. A threshold
with a measured knee behind it survives "why 4 and not 8"; one chosen by taste
does not. scripts/near_dup_d1.py prints that table.

**Why components and not pairs.** Three copies of one image are three pairs and
one group. Deriving a dropped count from a pair count double-counts, which is how
the same dedup produced two different totals in the previous session's report.

Two rules over the components:

  - **One distinct label in the component** -> keep one representative, chosen as
    the lexicographically first filename so the choice is deterministic and does
    not depend on row order.
  - **More than one distinct label** -> **drop the whole component.** The images
    are real and one of the labels is right, but which one is unknowable from
    here. Keeping a representative would be a coin flip recorded as ground truth,
    and it would be recorded in the split that every later number is measured on.

The dropped contradictory components are a **measured** label-noise floor for D1,
not an estimated one. They are listed in docs/DATA.md by filename and label.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.data import load_images        # noqa: E402

OUT = REPO / "configs" / "d1_dedup.json"
CAP_DN = 4
CH = 2000
DIM = 40 * 24


def near_identical_pairs(X: torch.Tensor, cap: int) -> list[tuple[int, int, int]]:
    """All i < j with max|a-b| <= cap, exhaustively. Returns (i, j, max|diff|).

    Screen on ||a-b||_2^2 <= cap^2 * DIM, which every qualifying pair satisfies,
    widened by 5% + 512 because these are fp32 sums of terms around 1.6e7 and the
    rounding is worth more than the tightness. The candidates are then checked
    exactly, so the screen can cost time but not correctness.
    """
    n = X.shape[0]
    sq = (X * X).sum(1)
    thr = (cap ** 2) * DIM * 1.05 + 512
    ci, cj = [], []
    for a in range(0, n, CH):
        D = sq[a:a + CH, None] + sq[None, :] - 2 * (X[a:a + CH] @ X.T)
        rows = torch.arange(a, min(a + CH, n), device=X.device)[:, None]
        D = D.masked_fill(torch.arange(n, device=X.device)[None, :] <= rows, 1e9)
        ii, jj = (D <= thr).nonzero(as_tuple=True)
        ci.append(ii + a)
        cj.append(jj)
    ci, cj = torch.cat(ci), torch.cat(cj)

    out = []
    for a in range(0, len(ci), 50_000):
        i, j = ci[a:a + 50_000], cj[a:a + 50_000]
        d = (X[i] - X[j]).abs().amax(1)
        for k in (d <= cap).nonzero(as_tuple=True)[0]:
            out.append((int(i[k]), int(j[k]), int(d[k])))
    return sorted(out)


def components(n: int, edges: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Connected components by union-find. Singletons are not returned."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b

    groups: dict[int, list[int]] = {}
    for i, j in edges:
        groups.setdefault(find(i), []).extend((i, j))
    return {r: sorted(set(v)) for r, v in groups.items()}


def main() -> int:
    images, labels, classes, paths = load_images()
    n = len(paths)
    X = images.reshape(n, -1).float()

    pairs = near_identical_pairs(X, CAP_DN)
    comps = components(n, [(i, j) for i, j, _ in pairs])
    lab = labels.cpu().tolist()

    consistent, contradictory = [], []
    for members in sorted(comps.values()):
        names = sorted(paths[i] for i in members)
        labs = {classes[lab[i]] for i in members}
        (contradictory if len(labs) > 1 else consistent).append(
            {"files": names, "labels": sorted(labs)})

    # A consistent component keeps its first filename; a contradictory one keeps
    # nothing. Every other image is untouched.
    drop = sorted({f for c in consistent for f in c["files"][1:]}
                  | {f for c in contradictory for f in c["files"]})

    sizes = sorted(len(c["files"]) for c in consistent + contradictory)
    print(f"\n<= {CAP_DN} DN: {len(pairs)} pairs -> {len(comps)} components "
          f"({len(consistent)} single-label, {len(contradictory)} contradictory)")
    print(f"component sizes: " + "  ".join(
        f"{s}x{sizes.count(s)}" for s in sorted(set(sizes))))
    print(f"dropped: {sum(len(c['files']) - 1 for c in consistent)} redundant "
          f"representatives + {sum(len(c['files']) for c in contradictory)} images in "
          f"contradictory components = {len(drop)}")
    print(f"D1 goes from {n} to {n - len(drop)} images")

    print(f"\ncontradictory components ({len(contradictory)}), the measured "
          "label-noise floor:")
    for c in contradictory:
        print(f"  {' + '.join(Path(f).name for f in c['files']):<28} "
              f"{' / '.join(c['labels'])}")

    OUT.write_text(json.dumps({
        "_": "GENERATED by scripts/dedup_d1.py -- do not hand-edit. Images listed in "
             "`drop` are excluded before scripts/split_d1.py assigns splits.",
        "threshold_max_abs_dn": CAP_DN,
        "n_before": n, "n_after": n - len(drop),
        "n_pairs": len(pairs), "n_components": len(comps),
        "n_components_single_label": len(consistent),
        "n_components_contradictory": len(contradictory),
        "n_dropped": len(drop),
        "contradictory_components": contradictory,
        "single_label_components": consistent,
        "drop": drop,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {OUT.relative_to(REPO).as_posix()} "
          f"({len(drop)} images excluded) -- now re-run scripts/split_d1.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
