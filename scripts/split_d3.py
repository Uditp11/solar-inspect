"""Pin D3 (PV01)'s train/val/test split, by scene, because PV01 ships without one.

PV01 tiles are 256 x 256 crops of orthophotos, named PV01_<x>_<y> with integer
coordinates. Neighbouring tiles are 2 or 3 name units apart along an axis (edge
contact); pairs 1 or 1.41 apart overlap with pixel-identical content. A random
tile split would put two halves of the same roof, and sometimes the same pixels,
on both sides. So tiles are grouped into scenes and scenes are split.

The rule, in order (ADR 0005 records why):

1. Joint linking. Two tiles are linked when their name-coordinate distance is
   <= L = 4.5 ("touching, including at a corner"), over all 645 tiles of all
   three subsets together: tiles of different subsets overlap pixel for pixel
   (Brick PV01_325573_1204537 with two FlatConcrete tiles), and one U-Net is
   trained on all three subsets. Scenes are the connected components.
2. SteelTile has three scenes (5, 46, and 43 tiles inside a 107-tile scene
   shared with 64 FlatConcrete tiles), too few for whole-scene assignment. So:
   - the 5-tile scene -> test;
   - the 46-tile scene: the 14 tiles at one end of its long axis (first
     principal axis of the tile coordinates) -> val, the rest -> train. The end
     is the one whose block costs fewer guard tiles;
   - the 43 SteelTile tiles of the 107-tile scene: the 9 at the end of their
     long axis farther from FlatConcrete (fewer FlatConcrete tiles within L)
     -> test; everything else in that scene, FlatConcrete included -> train.
3. Every other scene goes to one split whole, chosen by a seeded random search
   (SEED, N_DRAWS) for tile fractions near 70 / 15 / 15 within each subset.
4. Guard band. Every train tile within L of any val or test tile becomes
   "guard" and is used for nothing.

Checks, all asserted: no tile in two splits, every subset non-empty in train,
val and test, and for each subset the minimum distance from any val or test
tile to any train tile (of any subset) is > L.

No decision reads a mask. Masks are read once, after the assignment is fixed,
for the foreground fraction that is reported beside the counts.

Two outputs, as for D1:
  data/d3_split.json    the full 645-entry assignment. Gitignored.
  configs/d3_split.json committed: the rule, the seed, the counts, the
                        foreground fractions, the leakage minima, and a SHA-256
                        of the canonical assignment, which the loader checks.

    python scripts/split_d3.py

Deterministic and idempotent -- re-running leaves the working tree clean.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "d3_pv01"
FULL = REPO / "data" / "d3_split.json"
PINNED = REPO / "configs" / "d3_split.json"

SUBSETS = ("Brick", "FlatConcrete", "SteelTile")
SPLITS = ("train", "val", "test")
L = 4.5
SEED = 0
N_DRAWS = 20_000
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}
ST_VAL_BLOCK = 14       # tiles cut off the 46-tile SteelTile scene -> val
ST_TEST_BLOCK = 9       # tiles cut off the SteelTile part of the joint scene -> test
N_TILES = 645


def canonical(assignment: dict[str, str]) -> str:
    """The exact bytes the committed hash covers: sorted "subset/name,split" lines."""
    return "\n".join(f"{k},{s}" for k, s in sorted(assignment.items()))


def split_hash(assignment: dict[str, str]) -> str:
    return hashlib.sha256(canonical(assignment).encode("utf-8")).hexdigest()


def load_tiles() -> tuple[list[str], list[str], np.ndarray]:
    """Keys "subset/name", subset per tile, and (n, 2) name coordinates, sorted by key."""
    rows = []
    for sub in SUBSETS:
        for p in (ROOT / f"PV01_Rooftop_{sub}").glob("PV01_*.bmp"):
            if p.stem.endswith("_label"):
                continue
            _, x, y = p.stem.split("_")
            rows.append((f"{sub}/{p.stem}", sub, int(x), int(y)))
    rows.sort()
    assert len(rows) == N_TILES, f"expected {N_TILES} tiles, found {len(rows)}"
    return [r[0] for r in rows], [r[1] for r in rows], np.array([r[2:] for r in rows], float)


def components(xy: np.ndarray) -> np.ndarray:
    """Scene label per tile: connected components of the distance <= L graph."""
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    label = -np.ones(len(xy), int)
    for start in range(len(xy)):
        if label[start] >= 0:
            continue
        label[start], stack = start, [start]
        while stack:
            i = stack.pop()
            for j in np.flatnonzero((d[i] <= L) & (label < 0)):
                label[j] = start
                stack.append(int(j))
    return label


def axis_ends(xy: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the k tiles at each end of the first principal axis of xy.

    Ties in the projection are broken by row order, which is key order, so the
    result does not depend on the sign SVD happens to return for the axis.
    """
    c = xy - xy.mean(0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0]
    axis = axis if (axis[0], axis[1]) > (0, 0) else -axis
    proj = c @ axis
    order = np.lexsort((np.arange(len(xy)), proj))
    return order[:k], order[-k:]


def main() -> int:
    keys, subset, xy = load_tiles()
    subset = np.array(subset)
    dist = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    scene = components(xy)
    scenes = sorted(set(scene.tolist()))

    # --- SteelTile blocks ------------------------------------------------------
    st_scenes = sorted({int(s) for s in scene[subset == "SteelTile"]},
                       key=lambda s: int((scene == s).sum()))
    sizes = [(int((scene == s).sum()), int(((scene == s) & (subset == "SteelTile")).sum()))
             for s in st_scenes]
    assert sizes == [(5, 5), (46, 46), (107, 43)], f"SteelTile scenes changed: {sizes}"
    s5, s46, s107 = st_scenes

    split = np.array([""] * N_TILES, dtype=object)
    split[scene == s5] = "test"

    idx46 = np.flatnonzero(scene == s46)
    ends46 = axis_ends(xy[idx46], ST_VAL_BLOCK)
    # Cost of each end: train tiles of the scene that would touch the block.
    guard46 = [int(((dist[np.ix_(np.setdiff1d(idx46, idx46[e]), idx46[e])] <= L).any(1)).sum())
               for e in ends46]
    pick46 = int(np.argmin(guard46))                 # argmin takes the first on a tie
    split[idx46] = "train"
    split[idx46[ends46[pick46]]] = "val"

    idx43 = np.flatnonzero((scene == s107) & (subset == "SteelTile"))
    fc107 = np.flatnonzero((scene == s107) & (subset == "FlatConcrete"))
    ends43 = axis_ends(xy[idx43], ST_TEST_BLOCK)
    fc_touch = [int((dist[np.ix_(fc107, idx43[e])] <= L).any(1).sum()) for e in ends43]
    fc_min = [float(dist[np.ix_(fc107, idx43[e])].min()) for e in ends43]
    pick43 = int(np.argmin(fc_touch))
    assert fc_touch[pick43] < fc_touch[1 - pick43], f"no far end: {fc_touch}"
    split[scene == s107] = "train"
    split[idx43[ends43[pick43]]] = "test"

    # --- every other scene, whole: seeded random search -------------------------
    free = [s for s in scenes if s not in (s5, s46, s107)]
    assert not (subset[np.isin(scene, free)] == "SteelTile").any()
    fixed = {sub: {sp: int(((subset == sub) & (split == sp)).sum()) for sp in SPLITS}
             for sub in ("Brick", "FlatConcrete")}
    per_scene = np.array([[((scene == s) & (subset == sub)).sum() for sub in ("Brick", "FlatConcrete")]
                          for s in free])                           # (n_free, 2)
    totals = np.array([(subset == sub).sum() for sub in ("Brick", "FlatConcrete")])
    base = np.array([[fixed[sub][sp] for sp in SPLITS] for sub in ("Brick", "FlatConcrete")])
    target = totals[:, None] * np.array([TARGET[sp] for sp in SPLITS])[None]

    rng = random.Random(SEED)
    best, best_cost = None, float("inf")
    for _ in range(N_DRAWS):
        draw = rng.choices(range(3), weights=[TARGET[sp] for sp in SPLITS], k=len(free))
        onehot = np.eye(3, dtype=int)[draw]                         # (n_free, 3)
        got = base + per_scene.T @ onehot                           # (2, 3)
        if (got == 0).any():
            continue
        cost = float((((got - target) / totals[:, None]) ** 2).sum())
        if cost < best_cost:
            best, best_cost = draw, cost
    assert best is not None, "no draw gave every subset three non-empty splits"
    for s, k in zip(free, best):
        split[scene == s] = SPLITS[k]

    # --- guard band --------------------------------------------------------------
    held = np.isin(split, ("val", "test"))
    near_held = (dist[:, held] <= L).any(1)
    guard = (split == "train") & near_held
    split[guard] = "guard"

    # --- the guarantees ------------------------------------------------------------
    assert (split != "").all(), "a tile was not assigned"
    assert len(set(keys)) == N_TILES, "duplicate tile key"
    train = split == "train"
    leak = {}
    for sub in SUBSETS:
        for sp in ("val", "test"):
            m = (subset == sub) & (split == sp)
            assert m.any(), f"{sub} has no {sp} tiles"
            leak.setdefault(sub, {})[sp] = round(float(dist[np.ix_(m, train)].min()), 4)
            assert leak[sub][sp] > L, f"{sub} {sp} is {leak[sub][sp]} from train (<= {L})"
        assert ((subset == sub) & train).any(), f"{sub} has no train tiles"

    # --- report (masks read here only, after the assignment is fixed) ---------------
    pos = np.zeros(N_TILES)
    for i, k in enumerate(keys):
        sub, name = k.split("/")
        m = np.asarray(Image.open(ROOT / f"PV01_Rooftop_{sub}" / f"{name}_label.bmp"))
        pos[i] = (m != 0).mean()
    counts, fg = {}, {}
    for sub in SUBSETS:
        counts[sub], fg[sub] = {}, {}
        for sp in (*SPLITS, "guard"):
            m = (subset == sub) & (split == sp)
            counts[sub][sp] = int(m.sum())
            fg[sub][sp] = round(float(pos[m].mean()), 4) if m.any() else None
        counts[sub]["total"] = int((subset == sub).sum())

    print(f"joint scenes at L = {L}: {len(scenes)}")
    print(f"46-tile SteelTile scene, guard cost per end {guard46}, val end = {pick46}")
    print(f"SteelTile part of the 107-tile scene, FlatConcrete tiles within L per end "
          f"{fc_touch}, min distance {fc_min}, test end = {pick43}")
    print(f"search: seed {SEED}, {N_DRAWS} draws, best cost {best_cost:.6f}")
    print(f"{'subset':<13}{'train':>6}{'val':>5}{'test':>5}{'guard':>6}"
          f"{'fg train':>10}{'fg val':>8}{'fg test':>8}{'fg guard':>9}")
    for sub in SUBSETS:
        c, f = counts[sub], fg[sub]
        fmt = lambda v: "-" if v is None else f"{v:.4f}"
        print(f"{sub:<13}{c['train']:>6}{c['val']:>5}{c['test']:>5}{c['guard']:>6}"
              f"{fmt(f['train']):>10}{fmt(f['val']):>8}{fmt(f['test']):>8}{fmt(f['guard']):>9}")
    print(f"min distance to any train tile: {leak}")

    assignment = dict(zip(keys, split.tolist()))
    digest = split_hash(assignment)
    FULL.parent.mkdir(parents=True, exist_ok=True)
    FULL.write_text(
        json.dumps({"seed": SEED, "sha256": digest, "assignment": assignment}, indent=1, sort_keys=True),
        encoding="utf-8", newline="\n",
    )
    PINNED.write_text(
        json.dumps(
            {
                "_": "GENERATED by scripts/split_d3.py -- do not hand-edit. The full "
                     "assignment lives in data/d3_split.json (gitignored); this file "
                     "pins it. sha256 is over the sorted 'subset/name,split' lines.",
                "rule": {
                    "linking": "joint over all three subsets; tiles linked when the "
                               "name-coordinate distance is <= link_distance; scenes are "
                               "the connected components",
                    "link_distance": L,
                    "steeltile": {
                        "scene_5": "test",
                        "scene_46": f"{ST_VAL_BLOCK} tiles at the end of the first principal "
                                    "axis whose block costs fewer guard tiles -> val, rest -> train",
                        "scene_107": f"{ST_TEST_BLOCK} SteelTile tiles at the end of their first "
                                     "principal axis with fewer FlatConcrete tiles within "
                                     "link_distance -> test, rest of the scene -> train",
                    },
                    "other_scenes": "whole, seeded random search for per-subset fractions "
                                    "near target_fraction",
                    "guard": "train tiles within link_distance of any val or test tile; "
                             "used for nothing",
                },
                "seed": SEED,
                "n_draws": N_DRAWS,
                "target_fraction": TARGET,
                "n_scenes": len(scenes),
                "steeltile_measured": {
                    "scene_46_guard_cost_per_end": guard46,
                    "scene_107_flatconcrete_within_L_per_end": fc_touch,
                    "scene_107_min_distance_to_flatconcrete_per_end": fc_min,
                },
                "total": N_TILES,
                "sha256": digest,
                "counts": counts,
                "foreground_fraction": fg,
                "min_distance_to_train": leak,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"\nsha256 {digest}")
    print(f"wrote {FULL.relative_to(REPO).as_posix()} (gitignored) "
          f"and {PINNED.relative_to(REPO).as_posix()} (committed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
