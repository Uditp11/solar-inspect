"""Rebuild D2's train/val/test split, because the published one is contaminated.

The archive ships 353 image files that are only 252 unique images. 34 of the 35
published test images are byte-identical to a train or val image, so evaluating on
the published test split measures memorisation. See docs/DATA.md and ADR 0002.

This script:
  1. deduplicates by SHA-256 of the JPG bytes (deterministic representative:
     train > val > test, then filename order),
  2. groups the 252 unique frames into sorties by segmenting the acquisition
     timestamps at gaps > 300 s,
  3. assigns whole sorties to splits so no acquisition window is shared.

The split is materialised as a clean directory tree rather than as lists pointing back
into the original folders. Lists would be cheaper, but Ultralytics derives its
labels.cache path from the first image's label directory, so a split whose images span
several physical folders collides with another split's cache file and silently rescans
or, worse, reuses the wrong one. A physical tree gives each split its own cache.

Files are hardlinked where the filesystem allows it (NTFS does), so the tree costs
almost nothing on disk; it falls back to copying.

    python scripts/split_d2.py

Writes data/d2_split/{train,val,test}/{images,labels}/ and regenerates configs/d2.yaml.
Deterministic — same inputs give the same tree.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
D2 = REPO / "data" / "d2"
OUT = REPO / "data" / "d2_split"
YAML = REPO / "configs" / "d2.yaml"

GAP_SECONDS = 300

# Sortie -> split. Rationale in ADR 0002.
#   1 is a 216 box/frame density outlier -> train, so it cannot distort an eval.
#   4 is the temporally latest full sortie -> test, the "next flight" framing.
#   5 is the two label-less 17:02 transit frames -> train as background images.
ASSIGN = {1: "train", 2: "val", 3: "train", 4: "test", 5: "train"}


def unique_frames() -> list[tuple[datetime, Path]]:
    """One representative per distinct image, ordered by acquisition time."""
    seen: dict[str, Path] = {}
    for split in ("train", "val", "test"):          # preference order
        for f in sorted((D2 / split / "images").glob("*.jpg")):
            seen.setdefault(hashlib.sha256(f.read_bytes()).hexdigest(), f)
    frames = [(datetime.strptime(f.name.split("_")[1], "%Y%m%d%H%M%S"), f)
              for f in seen.values()]
    return sorted(frames)


def sorties(frames: list[tuple[datetime, Path]]) -> list[list[tuple[datetime, Path]]]:
    """Segment at gaps longer than GAP_SECONDS."""
    groups, cur = [], [frames[0]]
    for prev, nxt in zip(frames, frames[1:]):
        if (nxt[0] - prev[0]).total_seconds() > GAP_SECONDS:
            groups.append(cur)
            cur = []
        cur.append(nxt)
    groups.append(cur)
    return groups


def label_of(img: Path) -> Path:
    return img.parent.parent / "labels" / (img.stem + ".txt")


def n_boxes(img: Path) -> int:
    lab = label_of(img)
    if not lab.exists():
        return 0
    return len([l for l in lab.read_text().splitlines() if l.strip()])


def link(src: Path, dst: Path) -> None:
    """Hardlink if possible, else copy. Idempotent."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    frames = unique_frames()
    groups = sorties(frames)
    assert len(frames) == 252, f"expected 252 unique frames, got {len(frames)}"
    assert len(groups) == len(ASSIGN), f"expected {len(ASSIGN)} sorties, got {len(groups)}"

    print(f"{'#':>2} {'window':<14} {'imgs':>5} {'boxes':>6} {'mean/frm':>9}  split")
    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for i, g in enumerate(groups, 1):
        b = sum(n_boxes(f) for _, f in g)
        print(f"{i:>2} {g[0][0]:%H:%M}-{g[-1][0]:%H:%M}    {len(g):>5} {b:>6} "
              f"{b/len(g):>9.1f}  {ASSIGN[i]}")
        splits[ASSIGN[i]].extend(f for _, f in g)

    if OUT.exists():
        shutil.rmtree(OUT)          # rebuild from scratch so a stale tree cannot survive
    YAML.parent.mkdir(parents=True, exist_ok=True)

    print()
    counts = {}
    for name, imgs in splits.items():
        imgs = sorted(imgs)
        (OUT / name / "images").mkdir(parents=True, exist_ok=True)
        (OUT / name / "labels").mkdir(parents=True, exist_ok=True)
        for f in imgs:
            link(f, OUT / name / "images" / f.name)
            lab = label_of(f)
            if lab.exists():        # absent == background frame; Ultralytics expects no file
                link(lab, OUT / name / "labels" / lab.name)
        b = sum(n_boxes(f) for f in imgs)
        counts[name] = (len(imgs), b)
        print(f"  {name:5s} {len(imgs):>3} images  {b:>6} boxes  -> {OUT.name}/{name}/")

    total_i = sum(c[0] for c in counts.values())
    total_b = sum(c[1] for c in counts.values())
    print(f"  {'total':5s} {total_i:>3} images  {total_b:>6} boxes")
    assert total_i == 252 and total_b == 19525, "split does not account for every frame"

    YAML.write_text(
        "# GENERATED by scripts/split_d2.py — do not hand-edit; re-run after cloning.\n"
        "#\n"
        "# The published D2 splits are contaminated: 34 of 35 test images are\n"
        "# byte-identical to a train or val image. This is the deduplicated,\n"
        "# sortie-grouped replacement. See docs/DATA.md and ADR 0002.\n"
        "#\n"
        f"#   train  sorties 1+3+5  {counts['train'][0]:>3} images  {counts['train'][1]:>6} boxes\n"
        f"#   val    sortie  2      {counts['val'][0]:>3} images  {counts['val'][1]:>6} boxes\n"
        f"#   test   sortie  4      {counts['test'][0]:>3} images  {counts['test'][1]:>6} boxes\n"
        "#\n"
        "# val: points at the VALIDATION split. It must never point at test —\n"
        "# doing so selects best.pt on the test split and burns the evaluate-once\n"
        "# guarantee without any visible symptom.\n"
        f"path: {OUT.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: panel\n",
        encoding="utf-8",
    )
    print(f"\nwrote {YAML.relative_to(REPO).as_posix()}")

    # The guarantee this file exists to protect.
    lines = dict(l.split(": ", 1) for l in YAML.read_text(encoding="utf-8").splitlines()
                 if l[:1].isalpha() and ": " in l)
    assert lines["val"] == "val/images", f"val: does not point at val -> {lines['val']}"
    assert lines["train"] != lines["val"] != lines["test"], "split paths collide"
    print("check: val -> val/images, distinct from train and test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
