"""D2 as the detection scripts read it: split paths behind the val-is-not-test guard,
and ground-truth boxes in pixels.

`splits()` resolves `configs/d2.yaml` the way Ultralytics does (relative to the
yaml's own directory) and refuses to return if train or val resolves to, inside, or
above the test tree. Pointing `val:` at test selects best.pt on the test split with
no visible symptom, so every detection script calls this before Ultralytics reads
anything.

`check_checkpoint()` pins which file the single test evaluation may read: the
checkpoint is not in git, so its SHA-256 is committed in `configs/det_yolo.yaml`
before the test split is read, and any other file is refused.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
D2_YAML = REPO / "configs" / "d2.yaml"


def splits(data_yaml: Path = D2_YAML) -> dict[str, Path]:
    data_yaml = Path(data_yaml)
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    p = {k: (data_yaml.parent / cfg[k]).resolve() for k in ("train", "val", "test")}
    for k in ("train", "val"):
        if p[k] == p["test"] or p["test"] in p[k].parents or p[k] in p["test"].parents:
            raise SystemExit(f"{data_yaml.name}: {k}: resolves to the test tree ({p[k]}). "
                             "Refusing to run.")
    if p["train"] == p["val"]:
        raise SystemExit(f"{data_yaml.name}: train and val resolve to the same tree")
    return p


def ground_truth(images_dir: Path) -> tuple[list[str], list[np.ndarray]]:
    """Image file names (sorted) and their xyxy pixel boxes. A missing label file is a
    background image with no boxes, as Ultralytics reads it."""
    names, boxes = [], []
    for img in sorted(p for p in Path(images_dir).iterdir() if p.is_file()):
        lab = img.parent.parent / "labels" / f"{img.stem}.txt"
        rows = (np.loadtxt(lab, ndmin=2) if lab.exists() and lab.stat().st_size
                else np.zeros((0, 5)))
        with Image.open(img) as im:
            w, h = im.size
        cx, cy, bw, bh = (rows[:, 1:] * [w, h, w, h]).T
        names.append(img.name)
        boxes.append(np.column_stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]))
    return names, boxes


def check_checkpoint(weights: Path, pinned: str | None) -> str:
    """SHA-256 of `weights`; refuses unless it equals the pinned hash."""
    digest = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
    if digest != pinned:
        raise SystemExit(f"{Path(weights).name} has SHA-256 {digest[:12]}, the config pins "
                         f"{(pinned or 'nothing')[:12]}. Refusing to evaluate another checkpoint.")
    return digest
