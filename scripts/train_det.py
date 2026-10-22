"""Fine-tune YOLOv8n on D2 with Ultralytics, selecting on the val split only.

    python scripts/train_det.py                  # the committed configs/det_yolo.yaml
    python scripts/train_det.py epochs=2         # key=value overrides, recorded

Reads `configs/det_yolo.yaml`; overrides are recorded in the run's manifest beside
the config, so a smoke run never needs a config edit.

Guards, all before Ultralytics reads anything:
- `configs/d2.yaml` must not resolve train or val to the test tree
  (`solar_inspect.detection.data.splits`);
- the git SHA and whether the tree is dirty are taken first, since the run itself
  writes files.

Everything the run writes goes under `runs/` (gitignored): the training run in
`runs/det_<run>/`, and the downloaded weights, `yolov8n.pt` and the model Ultralytics
fetches for its AMP check, in `runs/weights/`. Ultralytics resolves both downloads
against the working directory, so the script works from there and passes absolute
paths for everything else. It prints `git status --porcelain` at the end, which
must be empty.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.train import git_state   # noqa: E402
from solar_inspect.detection.data import splits             # noqa: E402

CONFIG = "configs/det_yolo.yaml"
WEIGHTS = REPO / "runs" / "weights"


def overrides(argv: list[str]) -> dict:
    out = {}
    for a in argv:
        k, sep, v = a.partition("=")
        if not sep:
            raise SystemExit(f"expected key=value, got {a!r}")
        out[k] = yaml.safe_load(v)
    return out


def main(argv: list[str]) -> int:
    sha, dirty = git_state()
    cfg = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
    over = overrides(argv)
    cfg.update(over)
    data = REPO / cfg.pop("data")
    paths = splits(data)
    print(f"git {sha[:7]} dirty={dirty}; val -> {paths['val'].relative_to(REPO).as_posix()}")

    import torch
    from ultralytics import YOLO

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    os.chdir(WEIGHTS)
    model = YOLO(str(WEIGHTS / cfg.pop("model")))
    t0 = time.time()
    model.train(data=str(data), project=str(REPO / "runs"), name=f"det_{run}",
                exist_ok=False, **cfg)
    wall = time.time() - t0
    save_dir = Path(model.trainer.save_dir)

    import csv
    rows = list(csv.DictReader((save_dir / "results.csv").open(encoding="utf-8")))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    fit = [float(r["metrics/mAP50-95(B)"]) for r in rows]
    best = fit.index(max(fit))        # Ultralytics 8.4 saves best.pt on mAP50-95 alone
    manifest = {
        "run": run, "git_sha": sha, "git_dirty": dirty, "config_path": CONFIG,
        "overrides": over, "config": {**cfg, "data": data.relative_to(REPO).as_posix()},
        "splits": {k: v.relative_to(REPO).as_posix() for k, v in paths.items()},
        "epochs_run": len(rows), "best_epoch": best + 1,
        "best": {k: float(rows[best][k]) for k in rows[best] if k.startswith("metrics/")},
        "last": {k: float(rows[-1][k]) for k in rows[-1] if k.startswith(("metrics/", "train/"))},
        "first": {k: float(rows[0][k]) for k in rows[0] if k.startswith(("metrics/", "train/"))},
        "wall_s": wall,
        "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ultralytics": __import__("ultralytics").__version__,
    }
    (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                            encoding="utf-8", newline="\n")
    print(json.dumps({k: manifest[k] for k in ("run", "epochs_run", "best_epoch", "best",
                                                "wall_s", "peak_vram_reserved_mib")}, indent=2))
    print(f"wrote {save_dir.relative_to(REPO).as_posix()}/")

    import subprocess
    st = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                        capture_output=True, text=True).stdout
    print(f"git status --porcelain after the run: {st.strip() or '(empty)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
