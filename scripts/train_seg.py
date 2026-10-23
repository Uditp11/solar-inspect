"""Train the PV01 U-Net, selecting on the val split only.

    python scripts/train_seg.py                  # the committed configs/seg_unet.yaml
    python scripts/train_seg.py epochs=5         # key=value overrides, recorded

Everything the run writes goes under runs/seg_unet_<run>/ (gitignored):
  manifest.json   config, overrides, git SHA and dirty flag (taken before the run
                  writes anything), split hash, versions, GPU, peak VRAM, wall time,
                  the selected epoch and its per-subset val scores. Written at the
                  start with "status": "running" and rewritten at the end.
  epochs.jsonl    one line per epoch: train loss, lr, per-subset val IoU/Dice/
                  precision/recall, their mean IoU, seconds.
  last.pt         model, optimizer, scheduler and scaler after the latest epoch.
  best.pt         model at the epoch with the highest mean per-subset val IoU.

It prints `git status --porcelain` at the end, which must be empty.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.train import git_state   # noqa: E402

CONFIG = "configs/seg_unet.yaml"


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
    unknown = set(over) - set(cfg)
    if unknown:
        raise SystemExit(f"unknown config keys {sorted(unknown)}")
    cfg.update(over)
    print(f"git {sha[:7]} dirty={dirty}  overrides={over}")

    import numpy as np
    import torch

    from solar_inspect.segmentation.data import load_d3
    from solar_inspect.segmentation.train import EVAL_SPLIT, fit

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO / "runs" / f"seg_unet_{run}"
    out.mkdir(parents=True, exist_ok=False)

    d = load_d3()
    manifest = {
        "run": run, "status": "running", "git_sha": sha, "git_dirty": dirty,
        "config_path": CONFIG, "overrides": over, "config": cfg, "eval_split": EVAL_SPLIT,
        "split_sha256": d.split_sha256,
        "norm": {"mean": d.mean.flatten().tolist(), "std": d.std.flatten().tolist()},
        "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
        "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    path = out / "manifest.json"
    write = lambda m: path.write_text(json.dumps(m, indent=2) + "\n",   # noqa: E731
                                      encoding="utf-8", newline="\n")
    write(manifest)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    res = fit(cfg, d, out)
    epoch_s = [h["epoch_s"] for h in res["history"]]
    manifest.update({
        "status": "finished", "params": res["params"], "wall_s": res["wall_s"],
        "median_epoch_s": float(np.median(epoch_s)),
        "median_train_s": float(np.median([h["train_s"] for h in res["history"]])),
        "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "selected_epoch": res["best"]["epoch"], "best": res["best"],
        "first_train_loss": res["history"][0]["train_loss"],
        "last_train_loss": res["history"][-1]["train_loss"],
    })
    write(manifest)
    print(json.dumps({k: manifest[k] for k in ("run", "selected_epoch", "wall_s", "median_epoch_s",
                                                "peak_vram_reserved_mib")}, indent=2))
    print(f"wrote {out.relative_to(REPO).as_posix()}/")

    st = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                        capture_output=True, text=True).stdout
    print(f"git status --porcelain after the run: {st.strip() or '(empty)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
