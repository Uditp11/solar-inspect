"""The raw deduplication count of each tracker on the D5 development sequences.

    python scripts/dedup_flyover.py runs/flyover_<run>

Runs no tracker, greedy IoU and SORT, each at its defaults, over the cached detections
of every development seed in the run's config, and prints per tracker the line

    N frames, M detections, K ids, vs G true modules

where K is the number of distinct track ids reported and G the number of distinct
ground-truth ids visible in at least one frame, summed over the sequences. No metric
and no matching against ground truth: this is the count before any metric code.
Held-out seeds are not read. Writes dedup_dev.json into the run directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.train import git_state                # noqa: E402
from solar_inspect.tracking.sort import GreedyIoU, NoTracker, Sort       # noqa: E402

TRACKERS = {"no tracker": NoTracker, "greedy IoU": GreedyIoU, "SORT": Sort}


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit(__doc__)
    run = Path(argv[0]).resolve()
    sha, dirty = git_state()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    seeds = manifest["config"]["seeds"]["development"]
    frames = manifest["params"]["frames"]

    per_seed, total = {}, {name: {"detections": 0, "reported": 0, "tracks_started": 0, "ids": 0}
                           for name in TRACKERS}
    n_frames = g = 0
    for seed in seeds:
        z = np.load(run / f"seed_{seed:03d}.npz")
        assert manifest["sequences"][str(seed)]["set"] == "development"
        n_frames += frames
        g_seed = len(np.unique(z["gt_id"]))
        g += g_seed
        per_seed[seed] = {"frames": frames, "true_modules": g_seed}
        for name, cls in TRACKERS.items():
            tracker = cls()
            ids = np.concatenate([tracker.update(z["det_box"][z["det_frame"] == t])
                                  for t in range(frames)])
            r = {"detections": int(len(ids)), "reported": int((ids > 0).sum()),
                 "tracks_started": int(tracker._next_id - 1),
                 "ids": int(len(np.unique(ids[ids > 0])))}
            per_seed[seed][name] = r
            for k in r:
                total[name][k] += r[k]

    lines = {}
    for name, t in total.items():
        lines[name] = (f"{n_frames} frames, {t['detections']} detections, {t['ids']} ids, "
                       f"vs {g} true modules")
        print(f"{name:11s} {lines[name]}  ({t['tracks_started']} tracks started, "
              f"{t['reported']} detections carry a reported id)")
    out = {"git_sha": sha, "git_dirty": dirty, "run": run.name, "seeds": seeds,
           "frames": n_frames, "true_modules": g, "totals": total, "lines": lines,
           "per_seed": {str(k): v for k, v in per_seed.items()}}
    (run / "dedup_dev.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8",
                                        newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
