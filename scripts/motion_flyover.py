"""Why greedy IoU beats SORT on D5: one-step and coasting predictions on ground truth.

    python scripts/motion_flyover.py runs/flyover_<run> [runs/flyover_<run> ...]

Post-hoc: measured after the held-out sweep, to explain it; changes no setting or
number. Reads ground truth only (no detector, no tracker), development and held-out
sequences of each run; the runs may differ only in pan speed.

For every ground-truth id, along each run of consecutive frames where it is visible,
two predictions of a later box are compared with the true box:

- stay: the last true box (what greedy IoU matches against);
- Kalman: SORT's filter (`tracking/kalman.py`, SORT's constants), started at the run's
  first box and updated with every true box so far, then predicted forward.

One step ahead (the next frame) and coasting k = 5 and 10 steps with no update (the
occlusion case, against the true box k frames later, which may sit in a later visible
run). Per pan speed: median and 10th percentile IoU of each prediction, and the share
of predictions below IoU 0.1 (the trackers' gate). Also the median box width and
height, the median per-frame centre displacement, and the IoU of the true box with
itself one frame later, which is the stay prediction's one-step IoU by definition.
Writes runs/track_motion_<utc>/motion.json.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.train import git_state                  # noqa: E402
from solar_inspect.detection.boxes import iou                             # noqa: E402
from solar_inspect.tracking import kalman                                 # noqa: E402

GATE = 0.1
HORIZONS = (1, 5, 10)


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([iou(x[None], y[None])[0, 0] for x, y in zip(a, b)])


def sequence(z) -> dict[str, list]:
    """Per horizon: (stay boxes, Kalman boxes, true boxes); plus sizes and steps."""
    out = {k: ([], [], []) for k in HORIZONS}
    sizes, steps = [], []
    Q, R = kalman.default_Q(), kalman.default_R()
    for gid in np.unique(z["gt_id"]):
        m = z["gt_id"] == gid
        frames, boxes = z["gt_frame"][m], z["gt_box"][m]
        at = dict(zip(frames.tolist(), boxes))
        sizes += [b[2:] - b[:2] for b in boxes]
        x = P = None
        for t, b in zip(frames.tolist(), boxes):
            if t - 1 not in at:                    # a visible run starts: new filter
                x, P = kalman.initial_state(kalman.box_to_z(b))
            else:
                x, P = kalman.predict(x, P, Q)
                x, P = kalman.update(x, P, kalman.box_to_z(b), R)
            if t + 1 in at:
                c0, c1 = (b[:2] + b[2:]) / 2, (at[t + 1][:2] + at[t + 1][2:]) / 2
                steps.append(float(np.linalg.norm(c1 - c0)))
            xc, Pc = x, P
            for k in range(1, max(HORIZONS) + 1):
                xc, Pc = kalman.predict(xc, Pc, Q)
                if k in out and t + k in at:
                    out[k][0].append(b)
                    out[k][1].append(kalman.z_to_box(xc))
                    out[k][2].append(at[t + k])
    return {"pred": out, "sizes": sizes, "steps": steps}


def summary(ious: np.ndarray) -> dict:
    return {"n": int(len(ious)), "median": float(np.median(ious)),
            "p10": float(np.percentile(ious, 10)), "below_gate": float((ious < GATE).mean())}


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit(__doc__)
    sha, dirty = git_state()
    result = {}
    for run in map(lambda a: Path(a).resolve(), argv):
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if set(manifest.get("overrides", {})) - {"pan_speed"}:
            raise SystemExit(f"{run.name}: overrides other than pan_speed")
        speed = manifest["params"]["pan_speed"]
        acc = {k: ([], [], []) for k in HORIZONS}
        sizes, steps = [], []
        for seed in manifest["sequences"]:
            s = sequence(np.load(run / f"seed_{int(seed):03d}.npz"))
            for k in HORIZONS:
                for i in range(3):
                    acc[k][i].extend(s["pred"][k][i])
            sizes += s["sizes"]
            steps += s["steps"]
        sizes = np.array(sizes)
        row = {"run": run.name, "sequences": len(manifest["sequences"]),
               "median_box_w": float(np.median(sizes[:, 0])),
               "median_box_h": float(np.median(sizes[:, 1])),
               "median_step_px": float(np.median(steps)), "steps": len(steps)}
        for k in HORIZONS:
            stay, kf, true = (np.array(a) for a in acc[k])
            row[f"k{k}"] = {"stay": summary(pairwise_iou(stay, true)),
                            "kalman": summary(pairwise_iou(kf, true))}
        row["self_iou_next_frame_median"] = row["k1"]["stay"]["median"]
        result[str(speed)] = row
        print(f"pan {speed}: box {row['median_box_w']:.1f} x {row['median_box_h']:.1f} px, "
              f"step {row['median_step_px']:.2f} px  " + "  ".join(
                  f"k={k} stay {row[f'k{k}']['stay']['median']:.3f}/{row[f'k{k}']['stay']['p10']:.3f}/"
                  f"{row[f'k{k}']['stay']['below_gate']:.3f} kalman {row[f'k{k}']['kalman']['median']:.3f}/"
                  f"{row[f'k{k}']['kalman']['p10']:.3f}/{row[f'k{k}']['kalman']['below_gate']:.3f}"
                  for k in HORIZONS))
    out = REPO / "runs" / f"track_motion_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True)
    (out / "motion.json").write_text(json.dumps({"git_sha": sha, "git_dirty": dirty,
                                                 "by_pan_speed": result}, indent=2) + "\n",
                                     encoding="utf-8", newline="\n")
    print(f"wrote {(out / 'motion.json').relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
