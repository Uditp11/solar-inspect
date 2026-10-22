"""Score the trackers on D5: the development grid, then the held-out sweep.

    python scripts/track_flyover.py dev runs/flyover_<default run>
    python scripts/track_flyover.py sweep runs/flyover_<default run> runs/flyover_<cell run> ...

Definitions are declared in docs/EXPERIMENTS.md, "D5 tracking metrics and sweep".

`dev` reads development seeds only. For SORT and greedy IoU it runs every setting of
the declared grid and picks one per tracker by the declared rule; writes
runs/track_dev_<utc>/grid.json. The choice is then committed to configs/tracking.yaml.

`sweep` reads held-out seeds only, with each tracker at its setting in
configs/tracking.yaml (tracked and unmodified). Each extra run is one cell, named by
the single override in its manifest (pan_speed, occlusion_frames or conf); the
`max_age` and `iou_gate` cells re-run the trackers on the default run. Writes
runs/track_sweep_<utc>/sweep.json.

Ignore boxes are recomputed from each frame's cached homography and band and the val
label boxes, with the generator's own functions; the visible boxes recomputed the same
way must equal the cached ground truth, frame by frame.
"""
from __future__ import annotations

import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from make_flyover import guard                                              # noqa: E402
from solar_inspect.classification.train import git_state                   # noqa: E402
from solar_inspect.detection.data import ground_truth, splits              # noqa: E402
from solar_inspect.tracking.flyover import box_corners, hull, visibility, warp_points  # noqa: E402
from solar_inspect.tracking.metrics import Frame, score                    # noqa: E402
from solar_inspect.tracking.sort import GreedyIoU, NoTracker, Sort         # noqa: E402

CONFIG = "configs/tracking.yaml"
TRACKERS = {"no tracker": NoTracker, "greedy IoU": GreedyIoU, "SORT": Sort}
CONFIG_KEY = {"greedy IoU": "greedy", "SORT": "sort"}
GRID = {"max_age": [1, 3, 5, 10, 15], "min_hits": [1, 3], "iou_gate": [0.1, 0.3, 0.5]}
SWEEP_TRACKER = {"max_age": [1, 5, 15], "iou_gate": [0.1, 0.3, 0.5]}


def load_run(run: Path, which: str, labels: dict) -> tuple[dict, dict]:
    """manifest, and per seed of set `which`: (detections per frame, truth per frame,
    bounces). Truth per frame = (gt ids, gt boxes, ignore boxes)."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config"]["source_split"] != "val":
        raise SystemExit(f"{run.name}: not built from val")
    p = manifest["params"]
    window, vf = tuple(p["window"]), p["visible_fraction"]
    seqs = {}
    for seed in manifest["config"]["seeds"][which]:
        assert manifest["sequences"][str(seed)]["set"] == which
        z = np.load(run / f"seed_{seed:03d}.npz")
        corners = box_corners(labels[str(z["source_frame"])])
        dets, truth = [], []
        for t in range(p["frames"]):
            occ = z["occluder"][t] if z["occluder"][t].any() else None
            quads = warp_points(z["H"][t], corners)
            inside, visible = visibility(quads, window, occ)
            shown = visible >= vf
            m = z["gt_frame"] == t
            assert (np.flatnonzero(shown) + 1).tolist() == z["gt_id"][m].tolist(), (seed, t)
            assert np.allclose(hull(quads[shown], window), z["gt_box"][m])
            truth.append((z["gt_id"][m], z["gt_box"][m], hull(quads[(inside > 0) & ~shown], window)))
            dets.append(z["det_box"][z["det_frame"] == t])
        seqs[seed] = (dets, truth, int(z["bounces"]))
    return manifest, seqs


def track(name: str, kw: dict, dets, truth) -> list[Frame]:
    tr = TRACKERS[name](**kw)
    frames = []
    for d, (gi, gb, ib) in zip(dets, truth):
        ids = tr.update(d)
        r = ids > 0
        frames.append(Frame(gi, gb, ib, ids[r], d[r]))
    return frames


def evaluate(name: str, kw: dict, seqs: dict) -> dict:
    out = score({str(s): track(name, kw, d, t) for s, (d, t, _) in seqs.items()})
    per = [out[str(s)] for s in seqs]
    rel = [r["rel_count_error"] for r in per]
    out["overall"].update({
        "frames": sum(len(d) for d, _, _ in seqs.values()),
        "detections": sum(sum(len(x) for x in d) for d, _, _ in seqs.values()),
        "bounces": sum(b for _, _, b in seqs.values()),
        "sum_abs_K_minus_G": sum(abs(r["K_minus_G"]) for r in per),
        "sum_idf1": sum(r["idf1"] for r in per),
        "rel_median": float(np.median(rel)), "rel_min": float(min(rel)), "rel_max": float(max(rel)),
        "idsw_disagree": [s for s in seqs if out[str(s)]["idsw"] != out[str(s)]["idsw_mm"]],
    })
    return out


def labels_val() -> dict:
    names, boxes = ground_truth(splits()["val"])
    return dict(zip(names, boxes))


def out_dir(kind: str) -> Path:
    d = REPO / "runs" / f"track_{kind}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    d.mkdir(parents=True)
    return d


def write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path.relative_to(REPO).as_posix()}")


def dev(run: Path) -> None:
    sha, dirty = git_state()
    _, seqs = load_run(run, "development", labels_val())
    grid = {"no tracker": [{"setting": {}, **evaluate("no tracker", {}, seqs)["overall"]}]}
    choice = {}
    for name in ("greedy IoU", "SORT"):
        rows = []
        for ma, mh, gate in itertools.product(*GRID.values()):
            kw = {"max_age": ma, "min_hits": mh, "iou_gate": gate}
            o = evaluate(name, kw, seqs)["overall"]
            rows.append({"setting": kw, **o})
            print(f"{name:10s} max_age {ma:2d} min_hits {mh} gate {gate}: "
                  f"sum|K-G| {o['sum_abs_K_minus_G']:4d}  sum IDF1 {o['sum_idf1']:.4f}  "
                  f"K {o['K']} G {o['G']}  IDSW {o['idsw']}/{o['idsw_mm']}")
        best = min(rows, key=lambda r: (r["sum_abs_K_minus_G"], -r["sum_idf1"],
                                        r["setting"]["max_age"], r["setting"]["min_hits"],
                                        r["setting"]["iou_gate"]))
        grid[name], choice[name] = rows, best["setting"]
        print(f"chosen for {name}: {best['setting']}")
    write(out_dir("dev") / "grid.json", {"git_sha": sha, "git_dirty": dirty, "run": run.name,
                                         "seeds": list(seqs), "choice": choice, "grid": grid})


def sweep(default: Path, cells: list[Path]) -> None:
    guard(CONFIG)
    sha, dirty = git_state()
    settings = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
    kw0 = {"no tracker": {}, **{n: dict(settings[CONFIG_KEY[n]]) for n in CONFIG_KEY}}
    labels = labels_val()
    m0, seqs0 = load_run(default, "held_out", labels)
    if m0.get("overrides"):
        raise SystemExit(f"{default.name} has overrides: not the default run")
    defaults = {"pan_speed": m0["params"]["pan_speed"],
                "occlusion_frames": m0["params"]["occlusion_frames"],
                "conf": m0["detector"]["conf"]}

    def cell(factor, value, run, seqs, tracker_kw=None):
        res = {}
        for name in TRACKERS:
            kw = {**kw0[name], **(tracker_kw if tracker_kw and name != "no tracker" else {})}
            r = evaluate(name, kw, seqs)
            res[name] = {"setting": kw, "overall": r["overall"],
                         "per_sequence": {s: r[s] for s in r if s != "overall"}}
        o = {n: res[n]["overall"] for n in res}
        print(f"{factor:16s} {value!s:5s} " + "  ".join(
            f"{n}: K-G {o[n]['K_minus_G']:5d} IDF1 {o[n]['idf1']:.3f} MOTA {o[n]['mota']:.3f}"
            for n in res))
        return {"factor": factor, "value": value, "run": run.name,
                "bounces": {s: b for s, (_, _, b) in seqs.items()}, "trackers": res}

    rows = [cell(f, v, default, seqs0) for f, v in defaults.items()]
    for run in cells:
        m, seqs = load_run(run, "held_out", labels)
        (factor, value), = m["overrides"].items()
        rows.append(cell(factor, value, run, seqs))
    for factor, values in SWEEP_TRACKER.items():
        for v in values:
            rows.append(cell(factor, v, default, seqs0, {factor: v}))
    write(out_dir("sweep") / "sweep.json", {"git_sha": sha, "git_dirty": dirty, "settings": kw0,
                                            "defaults": defaults, "default_run": default.name,
                                            "cells": rows})


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "dev":
        dev(Path(argv[1]).resolve())
    elif len(argv) >= 2 and argv[0] == "sweep":
        sweep(Path(argv[1]).resolve(), [Path(a).resolve() for a in argv[2:]])
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
