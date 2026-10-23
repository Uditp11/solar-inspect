"""Figures for the D5 held-out sweep, written next to its sweep.json (runs/ only).

    python scripts/plot_track_sweep.py runs/track_sweep_<run>

- sweep.png: count error (K - G) / G and IDF1 against each swept factor, one panel per
  factor, the three trackers in each panel. Count error is on a log axis: no tracker
  sits near 30, the trackers near 0.5.
- overlay_seed<N>.png: six frames of the first held-out sequence of the default run,
  re-rendered by the generator, with SORT's reported boxes and track ids (red), the
  visible ground truth (green) and the band outline (yellow).
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
import numpy as np                                                          # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.detection.data import ground_truth, splits              # noqa: E402
from solar_inspect.tracking.flyover import FlyoverParams, make_sequence    # noqa: E402
from solar_inspect.tracking.sort import Sort                               # noqa: E402

STYLE = {"no tracker": ("#2a78d6", "o"), "greedy IoU": ("#eb6834", "s"), "SORT": ("#1baf7a", "^")}
LABEL = {"pan_speed": "pan speed (px/frame)", "occlusion_frames": "occlusion (frames)",
         "conf": "detector confidence", "max_age": "max_age", "iou_gate": "IoU gate"}


def sweep_figure(sweep: dict, out: Path) -> None:
    factors = list(LABEL)
    fig, axes = plt.subplots(2, len(factors), figsize=(16, 6.2), sharey="row")
    for c, factor in enumerate(factors):
        cells = sorted((x for x in sweep["cells"] if x["factor"] == factor),
                       key=lambda x: x["value"])
        xs = [x["value"] for x in cells]
        for r, key in enumerate(("rel_count_error", "idf1")):
            ax = axes[r, c]
            for name, (colour, marker) in STYLE.items():
                ax.plot(xs, [x["trackers"][name]["overall"][key] for x in cells], color=colour,
                        marker=marker, markersize=7, linewidth=2, label=name)
            ax.set_xticks(xs)
            ax.grid(alpha=0.25, linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 1:
                ax.set_xlabel(LABEL[factor])
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("(K - G) / G, held-out (log)")
    axes[1, 0].set_ylabel("IDF1, held-out")
    axes[1, 0].set_ylim(0, 1)
    axes[0, -1].legend(frameon=False, loc="center right")
    fig.suptitle("D5 held-out sweep, 20 synthetic sequences: count error and IDF1 per factor "
                 "(others at default)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out.relative_to(REPO).as_posix()}")


def overlay_strip(sweep: dict, out_dir: Path) -> None:
    run = REPO / "runs" / sweep["default_run"]
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    seed = manifest["config"]["seeds"]["held_out"][0]
    p = manifest["params"]
    params = FlyoverParams(**{f.name: tuple(p[f.name]) if isinstance(p[f.name], list) else p[f.name]
                              for f in fields(FlyoverParams)})
    z = np.load(run / f"seed_{seed:03d}.npz")
    val_dir = splits()["val"]
    names, boxes = ground_truth(val_dir)
    name = str(z["source_frame"])
    seq = make_sequence(cv2.imread(str(val_dir / name), cv2.IMREAD_COLOR),
                        boxes[names.index(name)], params, seed)
    assert np.array_equal(seq["gt_box"], z["gt_box"]), "re-render differs from the cache"
    sort = Sort(**sweep["settings"]["SORT"])
    shown = np.linspace(0, params.frames - 1, 6).round().astype(int).tolist()
    tiles = []
    for t in range(params.frames):
        dets = z["det_box"][z["det_frame"] == t]
        ids = sort.update(dets)
        if t not in shown:
            continue
        img = seq["frames"][t].copy()
        occ = seq["occluder"][t]
        if occ.any():
            cv2.polylines(img, [np.round(occ - 0.5).astype(np.int32)], True, (0, 255, 255), 1)
        for b in seq["gt_box"][seq["gt_frame"] == t]:
            cv2.rectangle(img, tuple(np.round(b[:2]).astype(int)),
                          tuple(np.round(b[2:] - 1).astype(int)), (0, 200, 0), 1)
        for i, b in zip(ids, dets):
            if i <= 0:
                continue
            p1, p2 = np.round(b[:2]).astype(int), np.round(b[2:] - 1).astype(int)
            cv2.rectangle(img, tuple(p1), tuple(p2), (0, 0, 255), 1)
            cv2.putText(img, str(i), (int(p1[0]) + 1, int(p1[1]) + 9),
                        cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 1)
        cv2.putText(img, f"t={t}", (4, 14), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        tiles.append(np.pad(img, ((2, 2), (2, 2), (0, 0))))
    out = out_dir / f"overlay_seed{seed}.png"
    cv2.imwrite(str(out), np.hstack(tiles))
    print(f"wrote {out.relative_to(REPO).as_posix()}")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit(__doc__)
    out_dir = Path(argv[0]).resolve()
    sweep = json.loads((out_dir / "sweep.json").read_text(encoding="utf-8"))
    sweep_figure(sweep, out_dir / "sweep.png")
    overlay_strip(sweep, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
