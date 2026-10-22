"""Generate the D5 flyover sequences and cache the shipped detector's detections on them.

    python scripts/make_flyover.py runs/det_<run>/weights/best.pt

Reads `configs/flyover.yaml`, which must be tracked and unmodified: the config is
declared before the sequences exist. Source frames come from the D2 val split only
(`detection.data.splits()` keeps val off the test tree; any other `source_split` is
refused). The checkpoint must have the SHA-256 pinned in `configs/det_yolo.yaml`, and
detections are kept at that config's `conf_operating`, `iou` and `max_det`.

Every sequence of the config, development and held-out, is generated; nothing is
scored here. Writes runs/flyover_<utc>/ (gitignored):

    seed_<NNN>.npz    ground truth (gt_frame, gt_id, gt_box), occlusions (occ_frame,
                      occ_id), detections (det_frame, det_box, det_score), and the
                      camera path (H, centre, theta, scale, occluder). Frames are not
                      stored: the generator is deterministic and re-renders them.
    manifest.json     config, detector settings, git and checkpoint hashes, and per
                      sequence: frames, ground-truth boxes, unique ids, detections,
                      the frames each id is occluded.
    overlay_seed<N>.png  six frames of the first development sequence: ground truth
                      with ids (green), detections (red), occluder outline (yellow).
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.train import git_state                  # noqa: E402
from solar_inspect.detection.data import check_checkpoint, ground_truth, splits  # noqa: E402
from solar_inspect.tracking.flyover import FlyoverParams, make_sequence    # noqa: E402

CONFIG = "configs/flyover.yaml"


def guard(*paths: str) -> None:
    for f in paths:
        if subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", f],
                          capture_output=True).returncode != 0:
            raise SystemExit(f"{f} is not tracked. Commit it before generating.")
        if subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", f],
                          capture_output=True, text=True).stdout.strip():
            raise SystemExit(f"{f} is modified. The committed version is the declaration.")


def params_from(cfg: dict) -> FlyoverParams:
    kw = {f.name: cfg[f.name] for f in fields(FlyoverParams)}       # KeyError if missing
    kw["window"], kw["scale_range"] = tuple(kw["window"]), tuple(kw["scale_range"])
    return FlyoverParams(**kw)


def runs_of(frames: np.ndarray) -> list[int]:
    """Lengths of the runs of consecutive frame indices."""
    return [len(list(g)) for _, g in itertools.groupby(enumerate(np.sort(frames)),
                                                        lambda x: x[1] - x[0])]


def overlay(seq: dict, det: dict, frames: list[int]) -> np.ndarray:
    tiles = []
    for t in frames:
        img = seq["frames"][t].copy()
        occ = seq["occluder"][t]
        if occ.any():
            cv2.polylines(img, [np.round(occ - 0.5).astype(np.int32)], True, (0, 255, 255), 1)
        for b in det["det_box"][det["det_frame"] == t]:
            cv2.rectangle(img, tuple(np.round(b[:2]).astype(int)),
                          tuple(np.round(b[2:] - 1).astype(int)), (0, 0, 255), 1)
        m = seq["gt_frame"] == t
        for i, b in zip(seq["gt_id"][m], seq["gt_box"][m]):
            p1, p2 = np.round(b[:2]).astype(int), np.round(b[2:] - 1).astype(int)
            cv2.rectangle(img, tuple(p1), tuple(p2), (0, 255, 0), 1)
            cv2.putText(img, str(i), (int(p1[0]) + 1, int(p1[1]) + 9),
                        cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)
        cv2.putText(img, f"t={t}", (4, 14), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        tiles.append(np.pad(img, ((2, 2), (2, 2), (0, 0))))
    return np.hstack(tiles)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit(__doc__)
    weights = Path(argv[0]).resolve()
    cfg = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
    det_cfg_path = cfg["detector"]["config"]
    guard(CONFIG, det_cfg_path)
    sha, dirty = git_state()
    if cfg["source_split"] != "val":
        raise SystemExit(f"source_split is {cfg['source_split']!r}: D5 is built from val only.")
    det_cfg = yaml.safe_load((REPO / det_cfg_path).read_text(encoding="utf-8"))
    weights_sha = check_checkpoint(weights, det_cfg.get("checkpoint_sha256"))
    print(f"checkpoint {weights.name} SHA-256 {weights_sha[:12]} = {det_cfg_path} checkpoint_sha256")

    p = params_from(cfg)
    val_dir = splits()["val"]
    names, boxes = ground_truth(val_dir)
    order = np.random.default_rng(cfg["frame_assignment_seed"]).permutation(len(names))
    dev, held = cfg["seeds"]["development"], cfg["seeds"]["held_out"]
    source = {s: names[order[i]] for i, s in enumerate(list(dev) + list(held))}
    assert len(set(source.values())) == len(source), "a source frame is shared"

    from ultralytics import YOLO, __version__ as ul_version
    model = YOLO(str(weights))
    pred_kw = dict(imgsz=cfg["detector"]["imgsz"], conf=det_cfg["conf_operating"],
                   iou=det_cfg["iou"], max_det=det_cfg["max_det"],
                   device=cfg["detector"]["device"], verbose=False)

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO / "runs" / f"flyover_{run}"
    out.mkdir(parents=True)
    summary = {}
    for seed, name in source.items():
        src = cv2.imread(str(val_dir / name), cv2.IMREAD_COLOR)
        seq = make_sequence(src, boxes[names.index(name)], p, seed)
        res = model.predict(list(seq["frames"]), **pred_kw)
        assert all(r.orig_shape == (p.window[1], p.window[0]) for r in res)
        det = {"det_frame": np.concatenate([np.full(len(r.boxes), t) for t, r in enumerate(res)]
                                           ).astype(np.int64),
               "det_box": np.concatenate([r.boxes.xyxy.cpu().numpy() for r in res]
                                         ).reshape(-1, 4).astype(float),
               "det_score": np.concatenate([r.boxes.conf.cpu().numpy() for r in res]).astype(float)}
        np.savez_compressed(out / f"seed_{seed:03d}.npz", **det,
                            **{k: v for k, v in seq.items() if k != "frames"},
                            source_frame=name, seed=seed)
        occluded = {int(i): seq["occ_frame"][seq["occ_id"] == i].tolist()
                    for i in np.unique(seq["occ_id"])}
        summary[seed] = {
            "set": "development" if seed in dev else "held_out", "source_frame": name,
            "labels_in_source": int(len(boxes[names.index(name)])),
            "frames": p.frames, "gt_boxes": int(len(seq["gt_id"])),
            "unique_ids": int(len(np.unique(seq["gt_id"]))),
            "detections": int(len(det["det_frame"])),
            "ids_occluded": len(occluded),
            "occlusion_run_lengths": sorted(r for f in occluded.values()
                                            for r in runs_of(np.array(f))),
            "occluded_frames": occluded,
            "bounces": int(seq["bounces"]),
            "mean_speed_px": float(np.linalg.norm(np.diff(seq["centre"], axis=0), axis=1).mean()),
        }
        s = summary[seed]
        print(f"seed {seed:3d} {s['set'][:4]}  {s['frames']} frames, {s['gt_boxes']} GT boxes, "
              f"{s['unique_ids']} ids, {s['detections']} detections, {s['ids_occluded']} ids "
              f"occluded, {s['bounces']} bounces, speed {s['mean_speed_px']:.2f}")
        if seed == dev[0]:
            strip = overlay(seq, det, np.linspace(0, p.frames - 1, 6).round().astype(int).tolist())
            cv2.imwrite(str(out / f"overlay_seed{seed}.png"), strip)

    manifest = {"run": run, "git_sha": sha, "git_dirty": dirty, "config_path": CONFIG,
                "config": cfg, "params": asdict(p), "detector_config": det_cfg_path,
                "detector": {**{k: v for k, v in pred_kw.items() if k != "verbose"},
                             "weights": weights.relative_to(REPO).as_posix(),
                             "weights_sha256": weights_sha, "ultralytics": ul_version},
                "sequences": {str(k): v for k, v in summary.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
                                       newline="\n")
    print(f"wrote {out.relative_to(REPO).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
