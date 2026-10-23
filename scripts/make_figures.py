"""The README's figures, written to docs/figures/, from outputs already under runs/.

    python scripts/make_figures.py

No model runs and no metric is computed, except for the tracking GIF, which
re-runs the deterministic tracker on one held-out D5 sequence and must reproduce
that sequence's recorded numbers before anything is written. Re-rendered plots
use the same file's data as the original; composites only crop, scale and stack
existing PNGs. Each output is a PNG about 1,200 px wide, at most 600 KB, and the
folder at most 10 MB; the script refuses otherwise.

Each figure, its source and its selection rule:

- seg_median_overlays.png: the three median test overlays configs/seg_eval.yaml
  declared before the test split was read (per subset, the lower-median U-Net
  IoU, ties by name), stacked Brick, FlatConcrete, SteelTile, from the ledger's
  overlay run seg_overlays_20261023T153449Z.
- seg_fc_bright_roof.png, **post-hoc**: on FlatConcrete test, the spatially
  separate tile group of 45 tiles (connected components at the split's linking
  distance, as scripts/seg_overlays.py counts them); in it, the tile with the most
  false-negative pixels by eval.json's per-tile counts, ties by name. Image | truth
  | prediction from the saved masks of seg_eval_test_20261023T153146Z, in
  seg_eval.yaml's colours.
- seg_transfer.png: top, the declared middle D2 val overlay ((n - 1) // 2 by file
  name) of the thermal check seg_thermal_20261023T153719Z; below, the declared
  median FlatConcrete overlay of the greyscale control seg_grey_20261023T170737Z.
  Stacked rather than side by side, so that each keeps legible text at 1,200 px.
- det_test_frames.png: from the post-hoc renders of det_posthoc_20261022T214418Z
  (the six test frames with the largest absolute count error), the first steep
  one (median box aspect w/h below 1.1, by its test_frames.csv) and the first
  not steep one, in the run's own order, side by side.
- det_count_vs_conf.png: the val count-error curve of det_oppoint_20261022T193813Z
  re-rendered from its operating_point.json grid.
- track_flyover.gif: D5 held-out seed 100 of the default flyover run named in
  track_sweep_20261022T235853Z/sweep.json, SORT at configs/tracking.yaml, every
  one of its 60 frames at 6 frames per second, the tracker's reported boxes with
  their ids. Before writing, SORT's ids, ID switches and count error on seed 100,
  scored by scripts/track_flyover.py's own functions, must equal that sweep's
  recorded per-sequence values; otherwise no GIF is written.
- track_sweep.png: from track_sweep_20261022T235853Z/sweep.json, the pan-speed
  cells: count error (K - G) / G against pan speed, greedy IoU and SORT (the
  no-tracker line, near 30, is left out of this one panel; it is in the README).
- cls_confusion_test.png, cls_reliability_test.png: copies of the PNGs of the
  single test pass cls_final_test_20260827T205903Z.
- d4_slopes.png: per plant, every inverter's AC slope on irradiation with its 95%
  Newey-West interval, ordered by slope, the plant median as a line, from
  d4_rank_20261021T180327Z/rank.json (the run the README's Bonferroni counts come
  from). Derived statistics only, no D4 row.
- d4_acf.png: a copy of d4_diag_20261021T162706Z/acf.png.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

RUNS = REPO / "runs"
OUT = REPO / "docs" / "figures"
WIDTH = 1200
MAX_PNG, MAX_GIF, MAX_ALL = 600 * 1024, 4 * 1024 * 1024, 10 * 1024 * 1024


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as p
    return p


def stack(paths: list[Path], out: Path, gap: int = 12) -> None:
    """Scale each PNG to WIDTH and stack them top to bottom on white."""
    from PIL import Image
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        ims.append(im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS))
    canvas = Image.new("RGB", (WIDTH, sum(i.height for i in ims) + gap * (len(ims) - 1)), "white")
    y = 0
    for im in ims:
        canvas.paste(im, (0, y))
        y += im.height + gap
    # ponytail: 256-colour palette keeps the three-overlay stack under 600 KB; the overlay
    # colours are flat, only the photo panels lose a little shading.
    canvas.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(out, optimize=True)


def seg_median_overlays() -> None:
    d = RUNS / "seg_overlays_20261023T153449Z"
    figs = json.loads((d / "boundary.json").read_text(encoding="utf-8"))["figures"]
    paths = [REPO / f["png"] for s in ("Brick", "FlatConcrete", "SteelTile")
             for f in figs if f["subset"] == s and f["which"] == "median"]
    assert len(paths) == 3
    stack(paths, OUT / "seg_median_overlays.png")


def seg_fc_bright_roof() -> None:
    from scipy.sparse.csgraph import connected_components

    from solar_inspect.segmentation.data import load_d3
    test_dir = RUNS / "seg_eval_test_20261023T153146Z"
    e = json.loads((test_dir / "eval.json").read_text(encoding="utf-8"))
    p = np.load(test_dir / "predictions.npz")
    link = json.loads((REPO / "configs/d3_split.json").read_text(encoding="utf-8"))["rule"]["link_distance"]
    sub = "FlatConcrete"
    names = p[f"{sub}_names"].tolist()
    per_tile = e["subsets"][sub]["per_tile"]
    xy = np.array([[int(v) for v in n.split("_")[1:]] for n in names], float)
    lab = connected_components(np.linalg.norm(xy[:, None] - xy[None], axis=-1) <= link,
                               directed=False)[1]
    groups = [np.flatnonzero(lab == g) for g in set(lab.tolist())]
    roof, = [g for g in groups if len(g) == 45]
    i = min(roof, key=lambda k: (-per_tile[k]["unet_counts"][2], names[k]))

    d = load_d3("cpu")
    if d.split_sha256 != e["split_sha256"]:
        raise SystemExit("split differs from the test evaluation's")
    rows = d.index[("test", sub)].tolist()
    assert [d.keys[r].split("/")[1] for r in rows] == names
    pred = np.unpackbits(p[f"{sub}_unet"], axis=-1).astype(bool)[i]
    truth = d.masks[rows[i], 0].numpy().astype(bool)
    c = [int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())]
    assert c == per_tile[i]["unet_counts"], "saved mask differs from eval.json"
    colours = {k: np.array(v, np.uint8) for k, v in e["config"]["overlays"]["colours"].items()}
    col = np.zeros((*truth.shape, 3), np.uint8)
    col[pred & truth] = colours["true_positive"]
    col[pred & ~truth] = colours["false_positive"]
    col[~pred & truth] = colours["false_negative"]
    tp, fp, fn = c
    iou = tp / (tp + fp + fn)

    from matplotlib.patches import Patch
    p_ = plt()
    fig, ax = p_.subplots(1, 3, figsize=(10, 3.9))
    for a, im, t in zip(ax, (d.images[rows[i]].permute(1, 2, 0).numpy(), truth, col),
                        ("image", "truth", "U-Net prediction")):
        a.imshow(im, cmap="gray" if im.ndim == 2 else None, interpolation="nearest")
        a.set_title(t, fontsize=10)
        a.set_axis_off()
    fig.legend(handles=[Patch(color=colours[k] / 255, label=k.replace("_", " "))
                        for k in ("true_positive", "false_positive", "false_negative")],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"Post-hoc. PV01 FlatConcrete test, the 45-tile roof: tile with the most missed "
                 f"pixels, {names[i]}, IoU {iou:.4f}, {fn:,} FN px", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "seg_fc_bright_roof.png", dpi=WIDTH / 10)
    p_.close(fig)
    print(f"  bright roof tile {names[i]}: IoU {iou:.4f}, FN {fn}")


def seg_transfer() -> None:
    stack([RUNS / "seg_thermal_20261023T153719Z/d2_val_middle.png",
           RUNS / "seg_grey_20261023T170737Z/grey_FlatConcrete_median.png"],
          OUT / "seg_transfer.png", gap=24)


def det_test_frames() -> None:
    from PIL import Image, ImageDraw, ImageFont
    d = RUNS / "det_posthoc_20261022T214418Z"
    with open(d / "test_frames.csv", newline="", encoding="utf-8") as f:
        aspect = {r["image"].split("_T_JPG")[0]: float(r["median_aspect_w_over_h"])
                  for r in csv.DictReader(f)}
    renders = sorted(d.glob("test_*.png"), key=lambda q: int(q.name.split("_")[1]))
    stem = lambda q: q.stem.split("_", 2)[2]                                     # noqa: E731
    steep = next(q for q in renders if aspect[stem(q)] < 1.1)
    flat = next(q for q in renders if aspect[stem(q)] >= 1.1)
    a, b = Image.open(steep).convert("RGB"), Image.open(flat).convert("RGB")
    gap, foot = 10, 30
    canvas = Image.new("RGB", (a.width + gap + b.width, max(a.height, b.height) + foot), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + gap, 0))
    font = ImageFont.load_default(size=15)
    ImageDraw.Draw(canvas).text(
        (6, canvas.height - foot + 7),
        f"left: steep rows (median box w/h {aspect[stem(steep)]:.2f}); right: rows across the frame "
        f"(w/h {aspect[stem(flat)]:.2f}). Green: labelled box; other colours as in each header.",
        fill="black", font=font)
    canvas.save(OUT / "det_test_frames.png", optimize=True)
    print(f"  steep {stem(steep)}, not steep {stem(flat)}")


def det_count_vs_conf() -> None:
    o = json.loads((RUNS / "det_oppoint_20261022T193813Z/operating_point.json").read_text(encoding="utf-8"))
    conf = [r["conf"] for r in o["grid"]]
    p_ = plt()
    fig, ax = p_.subplots(figsize=(8, 4.2))
    ax.plot(conf, [r["count_mae"] for r in o["grid"]], "o-", label="count MAE")
    ax.plot(conf, [r["count_signed_mean"] for r in o["grid"]], "s--", label="signed mean (pred - GT)")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.axvline(o["conf_operating"], color="grey", linestyle=":",
               label=f"operating conf {o['conf_operating']} (lowest MAE)")
    ax.set_xlabel("detector confidence threshold")
    ax.set_ylabel("panels per frame")
    ax.set_title(f"D2 val, {o['images']} frames, {o['boxes']:,} boxes, NMS IoU {o['pass']['iou']}")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "det_count_vs_conf.png", dpi=WIDTH / 8)
    p_.close(fig)


def track_flyover() -> None:
    import cv2
    import yaml
    from PIL import Image

    import track_flyover as tf
    from solar_inspect.detection.data import ground_truth, splits
    from solar_inspect.tracking.flyover import FlyoverParams, make_sequence
    seed = 100
    sweep = json.loads((RUNS / "track_sweep_20261022T235853Z/sweep.json").read_text(encoding="utf-8"))
    cell, = [c for c in sweep["cells"] if c["factor"] == "pan_speed"
             and c["run"] == sweep["default_run"]]
    recorded = cell["trackers"]["SORT"]["per_sequence"][str(seed)]
    kw = dict(yaml.safe_load((REPO / tf.CONFIG).read_text(encoding="utf-8"))["sort"])
    assert kw == sweep["settings"]["SORT"], "configs/tracking.yaml differs from the sweep's"

    run = RUNS / sweep["default_run"]
    labels = tf.labels_val()
    _, seqs = tf.load_run(run, "held_out", labels)
    dets, truth, _ = seqs[seed]
    frames = tf.track("SORT", kw, dets, truth)
    got = tf.score({str(seed): frames})[str(seed)]
    keys = ("K", "idsw", "K_minus_G")
    if any(got[k] != recorded[k] for k in keys):
        raise SystemExit(f"seed {seed} does not reproduce: {[(k, got[k], recorded[k]) for k in keys]}")
    print(f"  seed {seed} reproduces: " + ", ".join(f"{k} {got[k]}" for k in keys))

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
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

    scale = 1.5
    palette = (np.array(p_colors(), float) * 255).astype(int)
    out = []
    for t, fr in enumerate(frames):
        img = cv2.resize(seq["frames"][t], None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        occ = seq["occluder"][t]
        if occ.any():
            cv2.polylines(img, [np.round((occ - 0.5) * scale).astype(np.int32)], True, (0, 255, 255), 2)
        for i, b in zip(fr.hyp_ids, fr.hyp_boxes):
            colour = tuple(int(v) for v in palette[int(i) % len(palette)][::-1])
            p1, p2 = np.round(b[:2] * scale).astype(int), np.round(b[2:] * scale - 1).astype(int)
            cv2.rectangle(img, tuple(p1), tuple(p2), colour, 2)
            cv2.putText(img, str(int(i)), (int(p1[0]) + 3, int(p1[1]) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(img, f"synthetic D5, held-out seed {seed}, SORT   frame {t + 1}/{len(frames)}",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    pal = out[len(out) // 2].quantize(colors=48)
    out = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in out]
    out[0].save(OUT / "track_flyover.gif", save_all=True, append_images=out[1:],
                duration=1000 // 6, loop=0, optimize=True)


def p_colors() -> list:
    import matplotlib
    return [matplotlib.colors.to_rgb(c) for c in matplotlib.colormaps["tab20"].colors]


def track_sweep() -> None:
    s = json.loads((RUNS / "track_sweep_20261022T235853Z/sweep.json").read_text(encoding="utf-8"))
    cells = sorted((c for c in s["cells"] if c["factor"] == "pan_speed"), key=lambda c: c["value"])
    xs = [c["value"] for c in cells]
    p_ = plt()
    fig, ax = p_.subplots(figsize=(8, 4.2))
    for name, colour, marker in (("greedy IoU", "#eb6834", "s"), ("SORT", "#1baf7a", "^")):
        ax.plot(xs, [c["trackers"][name]["overall"]["rel_count_error"] for c in cells],
                color=colour, marker=marker, markersize=7, linewidth=2, label=name)
    ax.set_xticks(xs)
    ax.set_ylim(0, None)
    ax.set_xlabel("pan speed (px per frame)")
    ax.set_ylabel("(K - G) / G, surplus ids per true module")
    ax.set_title("D5 held-out sweep, 20 synthetic sequences per point (other factors at default)")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "track_sweep.png", dpi=WIDTH / 8)
    p_.close(fig)


def copies() -> None:
    for src, dst in (("cls_final_test_20260827T205903Z/confusion_test.png", "cls_confusion_test.png"),
                     ("cls_final_test_20260827T205903Z/reliability_test.png", "cls_reliability_test.png"),
                     ("d4_diag_20261021T162706Z/acf.png", "d4_acf.png")):
        shutil.copyfile(RUNS / src, OUT / dst)


def d4_slopes() -> None:
    r = json.loads((RUNS / "d4_rank_20261021T180327Z/rank.json").read_text(encoding="utf-8"))
    p_ = plt()
    fig, axes = p_.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (plant, d) in zip(axes, sorted(r["plants"].items())):
        inv = sorted(d["inverters"].values(), key=lambda v: v["perf_slope"])
        x = np.arange(1, len(inv) + 1)
        y = np.array([v["perf_slope"] for v in inv])
        lo, hi = np.array([v["perf_ci_low"] for v in inv]), np.array([v["perf_ci_high"] for v in inv])
        out = np.array([not v["ci_contains_plant_median"] for v in inv])
        for m, colour, label in ((~out, "#7f7f7f", "95% interval contains the median"),
                                 (out, "#d55e00", "95% interval excludes the median")):
            ax.errorbar(x[m], y[m], yerr=[y[m] - lo[m], hi[m] - y[m]], fmt="o", color=colour,
                        capsize=2, markersize=4, label=label)
        ax.axhline(d["summary"]["median_slope"], color="black", linestyle="--", linewidth=1,
                   label=f"plant median {d['summary']['median_slope']:.0f}")
        ax.set_title(f"Plant {plant}: {int(out.sum())} of {len(inv)} exclude the median")
        ax.set_xlabel("inverter, ordered by slope (1 = lowest)")
        ax.set_xticks(x[::3])
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_ylabel("AC kW per kW/m² (slope, 95% Newey-West)")
    fig.suptitle("D4, per-inverter slope of AC power on irradiation, producing counted intervals "
                 "(derived statistics, in-sample)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "d4_slopes.png", dpi=WIDTH / 10)
    p_.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (seg_median_overlays, seg_fc_bright_roof, seg_transfer, det_test_frames,
              det_count_vs_conf, track_flyover, track_sweep, copies, d4_slopes):
        print(f.__name__)
        f()
    total = 0
    for q in sorted(OUT.iterdir()):
        n = q.stat().st_size
        total += n
        from PIL import Image
        w, h = Image.open(q).size
        print(f"  {q.name:28s} {w:5d} x {h:<5d} {n / 1024:7.0f} KB")
        if n > (MAX_GIF if q.suffix == ".gif" else MAX_PNG):
            raise SystemExit(f"{q.name} is over its size limit")
    print(f"  total {total / 1024 / 1024:.2f} MB")
    if total > MAX_ALL:
        raise SystemExit("figures are over 10 MB together")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
