"""Post-hoc error analysis of the single D2 test evaluation, from saved outputs only.

    python scripts/error_analysis_det.py runs/det_test_<run> runs/det_val_<run>

Post-hoc: made after the single test evaluation. Nothing here changes the model, the
config, the operating point or the reported numbers; it measures what the saved
outputs say about where the errors are.

Inputs, and nothing else: the test evaluation's `predictions.npz` and `eval.json`, the
selected model's val evaluation (same files), and the D2 images and labels on disk.
The model is never run: the script takes no checkpoint, refuses an argument that looks
like one, refuses if any `runs/det_test_*` other than the named one exists, and checks
at the end that Ultralytics was never imported.

At the shipped settings (the passes ran at NMS IoU 0.7; detections are kept at the
config's `conf_operating`, 0.25), with true and false positives from `ap.match` at IoU
0.5 as `evaluate()` takes them:

1. False positives by class (`detection.errors.classify`), per split, and each class's
   per-frame mean, which with the misses adds up to the mean signed count error
   (counted - labelled = FP - FN).
2. Misses: share with a kept detection at IoU [0.1, 0.5) (found, badly placed), share
   touching the frame edge (within 1 px), and the miss rate by box-area tercile, the
   terciles taken over every D2 label so they mean the same in each split.
3. Per split (train, val, test), labels and pixels only: box size and aspect ratio,
   boxes per frame, nearest-neighbour IoU between ground-truth boxes, per-frame DN mean
   and standard deviation, and the mean DN inside labelled boxes minus outside them
   (8-bit rendered DN, uncalibrated, within one frame). Then, on test and on val, the
   Spearman correlation of per-frame AP@0.5 and signed count error with five frame
   properties, all ten reported; "clear" is declared as p < 0.005 (0.05 over the ten
   correlations of a split).
4. Overlays of the 6 test and 2 val frames with the largest absolute count error, a
   contact sheet, and a per-frame CSV of the test frames.

The matched ground truth is not returned by `ap.match`, so the misses come from a
replay of its rule on the kept detections of each frame; the script checks the
replay's true positives equal `ap.match`'s, and that the totals equal eval.json's.
Writes runs/det_posthoc_<utc>/ (gitignored).
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.train import git_state                # noqa: E402
from solar_inspect.detection.ap import ap_from_matches, match            # noqa: E402
from solar_inspect.detection.boxes import area, iou                      # noqa: E402
from solar_inspect.detection.data import ground_truth, splits            # noqa: E402
from solar_inspect.detection.errors import LOCALISATION_IOU, classify    # noqa: E402

FP_CLASSES = ("duplicate", "localisation", "background", "unclassified")
EDGE_PX = 1.0
NEIGHBOUR_IOU = 0.3
CLEAR_P = 0.005
OVERLAYS = {"test": 6, "val": 2}
COLOURS = {"gt": (0, 200, 0), "tp": (40, 110, 255), "duplicate": (255, 140, 0),
           "localisation": (255, 230, 0), "background": (230, 0, 0),
           "unclassified": (255, 255, 255), "fn": (255, 0, 255)}


def guard(argv: list[str]) -> tuple[Path, Path]:
    if len(argv) != 2 or any(a.lower().endswith((".pt", ".pth")) or "weights" in a.lower()
                             for a in argv):
        raise SystemExit("This analysis reads saved outputs only and never runs a model.\n"
                         + __doc__)
    test_dir, val_dir = ((REPO / a).resolve() for a in argv)
    others = sorted(p.name for p in (REPO / "runs").glob("det_test_*") if p != test_dir)
    if others or not test_dir.is_dir():
        raise SystemExit(f"expected exactly one test evaluation, {test_dir.name}; "
                         f"also found {others}. Refusing.")
    for d, split in ((test_dir, "test"), (val_dir, "val")):
        if json.loads((d / "eval.json").read_text(encoding="utf-8"))["split"] != split:
            raise SystemExit(f"{d.name} is not a {split} evaluation")
    return test_dir, val_dir


def replay(boxes, scores, gt) -> tuple[np.ndarray, np.ndarray]:
    """`ap.match`'s rule on one frame: (TP flag per detection in input order, matched
    flag per ground truth)."""
    tp, taken = np.zeros(len(boxes), dtype=bool), np.zeros(len(gt), dtype=bool)
    ious = iou(boxes, gt)
    for k in np.argsort(-scores, kind="stable"):
        if not len(gt):
            break
        cand = np.where(taken, -1.0, ious[k])
        j = int(cand.argmax())
        if cand[j] >= 0.5:
            taken[j] = tp[k] = True
    return tp, taken


def nn_iou(gt: np.ndarray) -> np.ndarray:
    """Per box, its largest IoU with another box of the frame (NaN if it is alone)."""
    if len(gt) < 2:
        return np.full(len(gt), np.nan)
    m = iou(gt, gt)
    np.fill_diagonal(m, -1.0)
    return m.max(axis=1)


def pixels(img_path: Path, gt: np.ndarray) -> tuple[float, float, float]:
    """DN mean, DN std, and mean DN inside the labelled boxes minus outside them."""
    with Image.open(img_path) as im:
        a = np.asarray(im)
    if a.ndim == 3:
        if not (a == a[..., :1]).all():
            raise SystemExit(f"{img_path.name}: channels differ; expected greyscale in RGB")
        a = a[..., 0]
    a = a.astype(float)
    inside = np.zeros(a.shape, dtype=bool)
    for x1, y1, x2, y2 in np.rint(gt).astype(int):
        inside[max(y1, 0):y2, max(x1, 0):x2] = True
    contrast = a[inside].mean() - a[~inside].mean() if inside.any() and (~inside).any() \
        else np.nan
    return a.mean(), a.std(), contrast


def frame_props(images: Path) -> dict:
    names, gts = ground_truth(images)
    nn = [nn_iou(g) for g in gts]
    wh = [np.column_stack([g[:, 2] - g[:, 0], g[:, 3] - g[:, 1]]) for g in gts]
    px = np.array([pixels(images / n, g) for n, g in zip(names, gts)])
    with Image.open(images / names[0]) as im:
        size = im.size
    return {"names": names, "gts": gts, "nn": nn, "wh": wh, "px": px, "size": size}


def q(x) -> list[float]:
    """median, Q1, Q3, ignoring NaN"""
    return [float(v) for v in np.nanpercentile(x, [50, 25, 75])]


def label_stats(fp: dict) -> dict:
    wh = np.concatenate(fp["wh"])
    nn = np.concatenate(fp["nn"])
    n = np.array([len(g) for g in fp["gts"]])
    frame_nn = np.array([np.nanmedian(v) if np.isfinite(v).any() else np.nan for v in fp["nn"]])
    return {
        "frames": len(n), "boxes": int(n.sum()),
        "width_px": q(wh[:, 0]), "height_px": q(wh[:, 1]), "area_px": q(wh[:, 0] * wh[:, 1]),
        "aspect_w_over_h": q(wh[:, 0] / wh[:, 1]),
        "boxes_per_frame": [float(np.median(n)), int(n.min()), int(n.max())],
        "nn_iou_median": float(np.nanmedian(nn)),
        "nn_iou_share_over_0.3": float((nn[np.isfinite(nn)] > NEIGHBOUR_IOU).mean()),
        "frames_with_median_nn_iou_over_0.2": int((frame_nn > 0.2).sum()),
        "dn_mean": q(fp["px"][:, 0]), "dn_std": q(fp["px"][:, 1]),
        "dn_inside_minus_outside": q(fp["px"][:, 2]),
    }


def analyse(eval_dir: Path, fp: dict, conf: float, tercile_edges) -> dict:
    e = json.loads((eval_dir / "eval.json").read_text(encoding="utf-8"))
    p = np.load(eval_dir / "predictions.npz")
    names, n = list(p["names"]), len(p["names"])
    if names != fp["names"]:
        raise SystemExit(f"{eval_dir.name}: image list differs from the labels on disk")
    boxes = [p["boxes"][p["image"] == i] for i in range(n)]
    scores = [p["scores"][p["image"] == i] for i in range(n)]
    gts = [p["gt_boxes"][p["gt_image"] == i] for i in range(n)]
    if not all(g.shape == h.shape and np.allclose(g, h) for g, h in zip(gts, fp["gts"])):
        raise SystemExit(f"{eval_dir.name}: saved ground truth differs from the labels on disk")

    # TP/FP exactly as evaluate() takes them: ap.match over the split at conf 0.001, kept >= conf
    sc, tp_all, n_gt = match(boxes, scores, gts)
    W, H = fp["size"]
    rows, fns = [], []
    for i in range(n):
        keep = scores[i] >= conf
        b, s, g = boxes[i][keep], scores[i][keep], gts[i]
        tp, taken = replay(b, s, g)
        ref = match([b], [s], [g])[1]
        if not np.array_equal(tp[np.argsort(-s, kind="stable")], ref):
            raise SystemExit(f"{names[i]}: replayed matching differs from ap.match")
        cls = classify(b, tp, g)
        _, tpf, _ = match([boxes[i]], [scores[i]], [g])
        miss = g[~taken]
        best = iou(miss, b).max(axis=1, initial=0.0)
        edge = ((miss[:, 0] <= EDGE_PX) | (miss[:, 1] <= EDGE_PX)
                | (miss[:, 2] >= W - EDGE_PX) | (miss[:, 3] >= H - EDGE_PX))
        all_edge = ((g[:, 0] <= EDGE_PX) | (g[:, 1] <= EDGE_PX)
                    | (g[:, 2] >= W - EDGE_PX) | (g[:, 3] >= H - EDGE_PX))
        fns.append({"best_iou": best, "edge": edge, "tercile": np.digitize(area(miss), tercile_edges),
                    "gt_tercile": np.digitize(area(g), tercile_edges), "gt_edge": all_edge})
        wh = fp["wh"][i]
        rows.append({
            "image": names[i], "labelled": len(g), "counted": int(keep.sum()),
            "signed_error": int(keep.sum()) - len(g), "tp": int(tp.sum()),
            **{c: int((cls == c).sum()) for c in FP_CLASSES}, "fn": int((~taken).sum()),
            "fn_found_badly_placed": int(((best >= LOCALISATION_IOU[0])
                                          & (best < LOCALISATION_IOU[1])).sum()),
            "fn_at_edge": int(edge.sum()),
            "ap50_101_point": ap_from_matches(tpf, len(g), "101"),
            "median_box_area_px": float(np.median(wh[:, 0] * wh[:, 1])) if len(g) else np.nan,
            "median_nn_iou": float(np.nanmedian(fp["nn"][i])) if len(g) > 1 else np.nan,
            "median_aspect_w_over_h": float(np.median(wh[:, 0] / wh[:, 1])) if len(g) else np.nan,
            "dn_mean": fp["px"][i, 0], "dn_std": fp["px"][i, 1],
            "dn_inside_minus_outside": fp["px"][i, 2],
            "_det": b, "_cls": cls, "_miss": miss,
        })

    # the totals must be the evaluation's own numbers
    at = e["at_conf"][str(conf)]
    k = sc >= conf
    tot = {c: sum(r[c] for r in rows) for c in ("tp", "fn", "counted", *FP_CLASSES)}
    signed = np.array([r["signed_error"] for r in rows])
    mine = (tot["tp"] / tot["counted"], tot["tp"] / n_gt, float(np.abs(signed).mean()),
            float(signed.mean()))
    ref = (at["precision"], at["recall"], at["count_mae"], at["count_signed_mean"])
    if tot["tp"] != int(tp_all[k].sum()) or not np.allclose(mine, ref, atol=1e-12, rtol=0):
        raise SystemExit(f"{eval_dir.name}: totals {mine} differ from eval.json {ref}")
    if any(r["signed_error"] != sum(r[c] for c in FP_CLASSES) - r["fn"] for r in rows):
        raise SystemExit("signed error is not FP - FN on some frame")

    fp_total = sum(tot[c] for c in FP_CLASSES)
    best = np.concatenate([f["best_iou"] for f in fns])
    edge = np.concatenate([f["edge"] for f in fns])
    ter = np.concatenate([f["tercile"] for f in fns])
    gt_ter = np.concatenate([f["gt_tercile"] for f in fns])
    gt_edge = np.concatenate([f["gt_edge"] for f in fns])
    return {
        "eval_run": eval_dir.name, "split": e["split"], "images": n, "boxes": n_gt,
        "conf": conf, "precision": mine[0], "recall": mine[1], "count_mae": mine[2],
        "signed_mean": mine[3], "counted": tot["counted"], "tp": tot["tp"], "fp": fp_total,
        "fn": tot["fn"],
        "fp_classes": {c: {"count": tot[c], "share": tot[c] / fp_total if fp_total else np.nan,
                           "per_frame_mean": tot[c] / n} for c in FP_CLASSES},
        "fn_per_frame_mean": tot["fn"] / n, "fp_per_frame_mean": fp_total / n,
        "misses": {
            "count": int(len(best)),
            "share_found_badly_placed": float(((best >= LOCALISATION_IOU[0])
                                               & (best < LOCALISATION_IOU[1])).mean()),
            "share_no_detection_over_0.1": float((best < LOCALISATION_IOU[0]).mean()),
            "share_at_edge": float(edge.mean()),
            "miss_rate_at_edge": float(edge.sum() / gt_edge.sum()) if gt_edge.any() else np.nan,
            "miss_rate_interior": float((~edge).sum() / (~gt_edge).sum()),
            "share_of_gt_at_edge": float(gt_edge.mean()),
            "miss_rate_by_area_tercile": [float((ter == t).sum() / (gt_ter == t).sum())
                                          for t in range(3)],
            "gt_by_area_tercile": [int((gt_ter == t).sum()) for t in range(3)],
        },
        "frames": rows,
    }


def correlations(rows: list[dict]) -> dict:
    props = ("median_box_area_px", "labelled", "median_nn_iou", "median_aspect_w_over_h",
             "dn_inside_minus_outside")
    out = {}
    for y in ("ap50_101_point", "signed_error"):
        for x in props:
            a = np.array([r[x] for r in rows], dtype=float)
            b = np.array([r[y] for r in rows], dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            rho, pv = spearmanr(a[ok], b[ok])
            out[f"{y} ~ {x}"] = {"rho": float(rho), "p": float(pv), "n": int(ok.sum()),
                                 "clear": bool(pv < CLEAR_P)}
    m = np.array([[r[x] for x in props] for r in rows], dtype=float)
    ok = np.isfinite(m).all(axis=1)
    out["between_properties"] = {"order": list(props),
                                 "rho": spearmanr(m[ok]).statistic.round(3).tolist()}
    return out


def dashed(d: ImageDraw.ImageDraw, box, colour, dash=3) -> None:
    x1, y1, x2, y2 = box
    for (a, b), (c, e) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                           ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        L = max(abs(c - a), abs(e - b))
        for t in np.arange(0, L, 2 * dash):
            t2 = min(t + dash, L)
            d.line([(a + (c - a) * t / L, b + (e - b) * t / L),
                    (a + (c - a) * t2 / L, b + (e - b) * t2 / L)], fill=colour)


def overlay(img_path: Path, r: dict, gt: np.ndarray, split: str) -> Image.Image:
    with Image.open(img_path) as im:
        base = im.convert("RGB")
    d = ImageDraw.Draw(base)
    for g in gt:
        d.rectangle(list(g), outline=COLOURS["gt"])
    for c in ("tp", *FP_CLASSES):
        for b in r["_det"][r["_cls"] == c]:
            d.rectangle(list(b), outline=COLOURS[c])
    for g in r["_miss"]:
        dashed(d, g, COLOURS["fn"])
    head = (f"{split}  {Path(r['image']).stem.split('_T_')[0]}  labelled {r['labelled']}  "
            f"counted {r['counted']}  ({r['signed_error']:+d})\n"
            f"TP {r['tp']} blue | duplicate {r['duplicate']} orange | localisation "
            f"{r['localisation']} yellow | background {r['background']} red | "
            f"unclassified {r['unclassified']} white | missed {r['fn']} dashed magenta")
    out = Image.new("RGB", (base.width, base.height + 34), "white")
    out.paste(base, (0, 34))
    ImageDraw.Draw(out).multiline_text((4, 3), head, fill="black", spacing=4)
    return out


def main(argv: list[str]) -> int:
    test_dir, val_dir = guard(argv)
    sha, dirty = git_state()
    conf = json.loads((test_dir / "eval.json").read_text(encoding="utf-8"))["conf_operating"]
    dirs = splits()
    props = {s: frame_props(dirs[s]) for s in ("train", "val", "test")}
    all_area = np.concatenate([w[:, 0] * w[:, 1] for s in props.values() for w in s["wh"]])
    edges = np.percentile(all_area, [100 / 3, 200 / 3])
    res = {s: analyse(d, props[s], conf, edges) for s, d in (("test", test_dir), ("val", val_dir))}

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = REPO / "runs" / f"det_posthoc_{run}"
    dest.mkdir(parents=True)
    tiles = []
    for s, k in OVERLAYS.items():
        worst = sorted(res[s]["frames"], key=lambda r: (-abs(r["signed_error"]), r["image"]))[:k]
        for j, r in enumerate(worst, 1):
            gt = props[s]["gts"][props[s]["names"].index(r["image"])]
            im = overlay(dirs[s] / r["image"], r, gt, s)
            im.save(dest / f"{s}_{j}_{Path(r['image']).stem.split('_T_')[0]}.png")
            tiles.append(im)
    tw, th = tiles[0].width // 2, tiles[0].height // 2
    sheet = Image.new("RGB", (4 * tw, 2 * th), "white")
    for j, im in enumerate(tiles):
        sheet.paste(im.resize((tw, th), Image.LANCZOS), ((j % 4) * tw, (j // 4) * th))
    sheet.save(dest / "contact_sheet.png")
    with open(dest / "test_frames.csv", "w", newline="", encoding="utf-8") as f:
        cols = [c for c in res["test"]["frames"][0] if not c.startswith("_")]
        w = csv.DictWriter(f, cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(res["test"]["frames"])

    out = {"run": run, "git_sha": sha, "git_dirty": dirty, "post_hoc": True,
           "conf": conf, "area_tercile_edges_px": edges.tolist(),
           "labels": {s: label_stats(p) for s, p in props.items()},
           **{s: {k: v for k, v in r.items() if k != "frames"} for s, r in res.items()},
           "correlations": {s: correlations(r["frames"]) for s, r in res.items()}}
    (dest / "analysis.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8",
                                        newline="\n")
    if "ultralytics" in sys.modules:
        raise SystemExit("Ultralytics was imported; this analysis must not touch the model")
    print(json.dumps(out, indent=2))
    print(f"\nwrote {dest.relative_to(REPO).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
