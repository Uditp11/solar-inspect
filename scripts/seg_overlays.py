"""Overlays of the declared test tiles, and the post-hoc boundary share of the errors.

    python scripts/seg_overlays.py runs/seg_eval_test_<run>

No model pass: reads the predicted masks the single test evaluation saved
(`predictions.npz`) and its `eval.json`, plus the PV01 test tiles and masks. The
per-tile counts recomputed from the saved masks must equal eval.json's exactly.

**Overlays.** The six tiles configs/seg_eval.yaml declared before the test split
was read (per subset, the lower-median and the lowest U-Net IoU, ties by name),
as eval.json recorded them. Each figure is image | truth | prediction, the last
coloured true positive, false positive and false negative in the config's colours.

**Post-hoc, decides nothing.** Per subset, the share of the U-Net's error pixels
(false positive + false negative) that lie within 3 px of a true mask boundary
(segmentation/evaluate.near_boundary), beside the share of all pixels that do:
whether the remaining error is at array edges or is whole arrays missed or
invented. The 3 px radius was chosen after the test evaluation, with no
measurement behind it. Beside it, the U-Net's IoU and recall per spatially
separate group of test tiles, from eval.json's per-tile counts, to say whether
a subset's error is spread or sits on one roof.

Writes runs/seg_overlays_<utc>/ (gitignored): six PNGs and boundary.json.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.train import git_state   # noqa: E402

RADIUS = 3


def per_group(names: list[str], per_tile: list[dict], link: float) -> list[dict]:
    """U-Net counts summed over each spatially separate group of test tiles (the
    connected components at the split's linking distance that eval.json counted)."""
    from scipy.sparse.csgraph import connected_components

    from solar_inspect.segmentation.metrics import scores
    xy = np.array([[int(v) for v in n.split("_")[1:]] for n in names], float)
    lab = connected_components(np.linalg.norm(xy[:, None] - xy[None], axis=-1) <= link,
                               directed=False)[1]
    out = []
    for g in sorted(set(lab.tolist())):
        idx = np.flatnonzero(lab == g)
        c = np.sum([per_tile[i]["unet_counts"] for i in idx], axis=0)
        s = scores(*c)
        out.append({"tiles": len(idx), "first_tile": names[idx[0]],
                    "tp": int(c[0]), "fp": int(c[1]), "fn": int(c[2]),
                    "unet_iou": s["iou"], "unet_recall": s["recall"]})
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit(__doc__)
    test_dir = (REPO / argv[0]).resolve()
    e = json.loads((test_dir / "eval.json").read_text(encoding="utf-8"))
    if e["split"] != "test":
        raise SystemExit(f"{test_dir.name} is a {e['split']} evaluation, not the test one")
    sha, dirty = git_state()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from solar_inspect.segmentation.data import SUBSETS, load_d3
    from solar_inspect.segmentation.evaluate import near_boundary

    d = load_d3("cpu")
    if d.split_sha256 != e["split_sha256"]:
        raise SystemExit("split differs from the test evaluation's")
    p = np.load(test_dir / "predictions.npz")
    colours = {k: np.array(v, np.uint8) for k, v in e["config"]["overlays"]["colours"].items()}
    pinned = json.loads((REPO / "configs/d3_split.json").read_text(encoding="utf-8"))
    link = pinned["rule"]["link_distance"]

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = REPO / "runs" / f"seg_overlays_{run}"
    dest.mkdir(parents=True)

    boundary, figures = {}, []
    for sub in SUBSETS:
        rows = d.index[("test", sub)].tolist()
        names = [d.keys[i].split("/")[1] for i in rows]
        if names != p[f"{sub}_names"].tolist():
            raise SystemExit(f"{sub}: tile order differs from the saved predictions")
        pred = np.unpackbits(p[f"{sub}_unet"], axis=-1).astype(bool)
        truth = d.masks[rows, 0].numpy().astype(bool)
        for i, rec in enumerate(e["subsets"][sub]["per_tile"]):
            c = [int((pred[i] & truth[i]).sum()), int((pred[i] & ~truth[i]).sum()),
                 int((~pred[i] & truth[i]).sum())]
            if c != rec["unet_counts"]:
                raise SystemExit(f"{sub} {names[i]}: counts {c} differ from eval.json")

        near = np.stack([near_boundary(t, RADIUS) for t in truth])
        fp, fn = pred & ~truth, ~pred & truth
        n_fp, n_fn = int(fp.sum()), int(fn.sum())
        boundary[sub] = {
            "tiles": len(names), "error_pixels": n_fp + n_fn, "fp_pixels": n_fp, "fn_pixels": n_fn,
            "share_of_errors_within_3px": float(((fp | fn) & near).sum() / max(n_fp + n_fn, 1)),
            "share_of_fp_within_3px": float((fp & near).sum() / max(n_fp, 1)),
            "share_of_fn_within_3px": float((fn & near).sum() / max(n_fn, 1)),
            "share_of_all_pixels_within_3px": float(near.mean()),
            "tiles_with_errors_beyond_3px": int(((fp | fn) & ~near).any((1, 2)).sum()),
            "fn_beyond_3px_by_tile": sorted(
                ([names[i], int((fn[i] & ~near[i]).sum())] for i in range(len(names))
                 if (fn[i] & ~near[i]).any()), key=lambda r: -r[1])[:5],
            "per_group": per_group(names, e["subsets"][sub]["per_tile"], link),
        }

        for which, sel in e["subsets"][sub]["overlay_tiles"].items():
            i = names.index(sel["tile"])
            img = d.images[rows[i]].permute(1, 2, 0).numpy()
            col = np.zeros((*truth[i].shape, 3), np.uint8)
            col[pred[i] & truth[i]] = colours["true_positive"]
            col[fp[i]] = colours["false_positive"]
            col[fn[i]] = colours["false_negative"]
            fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.9))
            for a, im, t in zip(ax, (img, truth[i], col), ("image", "truth", "U-Net prediction")):
                a.imshow(im, cmap="gray" if im.ndim == 2 else None, interpolation="nearest")
                a.set_title(t, fontsize=10)
                a.set_axis_off()
            fig.legend(handles=[Patch(color=colours[k] / 255, label=k.replace("_", " "))
                                for k in ("true_positive", "false_positive", "false_negative")],
                       loc="lower center", ncol=3, frameon=False, fontsize=9)
            fig.suptitle(f"PV01 {sub}, test, {which} U-Net IoU: {sel['tile']}, "
                         f"IoU {sel['unet_iou']:.4f}", fontsize=10)
            fig.tight_layout(rect=(0, 0.07, 1, 1))
            path = dest / f"{sub}_{which}_{sel['tile']}.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            figures.append({"subset": sub, "which": which, **sel,
                            "png": path.relative_to(REPO).as_posix()})

    out = {"run": run, "git_sha": sha, "git_dirty": dirty, "post_hoc": True,
           "test_run": test_dir.relative_to(REPO).as_posix(), "radius_px": RADIUS,
           "boundary": boundary, "figures": figures}
    (dest / "boundary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8",
                                        newline="\n")
    print(f"post-hoc, test, U-Net error pixels within {RADIUS} px of a true boundary:")
    for s, b in boundary.items():
        print(f"  {s:<13} errors {b['share_of_errors_within_3px']:.3f} (FP "
              f"{b['share_of_fp_within_3px']:.3f}, FN {b['share_of_fn_within_3px']:.3f}); "
              f"all pixels {b['share_of_all_pixels_within_3px']:.3f}; {b['fp_pixels']} FP, "
              f"{b['fn_pixels']} FN px; tiles with error beyond {RADIUS} px: "
              f"{b['tiles_with_errors_beyond_3px']} of {b['tiles']}; largest FN beyond: "
              f"{b['fn_beyond_3px_by_tile']}")
        for g in b["per_group"]:
            print(f"    group of {g['tiles']:>2} tiles from {g['first_tile']}: IoU "
                  f"{g['unet_iou']:.4f}, recall {g['unet_recall']:.4f}, FN {g['fn']} px")
    for f in figures:
        print(f"  {f['png']}")
    print(f"\nwrote {dest.relative_to(REPO).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
