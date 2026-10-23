"""The PV01 U-Net on greyscaled PV01 val tiles: does removing colour alone collapse it?

    python scripts/seg_grey_control.py          # the control, once
    python scripts/seg_grey_control.py --rgb    # sanity check, writes nothing

Rules in configs/seg_grey_control.yaml, committed before the first run. The model
is the one configs/seg_eval.yaml pins (same SHA-256), unchanged, at threshold 0.5.
PV01 **val** tiles only.

Each tile is converted to BT.601 luminance (segmentation/evaluate.py's
`luminance`), replicated to three channels, then put through the thermal
check's input path: / 255, standardised with the PV01 train mean and std stored
in the checkpoint. Per subset, all pixels counted together: IoU, predicted
positive fraction, true panel share, recall and precision (metrics.py's rules),
beside the selected epoch's colour-input val IoU from the training log, and the
declared reading.

`--rgb` switches the conversion off and must reproduce the selected epoch's val
IoUs exactly; it asserts that and writes nothing.

Writes runs/seg_grey_<utc>/ (gitignored): grey.json and one overlay per subset.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.train import git_state   # noqa: E402
from solar_inspect.detection.data import check_checkpoint   # noqa: E402

CONFIG = "configs/seg_grey_control.yaml"


def reading(iou: dict[str, float], lo: float, hi: float) -> str:
    if all(v < lo for v in iou.values()):
        return "colour is enough to explain the collapse"
    if all(v >= hi for v in iou.values()):
        return "colour does not explain it"
    return "partial"


def main(argv: list[str]) -> int:
    if argv not in ([], ["--rgb"]):
        raise SystemExit(__doc__)
    rgb_only = argv == ["--rgb"]
    sha, dirty = git_state()
    cfg = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
    if cfg["split"] != "val" or cfg["threshold"] != 0.5:
        raise SystemExit("this control runs on PV01 val at threshold 0.5 only")
    eval_cfg = yaml.safe_load((REPO / "configs/seg_eval.yaml").read_text(encoding="utf-8"))
    if cfg["checkpoint_sha256"] != eval_cfg["checkpoint_sha256"]:
        raise SystemExit("not the model the segmentation evaluation pins")
    run_dir = REPO / cfg["run"]
    weights = run_dir / cfg["checkpoint"]
    check_checkpoint(weights, cfg["checkpoint_sha256"])
    if cfg["conversion"]["luminance"] != [0.299, 0.587, 0.114]:
        raise SystemExit("luminance() implements BT.601 weights only")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.patches import Patch

    from solar_inspect.segmentation.data import SUBSETS, load_d3
    from solar_inspect.segmentation.evaluate import luminance, predict, tile_counts
    from solar_inspect.segmentation.metrics import scores
    from solar_inspect.segmentation.unet import UNet

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    d = load_d3()
    dev = d.images.device
    state = torch.load(weights, map_location=dev)
    if state["epoch"] != cfg["selected_epoch"] or state["split_sha256"] != d.split_sha256:
        raise SystemExit(f"checkpoint is epoch {state['epoch']}, split {state['split_sha256'][:8]}")
    model = UNet(in_ch=3, base=state["config"]["base"], depth=state["config"]["depth"])
    model.load_state_dict(state["model"])
    model.to(dev)
    mean, std = state["mean"].to(dev), state["std"].to(dev)
    amp = bool(cfg["amp"]) and dev.type == "cuda"
    log = {r["epoch"]: r for r in map(json.loads,
                                      (run_dir / "epochs.jsonl").read_text().splitlines())}
    colour_iou = {s: log[cfg["selected_epoch"]]["val"][s]["iou"] for s in SUBSETS}

    res, figs = {}, {}
    for sub in SUBSETS:
        rows = d.index[("val", sub)]
        names = [d.keys[i].split("/")[1] for i in rows.tolist()]
        u8 = d.images[rows] if rgb_only else luminance(d.images[rows])
        x = (u8.float().div_(255.0) - mean) / std
        pred = predict(model, x, cfg["eval_batch_size"], amp)
        target = d.masks[rows].bool()
        c = tile_counts(pred, target)
        s = scores(*c.sum(0).tolist())
        n_pix = target.numel()
        tile_iou = np.where(c.sum(1) > 0, c[:, 0] / np.maximum(c.sum(1), 1), 1.0)
        order = sorted(range(len(names)), key=lambda i: (tile_iou[i], names[i]))
        m = order[(len(order) - 1) // 2]
        res[sub] = {"tiles": len(names), "iou": s["iou"], "recall": s["recall"],
                    "precision": s["precision"],
                    "predicted_fraction": int(pred.sum()) / n_pix,
                    "panel_share": int(target.sum()) / n_pix,
                    "colour_input_val_iou": colour_iou[sub],
                    "median_tile": {"tile": names[m], "iou": float(tile_iou[m])},
                    "per_tile": [{"tile": n, "iou": float(tile_iou[i]), "counts": c[i].tolist()}
                                 for i, n in enumerate(names)]}
        figs[sub] = (u8[m, 0].cpu().numpy(), target[m, 0].cpu().numpy(), pred[m, 0].cpu().numpy())

    print(f"PV01 val, {'plain RGB (--rgb sanity check)' if rgb_only else 'greyscale (BT.601)'}")
    for sub in SUBSETS:
        r = res[sub]
        print(f"  {sub:<13} {r['tiles']:>3} tiles  IoU {r['iou']:.4f} (colour input "
              f"{r['colour_input_val_iou']:.4f})  predicted {r['predicted_fraction']:.4f}  "
              f"panel {r['panel_share']:.4f}  recall {r['recall']:.4f}  "
              f"precision {r['precision']:.4f}  median tile {r['median_tile']['tile']} "
              f"({r['median_tile']['iou']:.4f})")

    if rgb_only:
        diff = {s: res[s]["iou"] - colour_iou[s] for s in SUBSETS}
        print("RGB IoU minus training log, epoch", cfg["selected_epoch"], diff)
        if any(v != 0.0 for v in diff.values()):
            raise SystemExit("the plain-RGB path does not reproduce the training log")
        print("reproduces exactly; nothing written")
        return 0

    rd = cfg["reading"]
    verdict = reading({s: res[s]["iou"] for s in SUBSETS},
                      rd["colour_explains_below"], rd["colour_does_not_explain_at_least"])
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = REPO / "runs" / f"seg_grey_{run}"
    dest.mkdir(parents=True)
    colours = {k: np.array(v, np.uint8) / 255 for k, v in eval_cfg["overlays"]["colours"].items()}
    labels = {"true_positive": "true positive", "false_positive": "false positive",
              "false_negative": "false negative"}
    for sub, (img, t, p) in figs.items():
        col = np.zeros((*t.shape, 3))
        col[p & t] = colours["true_positive"]
        col[p & ~t] = colours["false_positive"]
        col[~p & t] = colours["false_negative"]
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.8))
        for a, im, title in zip(ax, (img, t, col), ("greyscale input (BT.601)", "truth",
                                                    "prediction")):
            a.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0,
                     vmax=255 if im.dtype == np.uint8 else 1, interpolation="nearest")
            a.set_title(title, fontsize=10)
            a.set_axis_off()
        fig.legend(handles=[Patch(color=colours[k], label=v) for k, v in labels.items()],
                   loc="lower center", ncol=3, frameon=False, fontsize=9)
        mt = res[sub]["median_tile"]
        fig.suptitle(f"PV01 {sub} val, greyscale, median tile {mt['tile']}: IoU {mt['iou']:.3f}",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(dest / f"grey_{sub}_median.png", dpi=110)
        plt.close(fig)

    out = {"run": run, "git_sha": sha, "git_dirty": dirty, "config_path": CONFIG, "config": cfg,
           "weights_sha256": cfg["checkpoint_sha256"], "split_sha256": d.split_sha256, "amp": amp,
           "subsets": res, "reading": verdict}
    (dest / "grey.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8",
                                    newline="\n")
    print(f"reading: {verdict}")
    print(f"\nwrote {dest.relative_to(REPO).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
