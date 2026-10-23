"""The PV01 U-Net on D2's greyscale thermal val frames: does it transfer?

    python scripts/seg_thermal_negative.py

Rules in configs/seg_thermal_negative.yaml, committed before the first run. The
model is the one configs/seg_eval.yaml pins (same SHA-256), unchanged and not
fine-tuned, at threshold 0.5. D2 **val** frames only; the test tree is never
listed (`splits()` refuses a val path inside it).

Each frame is read as greyscale, replicated to three channels, scaled to [0, 1]
and standardised with the PV01 train mean and std stored in the checkpoint, then
padded with zeros (bottom and right) to a multiple of 16 if needed and cropped
back after the forward pass.

D2 has boxes, not masks. The reference is the filled-box mask: a pixel is
positive when its centre lies inside any ground-truth box. Per frame and pooled
(pixel counts summed over frames): predicted positive fraction, filled-box
coverage, IoU with the filled-box mask, and the share of predicted positives
outside every box.

Writes runs/seg_thermal_<utc>/ (gitignored): thermal.json and three overlays.
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

from solar_inspect.classification.train import git_state            # noqa: E402
from solar_inspect.detection.data import check_checkpoint, ground_truth, splits  # noqa: E402

CONFIG = "configs/seg_thermal_negative.yaml"


def filled_boxes(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
    """bool (h, w): pixel centres inside any [x1, y1, x2, y2] box."""
    ys, xs = np.arange(h) + 0.5, np.arange(w) + 0.5
    m = np.zeros((h, w), bool)
    for x1, y1, x2, y2 in boxes:
        m |= ((ys >= y1) & (ys <= y2))[:, None] & ((xs >= x1) & (xs <= x2))[None, :]
    return m


def stats(pred: np.ndarray, box: np.ndarray) -> dict[str, float]:
    p, b = int(pred.sum()), int(box.sum())
    inter, n = int((pred & box).sum()), pred.size
    return {"predicted_fraction": p / n, "box_coverage": b / n,
            "box_mask_iou": inter / (p + b - inter) if p + b - inter else 1.0,
            "predicted_outside_boxes": (p - inter) / p if p else float("nan"),
            "pixels": n, "predicted": p, "box": b, "intersection": inter}


def main() -> int:
    sha, dirty = git_state()
    cfg = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
    if cfg["split"] != "val" or cfg["threshold"] != 0.5:
        raise SystemExit("this check runs on D2 val at threshold 0.5 only")
    eval_cfg = yaml.safe_load((REPO / "configs/seg_eval.yaml").read_text(encoding="utf-8"))
    if cfg["checkpoint_sha256"] != eval_cfg["checkpoint_sha256"]:
        raise SystemExit("not the model the segmentation evaluation pins")
    weights = REPO / cfg["run"] / cfg["checkpoint"]
    check_checkpoint(weights, cfg["checkpoint_sha256"])
    val_dir = splits()["val"]
    names, boxes = ground_truth(val_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.patches import Patch
    from PIL import Image

    from solar_inspect.segmentation.evaluate import predict
    from solar_inspect.segmentation.unet import UNet

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    state = torch.load(weights, map_location=dev)
    model = UNet(in_ch=3, base=state["config"]["base"], depth=state["config"]["depth"])
    model.load_state_dict(state["model"])
    model.to(dev)
    mean, std = state["mean"].to(dev), state["std"].to(dev)
    amp = bool(cfg["amp"]) and dev.type == "cuda"
    mult = 2 ** state["config"]["depth"]

    grey = np.stack([np.asarray(Image.open(val_dir / n).convert("L")) for n in names])
    h, w = grey.shape[1:]
    ph, pw = -h % mult, -w % mult
    preds = []
    for i in range(0, len(grey), cfg["batch_size"]):
        x = torch.from_numpy(grey[i:i + cfg["batch_size"]]).to(dev).float().div_(255.0)
        x = (x[:, None].expand(-1, 3, -1, -1) - mean) / std
        x = torch.nn.functional.pad(x, (0, pw, 0, ph))
        preds.append(predict(model, x, cfg["batch_size"], amp)[:, 0, :h, :w].cpu().numpy())
    pred = np.concatenate(preds)
    box = np.stack([filled_boxes(b, h, w) for b in boxes])

    per_frame = [{"frame": n, "boxes": len(b), **stats(pred[i], box[i])}
                 for i, (n, b) in enumerate(zip(names, boxes))]
    pooled = stats(pred, box)

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = REPO / "runs" / f"seg_thermal_{run}"
    dest.mkdir(parents=True)
    colours = {k: np.array(v, np.uint8) / 255 for k, v in eval_cfg["overlays"]["colours"].items()}
    figures = []
    for which, i in (("first", 0), ("middle", (len(names) - 1) // 2), ("last", len(names) - 1)):
        col = np.zeros((h, w, 3))
        col[pred[i] & box[i]] = colours["true_positive"]
        col[pred[i] & ~box[i]] = colours["false_positive"]
        col[~pred[i] & box[i]] = colours["false_negative"]
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
        for a, im, t in zip(ax, (grey[i], box[i], col),
                            ("D2 thermal frame (greyscale)", "filled ground-truth boxes",
                             "PV01 U-Net prediction")):
            a.imshow(im, cmap="gray" if im.ndim == 2 else None, interpolation="nearest")
            a.set_title(t, fontsize=10)
            a.set_axis_off()
        labels = {"true_positive": "predicted, inside a box", "false_positive":
                  "predicted, outside every box", "false_negative": "box, not predicted"}
        fig.legend(handles=[Patch(color=colours[k], label=v) for k, v in labels.items()],
                   loc="lower center", ncol=3, frameon=False, fontsize=9)
        f = per_frame[i]
        fig.suptitle(f"D2 val, {which} frame: {names[i].split('.rf.')[0]}; predicted "
                     f"{f['predicted_fraction']:.3f} of pixels, boxes cover "
                     f"{f['box_coverage']:.3f}, IoU {f['box_mask_iou']:.3f}", fontsize=10)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        path = dest / f"d2_val_{which}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        figures.append({"which": which, "frame": names[i], "png": path.relative_to(REPO).as_posix()})

    out = {"run": run, "git_sha": sha, "git_dirty": dirty, "config_path": CONFIG, "config": cfg,
           "frames": len(names), "boxes": int(sum(len(b) for b in boxes)), "frame_hw": [h, w],
           "padded": [ph, pw], "pooled": pooled, "per_frame": per_frame, "figures": figures}
    (dest / "thermal.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8",
                                       newline="\n")

    def q(k: str) -> str:
        v = np.array([f[k] for f in per_frame], float)
        v = v[~np.isnan(v)]
        return f"median {np.median(v):.4f} [min {v.min():.4f}, max {v.max():.4f}] over {len(v)}"

    print(f"D2 val, {len(names)} frames {w} x {h}, {out['boxes']} boxes, padding {ph} x {pw} px")
    for k in ("predicted_fraction", "box_coverage", "box_mask_iou", "predicted_outside_boxes"):
        print(f"  {k:<24} pooled {pooled[k]:.4f}   per frame {q(k)}")
    for f in figures:
        print(f"  {f['png']}")
    print(f"\nwrote {dest.relative_to(REPO).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
