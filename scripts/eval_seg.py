"""Score the PV01 U-Net, the colour rule and the null model per subset, with intervals.

    python scripts/eval_seg.py val      # any time; adds the epoch-selection check
    python scripts/eval_seg.py test     # once

Settings come from configs/seg_eval.yaml. On each split, per subset: IoU, Dice,
precision and recall of the U-Net (best.pt, logit > 0) and of the colour rule
from configs/seg_baseline.json, with percentile bootstrap intervals over tiles and
a paired interval of their difference (segmentation/evaluate.py); pixel accuracy
of all three models, the null all-background one included; the number of tiles
and of spatially separate groups behind each interval; every tile's U-Net and
colour-rule IoU.

**`test` reads the test split, and is meant to run once.** `guard()` refuses
unless the evaluation config and the configs it depends on are tracked and
committed and the whole tree is clean, and refuses if any `runs/seg_eval_test_*`
directory exists (a run that died after reading test counts as a pass; a second
one would not be the first). The checkpoint is not in git, so its SHA-256 must
equal the config's `checkpoint_sha256`. The test pass saves its predicted masks,
so the overlays and later questions need no second pass.

Writes runs/seg_eval_<split>_<utc>/eval.json and predictions.npz (gitignored).
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from solar_inspect.classification.train import git_state   # noqa: E402
from solar_inspect.detection.data import check_checkpoint   # noqa: E402

CONFIG = "configs/seg_eval.yaml"
GUARDED = (CONFIG, "configs/seg_baseline.json", "configs/seg_unet.yaml", "configs/d3_split.json")
N_PIX = 256 * 256


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def guard() -> dict[str, str]:
    """Refuse to read test unless every GUARDED file is committed, the tree is clean, and
    no test evaluation has started before. Returns {path: SHA of its last commit}."""
    done = sorted(REPO.glob("runs/seg_eval_test_*"))
    if done:
        raise SystemExit(f"a test evaluation already exists: {done[0].name}. "
                         "The test split is evaluated once.")
    if git("status", "--porcelain").stdout.strip():
        raise SystemExit("the working tree is not clean. Commit first: the committed "
                         "config is the one this evaluation is allowed to use.")
    shas = {}
    for f in GUARDED:
        if git("ls-files", "--error-unmatch", f).returncode != 0:
            raise SystemExit(f"{f} is not tracked. Commit it first.")
        sha = git("log", "-1", "--format=%H", "--", f).stdout.strip()
        if not sha:
            raise SystemExit(f"no commit touches {f}")
        print(f"guard: {f} committed in {sha[:7]}, tree clean")
        shas[f] = sha
    return shas


def groups(names: list[str], link: float) -> int:
    """Connected components of these tiles at name-coordinate distance <= link."""
    from scipy.sparse.csgraph import connected_components
    xy = np.array([[int(v) for v in n.split("_")[1:]] for n in names], float)
    adj = np.linalg.norm(xy[:, None] - xy[None], axis=-1) <= link
    return int(connected_components(adj, directed=False)[0])


def pick(names: list[str], iou: np.ndarray) -> dict[str, dict]:
    """The declared overlay tiles: lower median and lowest U-Net IoU, ties by name."""
    order = sorted(range(len(names)), key=lambda i: (iou[i], names[i]))
    return {k: {"tile": names[i], "unet_iou": float(iou[i])}
            for k, i in (("median", order[(len(order) - 1) // 2]), ("lowest", order[0]))}


def selection_check(run_dir: Path, cfg: dict, subsets) -> dict:
    rows = [json.loads(line) for line in (run_dir / "epochs.jsonl").read_text().splitlines()]
    by = {r["epoch"]: r for r in rows}
    (m0, m1), (r0, r1) = cfg["late_mean_epochs"], cfg["range_epochs"]
    out = {}
    for s in subsets:
        late = [by[e]["val"][s]["iou"] for e in range(m0, m1 + 1)]
        span = [by[e]["val"][s]["iou"] for e in range(r0, r1 + 1)]
        sel = by[cfg["epoch"]]["val"][s]["iou"]
        out[s] = {"selected": sel, f"mean_{m0}_{m1}": float(np.mean(late)),
                  "selected_minus_mean": sel - float(np.mean(late)),
                  f"min_{r0}_{r1}": min(span), f"max_{r0}_{r1}": max(span)}
    return out


def main(argv: list[str]) -> int:
    if argv not in (["val"], ["test"]):
        raise SystemExit(__doc__)
    split = argv[0]
    config_shas = guard() if split == "test" else None
    sha, dirty = git_state()
    cfg = yaml.safe_load((REPO / CONFIG).read_text(encoding="utf-8"))
    run_dir = REPO / cfg["run"]
    weights = run_dir / cfg["checkpoint"]
    weights_sha = check_checkpoint(weights, cfg["checkpoint_sha256"])
    print(f"guard: {weights.name} SHA-256 {weights_sha[:12]} = the config's checkpoint_sha256")
    if cfg["threshold"] != 0.5:
        raise SystemExit("predict() thresholds the logit at 0, which is sigmoid 0.5 only")

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = REPO / "runs" / f"seg_eval_{split}_{run}"
    dest.mkdir(parents=True)        # before test is read: a crash still leaves the marker

    import torch

    from solar_inspect.segmentation.data import SUBSETS, load_d3
    from solar_inspect.segmentation.evaluate import (
        colour_rule, paired_bootstrap, predict, tile_counts,
    )
    from solar_inspect.segmentation.unet import UNet

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    d = load_d3()
    state = torch.load(weights, map_location=d.images.device)
    if state["epoch"] != cfg["selected_epoch"] or state["split_sha256"] != d.split_sha256:
        raise SystemExit(f"checkpoint is epoch {state['epoch']}, split {state['split_sha256'][:8]}")
    if not torch.allclose(state["mean"].to(d.mean), d.mean) or \
            not torch.allclose(state["std"].to(d.std), d.std):
        raise SystemExit("normalisation differs from the checkpoint's")
    model = UNet(in_ch=3, base=state["config"]["base"], depth=state["config"]["depth"])
    model.load_state_dict(state["model"])
    model.to(d.images.device)
    amp = bool(cfg["amp"]) and d.images.device.type == "cuda"

    base = json.loads((REPO / cfg["baseline"]).read_text(encoding="utf-8"))
    if base["split_sha256"] != d.split_sha256:
        raise SystemExit("the colour rule was chosen on another split")
    rule = base["chosen"]
    pinned = json.loads((REPO / "configs/d3_split.json").read_text(encoding="utf-8"))
    link = pinned["rule"]["link_distance"]
    bs = cfg["bootstrap"]

    res, preds = {}, {}
    for sub in SUBSETS:
        rows = d.index[(split, sub)]
        names = [d.keys[i].split("/")[1] for i in rows.tolist()]
        target = d.masks[rows].bool()
        unet = predict(model, d.batch(rows), cfg["eval_batch_size"], amp)
        colour = colour_rule(d.images[rows], rule["feature"], rule["direction"], rule["threshold"])
        cu, cc = tile_counts(unet, target), tile_counts(colour, target)
        fg = float(target.float().mean())
        boot = paired_bootstrap(cu, cc, draws=bs["draws"], seed=bs["seed"])
        tile_iou = {k: np.where(c.sum(1) > 0, c[:, 0] / np.maximum(c.sum(1), 1), 1.0)
                    for k, c in (("unet", cu), ("colour", cc))}
        res[sub] = {
            "tiles": len(names), "groups": groups(names, link), "foreground": fg,
            "unet": boot["a"], "colour_rule": boot["b"], "unet_minus_colour_rule": boot["a_minus_b"],
            "pixel_accuracy": {"unet": 1 - (cu[:, 1] + cu[:, 2]).sum() / (len(names) * N_PIX),
                               "colour_rule": 1 - (cc[:, 1] + cc[:, 2]).sum() / (len(names) * N_PIX),
                               "null_all_background": 1 - fg},
            "null_all_background": {"iou": 0.0, "dice": 0.0},
            "overlay_tiles": pick(names, tile_iou["unet"]),
            "per_tile": [{"tile": n, "foreground": float(target[i].float().mean()),
                          "unet_iou": float(tile_iou["unet"][i]),
                          "colour_rule_iou": float(tile_iou["colour"][i]),
                          "unet_counts": cu[i].tolist()} for i, n in enumerate(names)],
        }
        preds[sub] = (names, unet[:, 0].cpu().numpy())

    out = {"run": run, "split": split, "git_sha": sha, "git_dirty": dirty,
           "config_path": CONFIG, "config": cfg, "config_commit_shas": config_shas,
           "weights": weights.relative_to(REPO).as_posix(), "weights_sha256": weights_sha,
           "split_sha256": d.split_sha256, "colour_rule": base["rule"], "amp": amp,
           "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "subsets": res}

    if split == "val":
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        out["selection_check"] = selection_check(
            run_dir, {**cfg["selection_check"], "epoch": cfg["selected_epoch"]}, SUBSETS)
        out["reproduces"] = {s: {"unet_iou_minus_training_log": res[s]["unet"]["iou"]["point"]
                                 - manifest["best"]["val"][s]["iou"],
                                 "colour_iou_minus_baseline_json": res[s]["colour_rule"]["iou"]["point"]
                                 - base["val"][s]["iou"]} for s in SUBSETS}

    (dest / "eval.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    np.savez_compressed(dest / "predictions.npz",
                        **{f"{s}_names": np.array(preds[s][0]) for s in SUBSETS},
                        **{f"{s}_unet": np.packbits(preds[s][1], axis=-1) for s in SUBSETS})

    print(f"\n=== {split.upper()}{', once' if split == 'test' else ''}: U-Net vs colour rule "
          f"({base['rule']}) vs null, per subset; {bs['draws']} paired tile resamples, seed "
          f"{bs['seed']} ===")
    for s in SUBSETS:
        r = res[s]
        u, c, dlt = r["unet"], r["colour_rule"], r["unet_minus_colour_rule"]
        f = lambda m: f"{m['point']:.4f} [{m['ci95'][0]:.4f}, {m['ci95'][1]:.4f}]"  # noqa: E731
        print(f"{s} ({r['tiles']} tiles, {r['groups']} groups, fg {r['foreground']:.4f})")
        for m in ("iou", "dice", "precision", "recall"):
            print(f"  {m:<10} U-Net {f(u[m])}   colour {f(c[m])}   diff {f(dlt[m])}")
        pa = r["pixel_accuracy"]
        print(f"  pixel acc  U-Net {pa['unet']:.4f}   colour {pa['colour_rule']:.4f}   "
              f"null {pa['null_all_background']:.4f}")
        print("  overlay tiles: " + "  ".join(f"{k} {v['tile']} ({v['unet_iou']:.4f})"
                                             for k, v in r["overlay_tiles"].items()))
    if split == "val":
        print("selection check:", json.dumps(out["selection_check"], indent=1))
        print("reproduces:", json.dumps(out["reproduces"]))
    print(f"\nwrote {dest.relative_to(REPO).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
