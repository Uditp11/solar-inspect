"""Train the U-Net on PV01, select the epoch on val, log every epoch.

Driven by scripts/train_seg.py, which owns the config, the run directory and the
manifest. This module owns the loop.

Evaluation is on **val**, hardcoded, per subset. The test split is loaded with
everything else (one tensor) and never indexed here; it is scored once, after
the final config is committed. Guard tiles are never indexed here either.

**Best epoch = highest mean of the three per-subset val IoUs.** Not pooled IoU:
FlatConcrete holds 60 of the 95 val tiles and pooled IoU would mostly be its
number. The mean gives Brick, at 5% panel pixels, the same say as the others.

Determinism: seeded, cudnn deterministic and benchmark off, as in module 4. The
backward pass of bilinear upsampling on CUDA accumulates with atomics, so two runs
of one config agree closely but not bit for bit.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from solar_inspect.segmentation.data import SUBSETS, D3Data
from solar_inspect.segmentation.metrics import counts_torch, scores, soft_dice_loss
from solar_inspect.segmentation.unet import UNet

EVAL_SPLIT = "val"          # never "test" -- see the module docstring


def augment(x: torch.Tensor, y: torch.Tensor, g: torch.Generator
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per tile: a random 90-degree rotation, then a horizontal flip half the time.

    Those two generate all eight flips and rotations of a square (a vertical flip
    is a horizontal flip plus a half turn), each with probability 1/8. The same
    transform is applied to the tile and its mask.
    """
    k = torch.randint(0, 4, (len(x),), generator=g, device=g.device).tolist()
    flip = torch.rand(len(x), generator=g, device=g.device).lt(0.5).tolist()
    xs, ys = [], []
    for xi, yi, ki, fi in zip(x, y, k, flip):
        xi, yi = torch.rot90(xi, ki, (1, 2)), torch.rot90(yi, ki, (1, 2))
        if fi:
            xi, yi = xi.flip(2), yi.flip(2)
        xs.append(xi)
        ys.append(yi)
    return torch.stack(xs), torch.stack(ys)


@torch.no_grad()
def evaluate(model: nn.Module, d: D3Data, split: str, batch: int, amp: bool
             ) -> dict[str, dict[str, float]]:
    """Per-subset IoU, Dice, precision, recall over all pixels of the subset's tiles."""
    model.eval()
    out = {}
    for sub in SUBSETS:
        rows = d.index[(split, sub)]
        c = torch.zeros(3, dtype=torch.int64, device=d.images.device)
        for i in range(0, len(rows), batch):
            r = rows[i:i + batch]
            with torch.autocast(d.images.device.type, dtype=torch.float16, enabled=amp):
                logits = model(d.batch(r))
            c += counts_torch(logits > 0, d.masks[r].bool())
        out[sub] = scores(*c.tolist())
    return out


def fit(cfg: dict, d: D3Data, run_dir: Path) -> dict:
    """Train for cfg["epochs"], writing epochs.jsonl, last.pt and best.pt to run_dir."""
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = d.images.device
    amp = bool(cfg["amp"]) and device.type == "cuda"
    model = UNet(in_ch=3, base=cfg["base"], depth=cfg["depth"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    g = torch.Generator(device=device).manual_seed(cfg["seed"])

    train_rows = d.rows("train")
    bs = cfg["batch_size"]
    w_bce, w_dice = cfg["loss"]["bce"], cfg["loss"]["dice"]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params  train={len(train_rows)} "
          f"{EVAL_SPLIT}={len(d.rows(EVAL_SPLIT))}  {math.ceil(len(train_rows) / bs)} steps/epoch  "
          f"amp={amp}", flush=True)

    log = (run_dir / "epochs.jsonl").open("a", encoding="utf-8", newline="\n")
    best = {"epoch": 0, "mean_val_iou": -1.0}
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        te = time.perf_counter()
        model.train()
        perm = train_rows[torch.randperm(len(train_rows), generator=g, device=device)]
        total = 0.0
        for i in range(0, len(perm), bs):
            r = perm[i:i + bs]
            x, y = d.batch(r), d.masks[r].float()
            if cfg["augment"]:
                x, y = augment(x, y, g)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = (w_bce * F.binary_cross_entropy_with_logits(logits.float(), y)
                    + w_dice * soft_dice_loss(logits, y))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss is {loss.item()} at epoch {epoch}, step {i // bs}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(r)
        lr = opt.param_groups[0]["lr"]
        sched.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_s = time.perf_counter() - te

        val = evaluate(model, d, EVAL_SPLIT, cfg["eval_batch_size"], amp)
        mean_iou = sum(val[s]["iou"] for s in SUBSETS) / len(SUBSETS)
        row = {"epoch": epoch, "train_loss": total / len(perm), "lr": lr,
               "val": val, "mean_val_iou": mean_iou,
               "train_s": train_s, "epoch_s": time.perf_counter() - te}
        history.append(row)
        log.write(json.dumps(row) + "\n")
        log.flush()

        state = {"epoch": epoch, "model": model.state_dict(), "config": cfg,
                 "mean": d.mean.cpu(), "std": d.std.cpu(), "split_sha256": d.split_sha256}
        torch.save({**state, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict()}, run_dir / "last.pt")
        if mean_iou > best["mean_val_iou"]:
            best = {"epoch": epoch, "mean_val_iou": mean_iou, "val": val}
            torch.save(state, run_dir / "best.pt")

        print(f"epoch {epoch:>3}  loss {row['train_loss']:.4f}  {EVAL_SPLIT} IoU "
              + "  ".join(f"{s} {val[s]['iou']:.4f}" for s in SUBSETS)
              + f"  mean {mean_iou:.4f}  ({row['epoch_s']:.1f} s)", flush=True)
    log.close()
    return {"params": n_params, "wall_s": time.perf_counter() - t0, "best": best,
            "history": history}
