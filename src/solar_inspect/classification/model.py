"""A CNN sized for 40x24 crops, which is not the same problem as 224x224.

**You get at most three downsamples here.** 40x24 -> 20x12 -> 10x6 -> 5x3. A
fourth pool leaves 2x1 and a fifth leaves nothing at all. ImageNet backbones
assume 224 square and downsample five times, so dropping a ResNet in unchanged
either destroys the spatial extent or forces an upsample to a resolution the data
never had. That constraint is the whole architectural story of this module and it
belongs in the code, not only in the explainer.

This is the baseline: one 3x3 conv per stage, BN, ReLU, 2x2 max pool, then a
flatten and a linear layer. ~116k parameters. It exists to be a floor, not to be
good.
"""
from __future__ import annotations

import torch
from torch import nn

HW = (40, 24)


class TinyCNN(nn.Module):
    def __init__(self, n_classes: int = 12, widths: tuple[int, ...] = (32, 64, 128)) -> None:
        super().__init__()
        if len(widths) > 3:
            raise ValueError(
                f"{len(widths)} stages on a {HW[0]}x{HW[1]} input leaves nothing to pool: "
                "three is the ceiling (40x24 -> 20x12 -> 10x6 -> 5x3)."
            )
        layers: list[nn.Module] = []
        c_in = 1
        for c_out in widths:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        h, w = HW[0] >> len(widths), HW[1] >> len(widths)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c_in * h * w, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


if __name__ == "__main__":
    m = TinyCNN()
    x = torch.zeros(4, 1, *HW)
    print(m)
    print("params:", sum(p.numel() for p in m.parameters()))
    print("features:", tuple(m.features(x).shape), "-> logits:", tuple(m(x).shape))
