"""U-Net, from torch.nn layers only: no segmentation library, no pretrained weights.

    encoder   double conv, then 2x2 max pool, `depth` times; channels base * 2^i
    bottom    double conv at base * 2^depth
    decoder   upsample, concatenate the encoder's map at that scale (the skip),
              double conv, `depth` times
    head      1x1 conv to one logit per pixel

A double conv is (3x3 conv, batch norm, ReLU) twice. The convs before batch norm
carry no bias: batch norm's own shift makes one redundant.

**Skips.** Pooling throws away where an edge was to within a few pixels; the
bottom of the net knows *that* there is an array, not *where its border is*. The
skip hands the decoder the encoder's full-resolution map at each scale, so the
mask boundary is drawn from features that still have it.

**Upsampling is bilinear, then a 3x3 conv**, not a transposed conv. A stride-2
transposed conv with a 3x3 kernel covers output pixels unevenly (some get two
kernel taps, some one), which prints a checkerboard into the output unless the
weights learn to cancel it. Bilinear interpolation spreads every input pixel
evenly, and the conv after it does the learning.
"""
from __future__ import annotations

import torch
from torch import nn


def double_conv(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class Up(nn.Module):
    """x2 bilinear upsample, 3x3 conv halving the channels, concat the skip, double conv."""

    def __init__(self, c_in: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c_in, c_in // 2, 3, padding=1),
        )
        self.conv = double_conv(c_in, c_in // 2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([skip, self.up(x)], dim=1))


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32, depth: int = 4) -> None:
        super().__init__()
        widths = [base * 2**i for i in range(depth + 1)]
        self.depth = depth
        self.downs = nn.ModuleList(
            double_conv(c_in, c_out) for c_in, c_out in zip([in_ch, *widths[:-2]], widths[:-1]))
        self.pool = nn.MaxPool2d(2)
        self.bottom = double_conv(widths[-2], widths[-1])
        # ups[0] is the shallowest (full resolution), matching downs[0]
        self.ups = nn.ModuleList(Up(c) for c in widths[1:])
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if h % 2**self.depth or w % 2**self.depth:
            raise ValueError(f"input {h}x{w} is not divisible by 2^{self.depth}")
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottom(x)
        for up, skip in zip(reversed(self.ups), reversed(skips)):
            x = up(x, skip)
        return self.head(x)


if __name__ == "__main__":
    m = UNet()
    print("params:", sum(p.numel() for p in m.parameters()))
    print("out:", tuple(m(torch.zeros(1, 3, 256, 256)).shape))
