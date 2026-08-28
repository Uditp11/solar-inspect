"""ImageNet-pretrained ResNet-18, adapted to 1-channel 40x24 crops.

This is the transfer-learning baseline and the distillation teacher. Two
adaptations are needed to point a 224x224 RGB backbone at a 40x24 greyscale
crop, and both are choices rather than defaults. They are made here, in the
code, with the alternative that was not taken written down beside them.


DECISION 1 -- ASPECT RATIO. D1 crops are 40x24, which is 5:3. A square resize to
96x96 stretches the width by 1.67x and every defect signature with it.

**Taken: the distorting resize.** The distortion is a fixed transform applied
identically to every image in train, val and test, so it introduces no
train/test mismatch -- it is a change of basis, not a corruption. And ResNet-18's
ImageNet features were themselves trained under RandomResizedCrop, which jitters
aspect ratio over 3/4 to 4/3 by design, so the backbone has never assumed a
faithful aspect ratio in the first place.

**Not taken: pad to 5:3 inside the square, then resize.** It preserves the
geometry, at two costs. 40% of the 96x96 canvas becomes constant fill, so 40% of
the compute and 40% of the receptive field of every early filter is spent on a
region with no information in it. Worse, the fill introduces a hard synthetic
edge at a fixed position in every image -- a straight, maximal-contrast border
with no counterpart anywhere in ImageNet, sitting exactly where conv1's 7x7
filters will respond hardest.

Not ablated. One of these is picked and stated; measuring both would be a fifth
arm the budget does not have, and the difference would be inside the noise floor.


DECISION 2 -- ONE CHANNEL INTO A THREE-CHANNEL conv1. The stem is
Conv2d(3, 64, 7). The two standard fixes are to replicate the grey channel three
times, or to sum conv1's weights across the input channel.

**They are the same function at initialisation -- but only under a single shared
normalisation** -- and it is worth being precise about that rather than treating
it as a real fork. Feeding x replicated three times gives sum_c W[:, c] * x, which
is exactly the 1-channel convolution whose weight is sum_c W[:, c]. Identical
output, to floating point.

The condition is doing real work and is not a technicality. The step above needs
all three input channels to carry the same number. Normalise per-channel with
ImageNet's (0.485/0.456/0.406, 0.229/0.224/0.225) and they do not: the channels
become x_1, x_2, x_3, sum_c W[:, c] * x_c is not sum_c W[:, c] * x, and the
equivalence is false. This model normalises with D1's own train-split mean and std
-- one constant on the one real channel, the same statistics data.py computes for
every other model in this project -- so the condition holds here. The self-check
below feeds the reference stem `up.repeat(1, 3, 1, 1)`, i.e. identical channels,
so it tests exactly the case the claim is made about.

They differ in three ways that only show up later:

  - **Parameters.** Replication keeps 3 x 64 x 7 x 7 = 9,408 weights in conv1
    where 3,136 suffice. The extra 6,272 are three copies of one filter, free to
    drift apart while fine-tuning on 13,965 images.
  - **Normalisation.** Replication invites ImageNet's per-channel mean/std
    (0.485/0.456/0.406), three different constants for three copies of the same
    number. On an 8-bit thermal render those constants describe nothing.
  - **Gradients.** Summed, the three ImageNet filters are tied for the rest of
    training. Replicated, they are not.

**Taken: sum the weights.** D1's own train-split mean and std are used, the same
ones `data.py` computes for every other model in this project, and conv1 becomes
Conv2d(1, 64, 7) carrying the summed ImageNet filters.


The resize lives inside `forward`, not in the data pipeline, so that every caller
-- training, evaluation, the leakage subgroup breakdown, the latency benchmark --
sees one model that takes (N, 1, 40, 24) exactly like TinyCNN does. It also means
the interpolation is inside the number the latency table reports, which is where
it belongs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

HW = (40, 24)


class ResNet18Gray(nn.Module):
    def __init__(self, n_classes: int = 12, size: int = 96,
                 pretrained: bool = True) -> None:
        super().__init__()
        self.size = size
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        stem = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Decision 2: sum, not replicate. Identical forward to replication at
            # init; one third of the parameters and one normalisation constant.
            stem.weight.copy_(net.conv1.weight.sum(dim=1, keepdim=True))
        net.conv1 = stem
        net.fc = nn.Linear(net.fc.in_features, n_classes)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Decision 1: distorting resize to a square. bilinear, antialias off --
        # this is an upsample, so there is nothing to alias.
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear",
                          align_corners=False)
        return self.net(x)


if __name__ == "__main__":
    m = ResNet18Gray()
    x = torch.zeros(4, 1, *HW)
    print("params:", f"{sum(p.numel() for p in m.parameters()):,}")
    print("logits:", tuple(m(x).shape))

    # The claim in decision 2, checked rather than asserted: summing conv1's
    # weights and replicating the channel are the same function at init.
    ref = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    with torch.no_grad():
        up = F.interpolate(x + 0.3, size=(96, 96), mode="bilinear", align_corners=False)
        a, b = m.net.conv1(up), ref.conv1(up.repeat(1, 3, 1, 1))
    print("summed vs replicated conv1, max |diff|:", float((a - b).abs().max()))
    assert torch.allclose(a, b, atol=1e-5), "they are supposed to be the same function"
