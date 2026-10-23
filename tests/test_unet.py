"""The U-Net, tested before it exists.

Parameter count of the smallest net, worked by hand (in 3, base 4, depth 1).
Convs in double-conv blocks have no bias (batch norm follows); batch norm has a
weight and a bias per channel (running statistics are buffers, not parameters):

    encoder  double conv 3 -> 4      9*3*4 + 2*4 + 9*4*4 + 2*4   =  268
    bottom   double conv 4 -> 8      9*4*8 + 2*8 + 9*8*8 + 2*8   =  896
    up       3x3 conv 8 -> 4, bias   9*8*4 + 4                   =  292
    decoder  double conv 8 -> 4      9*8*4 + 2*4 + 9*4*4 + 2*4   =  448
    head     1x1 conv 4 -> 1, bias   4 + 1                       =    5
                                                                   -----
                                                                    1909
"""
from __future__ import annotations

import pytest
import torch

from solar_inspect.segmentation.unet import UNet


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def test_smallest_net_has_the_hand_counted_parameters():
    assert n_params(UNet(in_ch=3, base=4, depth=1)) == 1909


def test_default_net_parameter_count_is_printed():
    n = n_params(UNet())
    print(f"UNet() parameters: {n:,}")
    assert n > 0


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_256_input_gives_one_logit_per_pixel(depth):
    m = UNet(base=8, depth=depth).eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 256, 256))
    assert y.shape == (2, 1, 256, 256)


def test_input_not_divisible_by_the_pooling_is_refused():
    with pytest.raises(ValueError):
        UNet(base=4, depth=4)(torch.randn(1, 3, 100, 100))


@pytest.mark.parametrize("level", [0, 1, 2])
def test_each_skip_connection_is_used(level):
    """Zero one skip, leave the downward path alone: the output must change."""
    torch.manual_seed(0)
    m = UNet(base=4, depth=3).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = m(x)
        h = m.ups[level].register_forward_pre_hook(
            lambda mod, args: (args[0], torch.zeros_like(args[1])))
        y_cut = m(x)
        h.remove()
        y_again = m(x)
    assert torch.equal(y, y_again)
    assert not torch.allclose(y, y_cut)


def test_receptive_field_of_depth_4_is_252_px():
    """Measured, then pinned: the input window one output pixel depends on.

    Back-propagate from the centre output pixel and take the extent of the
    non-zero input gradient. A linearised copy (ReLU removed, max pooling made
    average pooling) keeps every path alive, so no gradient is zero by accident.
    The window depends on depth and kernel sizes, not on width, so base 4 stands
    for the trained base 32. The full-width trained architecture gives the same.
    """
    torch.manual_seed(0)
    m = UNet(base=4, depth=4).double().eval()
    for mod in list(m.modules()):
        for name, child in mod.named_children():
            if isinstance(child, torch.nn.ReLU):
                setattr(mod, name, torch.nn.Identity())
            elif isinstance(child, torch.nn.MaxPool2d):
                setattr(mod, name, torch.nn.AvgPool2d(2))
    x = torch.randn(1, 3, 320, 320, dtype=torch.double, requires_grad=True)
    m(x)[0, 0, 160, 160].backward()
    rows = torch.nonzero(x.grad.abs().sum((0, 1, 3)))[:, 0]
    cols = torch.nonzero(x.grad.abs().sum((0, 1, 2)))[:, 0]
    assert int(rows.max() - rows.min() + 1) == 252
    assert int(cols.max() - cols.min() + 1) == 252
