"""FIDNet variant with multiscale dilated convolutions on the amplitude only."""

import torch
from torch import nn

from .FIDNet import FIDNet


class MultiScaleAmplitudeConv(nn.Module):
    """Predict an amplitude residual without an internal residual connection.

    Rates are the actual Conv2d dilation values, with no extra multiplier.
    Normalization, activations, dropout and global pooling are deliberately
    omitted so this experiment focuses on the multiscale convolution structure.
    """

    def __init__(self, channels, dilation_rates=(1, 2, 3)):
        super().__init__()
        dilation_rates = tuple(dilation_rates)
        if not dilation_rates or any(
            not isinstance(rate, int) or rate < 1 for rate in dilation_rates
        ):
            raise ValueError("dilation_rates must contain positive integers")

        self.pre_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.branches = nn.ModuleList(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=rate,
                dilation=rate,
                groups=channels,
                bias=False,
            )
            for rate in dilation_rates
        )
        self.fusion = nn.Conv2d(
            channels * len(dilation_rates), channels, kernel_size=1
        )

    def forward(self, x):
        x = self.pre_conv(x)
        return self.fusion(torch.cat([branch(x) for branch in self.branches], dim=1))


class FIDNetMS(FIDNet):
    """Keep FIDNet unchanged except for every APB's amplitude convolution.

    Input/output: [batch, spectral_bands, height, width]. The original network
    requires height and width divisible by four for its 3D U-Net skip additions.
    hid_S must be divisible by four for the original cross-attention heads.
    """

    def __init__(self, hid_S, hid_T, opt=None, dilation_rates=(1, 2, 3)):
        super().__init__(hid_S=hid_S, hid_T=hid_T, opt=opt)
        dilation_rates = tuple(dilation_rates)
        for block in self.fourierNet.enc_fre.enc:
            block.amp_fuse_1 = MultiScaleAmplitudeConv(hid_S, dilation_rates)
