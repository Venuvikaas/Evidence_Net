"""Strong direct-restoration baseline.

A compact encoder-decoder CNN that maps the 128x128 input directly to a
256x256 restoration, with no deterministic anchor. It is the "equal-capacity
direct restoration" comparison for the Base Reconstruction: same interface,
same training procedure, but the output is entirely learned.
"""

from __future__ import annotations

import torch
from torch import nn


class _ConvBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DirectRestoration(nn.Module):
    """Encoder-decoder direct restoration: 128x128 -> 256x256 in [0, 1]."""

    def __init__(self, hidden_channels: int = 32, depth: int = 3) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.encoder = nn.ModuleList()
        channels = hidden_channels
        self.encoder.append(_ConvBlock(1, channels))
        for _ in range(depth - 1):
            self.encoder.append(_ConvBlock(channels, channels * 2))
            channels *= 2
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.decoder = nn.Sequential(
            _ConvBlock(channels, hidden_channels),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x
        for block in self.encoder:
            features = block(features)
        features = self.up(features)
        output = self.decoder(features)
        # Sigmoid bounds to (0, 1) with gradients everywhere; a hard clamp
        # saturates at zero when a randomly initialized net outputs negative
        # values, freezing training.
        return torch.sigmoid(output)
