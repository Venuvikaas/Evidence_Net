"""Candidate Base Reconstruction.

Per the product definition (``docs/product-definition.md`` section 10.2) the
Base Reconstruction is

    b = U(y) + h_b(f(y))

where ``U`` is the deterministic bilinear 2x up-sampling and ``h_b`` is a
learned refinement computed from multi-scale features of the input. The
deterministic anchor keeps the model fidelity-oriented and lower-intervention
by construction; its learned component only corrects the anchor rather than
generating an unrestricted image.
"""

from __future__ import annotations

import torch
from torch import nn


class _FeatureBlock(nn.Module):
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


class BaseReconstruction(nn.Module):
    """b = U(y) + h_b(f(y)); 128x128 input -> 256x256 output in [0, 1]."""

    def __init__(self, hidden_channels: int = 32, depth: int = 3) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.features = nn.ModuleList()
        channels = hidden_channels
        self.features.append(_FeatureBlock(1, channels))
        for _ in range(depth - 1):
            self.features.append(_FeatureBlock(channels, channels * 2))
            channels *= 2
        # h_b: upsample features to the 256x256 grid, then predict the
        # refinement added on top of the deterministic anchor U(y).
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.refinement = nn.Sequential(
            _FeatureBlock(channels, hidden_channels),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    @staticmethod
    def anchor(inputs: torch.Tensor) -> torch.Tensor:
        """Deterministic anchor U(y): bilinear 2x up-sampling of the input."""
        return nn.functional.interpolate(
            inputs, scale_factor=2, mode="bilinear", align_corners=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x
        for block in self.features:
            features = block(features)
        refined = self.refinement(self.up(features))
        return torch.clamp(self.anchor(x) + refined, 0.0, 1.0)
