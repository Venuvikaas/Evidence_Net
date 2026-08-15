"""Bounded Detail Proposal.

Per the product definition (``docs/product-definition.md`` section 10.3) and
the Phase 4 contract (``docs/proposal-contract.md``), the detail proposal is
an amplitude-bounded residual computed from the degraded input and the frozen
Base output:

    d = alpha * tanh(h_d(f(y), b))

with ``|d| <= alpha`` elementwise. The ungated candidate is ``c = b + d`` and
the gated reconstruction is ``x_hat = b + g * d`` with gate ``g in [0, 1]``.
The Base parameters are frozen (stop-gradient) during proposal training.
"""

from __future__ import annotations

import torch
from torch import nn


class _Block(nn.Module):
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


class DetailProposer(nn.Module):
    """Bounded proposal head: ``(y, b) -> d`` with ``|d| <= amplitude``.

    The degraded input is up-sampled to the output grid and concatenated with
    the Base output; the head predicts a bounded residual on that grid.
    """

    def __init__(self, hidden_channels: int = 32, depth: int = 2, amplitude: float = 0.1) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if amplitude <= 0.0:
            raise ValueError("amplitude must be > 0")
        self.amplitude = amplitude
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        blocks: list[nn.Module] = []
        channels = hidden_channels
        # Input to the blocks is [up(y), b] -> 2 channels on the output grid.
        blocks.append(_Block(2, channels))
        for _ in range(depth - 1):
            blocks.append(_Block(channels, channels))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

    def forward(self, y: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        features = torch.cat([self.up(y), b], dim=1)
        return self.amplitude * torch.tanh(self.head(self.blocks(features)))


class BoundedDetailProposal(nn.Module):
    """Frozen Base plus bounded proposal, exposed through one interface.

    ``forward`` returns the ungated candidate ``c = clamp(b + d, 0, 1)`` so
    the model satisfies the training harness interface. ``propose`` returns
    the (base, proposal, candidate) triple used by the oracle study.
    """

    def __init__(self, base: nn.Module, proposer: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.proposer = proposer
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def base_output(self, y: torch.Tensor) -> torch.Tensor:
        """Frozen Base reconstruction on the output grid."""
        return self.base(y)

    def propose(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(b, d, c)``: base, bounded proposal, ungated candidate."""
        b = self.base(y)
        d = self.proposer(y, b)
        c = torch.clamp(b + d, 0.0, 1.0)
        self.last_base = b.detach()
        self.last_proposal = d
        return b, d, c

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.propose(y)[2]


def fuse(
    base_output: torch.Tensor, proposal: torch.Tensor, gate: float | torch.Tensor
) -> torch.Tensor:
    """Apply the fusion rule ``x_hat = b + g * d``.

    ``gate == 0`` returns exactly the Base; ``gate == 1`` returns exactly the
    ungated candidate. ``gate`` may be a scalar or a tensor broadcastable to
    the output grid (e.g. a patch gate map).
    """
    return base_output + gate * proposal
