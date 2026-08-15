"""Target residual generation (Phase 4, box 3).

The proposal is trained against the residual between the clean target and the
frozen Base output:

    d* = x - stopgrad(b)

The stop-gradient is essential: the Base parameters never receive gradients
from proposal training (they are frozen in ``BoundedDetailProposal``), so
``b`` is detached here as well and the residual target is treated as a fixed
label. Both tensor-level and dataset-level helpers are provided.
"""

from __future__ import annotations

from collections.abc import Sequence, Sized
from typing import cast

import torch
from torch import nn
from torch.utils.data import Dataset


def residual_target(target: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
    """``x - stopgrad(b)`` on the output grid; both inputs already on the grid."""
    return target.detach() - base_output.detach()


def residual_targets(
    targets: Sequence[torch.Tensor], base_outputs: Sequence[torch.Tensor]
) -> list[torch.Tensor]:
    """Elementwise residual targets for a batch of (target, base) pairs."""
    return [
        residual_target(target, base_output)
        for target, base_output in zip(targets, base_outputs, strict=True)
    ]


class ResidualTargetDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Pairs a source dataset of (input, target, id) with frozen Base outputs.

    Yields ``(input, residual_target, sample_id)`` so the trainer can fit the
    proposer directly to the residual without recomputing the frozen Base per
    epoch. The Base must be in eval mode and is called once per sample.
    """

    def __init__(
        self,
        source: Dataset[tuple[torch.Tensor, torch.Tensor, str]],
        base: nn.Module,
        *,
        device: torch.device | None = None,
    ) -> None:
        self.source = source
        self.base = base
        self.device = device or torch.device("cpu")
        self.base.to(self.device)
        self.base.eval()

    def __len__(self) -> int:
        return len(cast(Sized, self.source))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        inputs, target, sample_id = self.source[index]
        with torch.no_grad():
            base_output = self.base(inputs.to(self.device)).cpu()
        residual = residual_target(target, base_output)
        return inputs, residual, sample_id
