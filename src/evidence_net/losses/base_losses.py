"""Base reconstruction losses.

A configurable composite loss with four differentiable terms:

- **pixel**: L1 or L2 difference.
- **structural**: windowed local-structure similarity (mean/variance of
  local windows; 1 = identical per window, averaged).
- **edge**: L1 on Sobel gradient magnitudes.
- **frequency**: relative band-power difference computed in the FFT domain.

Weights come from ``training.config.LossConfig``; all terms are computed on
``[0, 1]`` tensors on the same grid.
"""

from __future__ import annotations

import torch
from torch import nn

from evidence_net.training.config import LossConfig

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).reshape(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).reshape(1, 1, 3, 3)

FREQUENCY_BANDS = ((0.0, 1 / 8), (1 / 8, 1 / 2), (1 / 2, 1.0))


def pixel_loss(prediction: torch.Tensor, target: torch.Tensor, *, kind: str = "l1") -> torch.Tensor:
    """L1 (default) or L2 pixel loss between prediction and target."""
    if kind == "l1":
        return nn.functional.l1_loss(prediction, target)
    if kind == "l2":
        return nn.functional.mse_loss(prediction, target)
    raise ValueError(f"unknown pixel loss kind: {kind}")


def _window_mean(array: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    """Local mean over square windows (zero-padded, normalized)."""
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=array.device, dtype=array.dtype)
    summed = nn.functional.conv2d(array, kernel, padding=kernel_size // 2)
    return summed / (kernel_size * kernel_size)


def structural_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Windowed local-structure loss: mean |corr-like| over local windows.

    For each window we compare (a) the mean offset and (b) the local standard
    deviation, giving a differentiable, bounded term: 0 for identical images.
    """
    mean_p = _window_mean(prediction)
    mean_t = _window_mean(target)
    std_p = (_window_mean(prediction**2) - mean_p**2).clamp(min=1e-8).sqrt()
    std_t = (_window_mean(target**2) - mean_t**2).clamp(min=1e-8).sqrt()
    mean_term = (mean_p - mean_t).abs().mean()
    std_term = (std_p - std_t).abs().mean()
    return mean_term + std_term


def edge_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between Sobel gradient magnitudes (edges must be preserved)."""
    device = prediction.device
    sobel_x = _SOBEL_X.to(device)
    sobel_y = _SOBEL_Y.to(device)
    gx_p = nn.functional.conv2d(prediction, sobel_x, padding=1)
    gy_p = nn.functional.conv2d(prediction, sobel_y, padding=1)
    gx_t = nn.functional.conv2d(target, sobel_x, padding=1)
    gy_t = nn.functional.conv2d(target, sobel_y, padding=1)
    # sqrt is non-differentiable at 0 (flat regions); epsilon keeps gradients finite.
    epsilon = 1e-8
    magnitude_p = torch.sqrt(gx_p**2 + gy_p**2 + epsilon)
    magnitude_t = torch.sqrt(gx_t**2 + gy_t**2 + epsilon)
    return (magnitude_p - magnitude_t).abs().mean()


def _band_power(tensor: torch.Tensor, radial: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    power = torch.fft.rfft2(tensor).abs() ** 2
    mask = (radial >= lo) & (radial < hi)
    if not mask.any():
        return torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
    # power is (B, 1, H, W//2+1); mask is (H, W//2+1). Flatten both.
    flat_power = power.reshape(power.shape[0], -1)
    flat_mask = mask.reshape(-1)
    return flat_power[:, flat_mask].mean()


def _radial_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    freq_y = torch.fft.fftfreq(height, device=device)[:, None]
    freq_x = torch.fft.rfftfreq(width, device=device)[None, :]
    radial = torch.sqrt(freq_y**2 + freq_x**2)
    return radial / 0.5  # normalized to Nyquist


def frequency_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean relative band-power difference over the contract frequency bands."""
    height, width = target.shape[-2:]
    radial = _radial_grid(height, width, target.device)
    terms: list[torch.Tensor] = []
    for lo, hi in FREQUENCY_BANDS:
        power_t = _band_power(target, radial, lo, hi)
        power_p = _band_power(prediction, radial, lo, hi)
        if float(power_t) > 1e-12:
            terms.append(((power_p - power_t) / power_t).abs())
    if not terms:
        return torch.tensor(0.0, device=target.device, dtype=target.dtype)
    return torch.stack(terms).mean()


class BaseLoss(nn.Module):
    """Weighted composite loss (pixel + structural + edge + frequency)."""

    def __init__(self, weights: LossConfig) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.compute(prediction, target)

    def compute(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
        total = total + self.weights.pixel * pixel_loss(prediction, target)
        if self.weights.structural > 0.0:
            total = total + self.weights.structural * structural_loss(prediction, target)
        if self.weights.edge > 0.0:
            total = total + self.weights.edge * edge_loss(prediction, target)
        if self.weights.frequency > 0.0:
            total = total + self.weights.frequency * frequency_loss(prediction, target)
        return total

    def components(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """Per-term weighted values (for reporting; no gradient graph kept)."""
        with torch.no_grad():
            return {
                "pixel": float(self.weights.pixel * pixel_loss(prediction, target)),
                "structural": float(self.weights.structural * structural_loss(prediction, target)),
                "edge": float(self.weights.edge * edge_loss(prediction, target)),
                "frequency": float(self.weights.frequency * frequency_loss(prediction, target)),
            }


class ProposalLoss(BaseLoss):
    """Composite loss plus fidelity to the target residual (Phase 4).

    ``total = BaseLoss(candidate, x) + residual * L1(d, x - stopgrad(b))``

    The residual term gives the proposer a direct gradient toward the target
    residual ``d* = x - b`` (per product definition 10.3), which the pure
    composite candidate loss lacks when the frozen Base already minimizes it.
    ``b`` is read from the wrapper's cached forward pass (detached) and ``d``
    from the same pass (with its graph), so the trainer's ``(pred, target)``
    interface needs no changes.
    """

    def __init__(self, weights: LossConfig, proposal_model) -> None:
        super().__init__(weights)
        self.proposal_model = proposal_model

    def compute(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = super().compute(prediction, target)
        if self.weights.residual > 0.0:
            proposal = self.proposal_model.last_proposal
            base_output = self.proposal_model.last_base
            residual = target - base_output  # d* = x - stopgrad(b)
            total = total + self.weights.residual * nn.functional.l1_loss(proposal, residual)
        return total
