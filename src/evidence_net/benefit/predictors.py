"""Benefit predictors (Phase 5, support-definition-v1).

A predictor maps **inference-time features** — the degraded input, the
frozen Base output, and the bounded proposal — to a per-patch score for the
benefit event. The ground truth is the deterministic label of
``support-definition-v1`` (never the target at inference).

Three declared simple baselines come first (Gate 4 requires the learned
predictor to beat them):

- ``ResidualMagnitudeBaseline`` — score grows with mean |proposal| per patch.
- ``LocalSignalBaseline`` — score grows with local input signal (patch std),
  encoding the EXP-004 finding that benefit concentrates in structured
  (non-flat) regions where the proposal is large.
- ``AttentionGateBaseline`` — a small reconstruction-trained CNN over the
  concatenated (input, base, proposal) grid predicting the patch gate.

The minimal learned predictor (``MinimalBenefitPredictor``) is an MLP over
per-patch feature vectors built from the same three grids. It is trained
separately from the proposal and Base models (Phase 5 box 12: two-stage).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from evidence_net.benefit.labels import PATCH_GRID, PATCH_SIZE

FEATURE_VERSION = "features-v1"


class PredictorError(ValueError):
    """Raised when a predictor cannot score the given grids."""


def _as_float64(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)


def _validate_grids(
    input_grid: np.ndarray, base_grid: np.ndarray, proposal_grid: np.ndarray
) -> None:
    shapes = {array.shape for array in (input_grid, base_grid, proposal_grid)}
    if len(shapes) != 1:
        raise PredictorError(
            f"input/base/proposal grids must share one shape, got "
            f"{input_grid.shape}, {base_grid.shape}, {proposal_grid.shape}"
        )
    if input_grid.shape != (PATCH_GRID * PATCH_SIZE, PATCH_GRID * PATCH_SIZE):
        raise PredictorError(
            f"predictors are defined on the {PATCH_GRID * PATCH_SIZE}x"
            f"{PATCH_GRID * PATCH_SIZE} output grid, got {input_grid.shape}"
        )


def _patch_scores(array: np.ndarray, reducer: str = "mean") -> np.ndarray:
    """Reduce a grid to per-patch (16x16) scores."""
    values = _as_float64(array)
    scores = np.zeros((PATCH_GRID, PATCH_GRID), dtype=np.float64)
    for row in range(PATCH_GRID):
        for col in range(PATCH_GRID):
            patch = values[
                row * PATCH_SIZE : (row + 1) * PATCH_SIZE,
                col * PATCH_SIZE : (col + 1) * PATCH_SIZE,
            ]
            if reducer == "mean":
                scores[row, col] = float(np.mean(np.abs(patch)))
            elif reducer == "std":
                scores[row, col] = float(np.std(patch))
            else:  # pragma: no cover - internal
                raise PredictorError(f"unknown reducer {reducer}")
    return scores


class Predictor:
    """Interface: per-patch score on the output grid, higher = more benefit."""

    name: str = "predictor"

    def score(
        self, input_grid: np.ndarray, base_grid: np.ndarray, proposal_grid: np.ndarray
    ) -> np.ndarray:
        """Return a (16, 16) float score map (higher -> more likely benefit)."""
        raise NotImplementedError


class ResidualMagnitudeBaseline(Predictor):
    """Score = mean |proposal| per patch (a declared simple heuristic)."""

    name = "residual-magnitude"

    def score(
        self, input_grid: np.ndarray, base_grid: np.ndarray, proposal_grid: np.ndarray
    ) -> np.ndarray:
        _validate_grids(input_grid, base_grid, proposal_grid)
        return _patch_scores(proposal_grid, reducer="mean")


class LocalSignalBaseline(Predictor):
    """Score = mean |input - base| signal per patch.

    Encodes the EXP-004 finding that benefit concentrates where the degraded
    input carries structure the Base must recover (flat regions are already
    solved by Base and the proposal rarely helps there).
    """

    name = "local-signal"

    def score(
        self, input_grid: np.ndarray, base_grid: np.ndarray, proposal_grid: np.ndarray
    ) -> np.ndarray:
        _validate_grids(input_grid, base_grid, proposal_grid)
        return _patch_scores(input_grid - base_grid, reducer="mean")


def _logit_to_patch_gate(score: np.ndarray) -> np.ndarray:
    """Sigmoid a raw score map into a patch gate in [0, 1]."""
    return 1.0 / (1.0 + np.exp(-np.clip(score, -30.0, 30.0)))


def _gate_output(base_grid: np.ndarray, proposal_grid: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Compose ``x_hat = b + g * d`` on the output grid from a patch gate."""
    gate_map = np.repeat(np.repeat(gate, PATCH_SIZE, axis=0), PATCH_SIZE, axis=1)
    return np.clip(base_grid + gate_map * proposal_grid, 0.0, 1.0)


@dataclass(frozen=True)
class BaselineResult:
    """Score map and the corresponding gated output for one sample."""

    sample_id: str
    predictor: str
    score: np.ndarray  # (16, 16)
    gate: np.ndarray  # (16, 16) in [0, 1]
    gated_output: np.ndarray  # output grid


def apply_baseline(
    predictor: Predictor,
    sample_id: str,
    input_grid: np.ndarray,
    base_grid: np.ndarray,
    proposal_grid: np.ndarray,
) -> BaselineResult:
    """Score one sample and compose the gate-gated output."""
    score = predictor.score(input_grid, base_grid, proposal_grid)
    gate = _logit_to_patch_gate(score)
    gated = _gate_output(base_grid, proposal_grid, gate)
    return BaselineResult(
        sample_id=sample_id,
        predictor=predictor.name,
        score=score,
        gate=gate,
        gated_output=gated,
    )


def patch_features(
    input_grid: np.ndarray,
    base_grid: np.ndarray,
    proposal_grid: np.ndarray,
) -> np.ndarray:
    """Per-patch feature vector (16, 16, F) for the minimal predictor.

    Features per patch (versioned ``features-v1``):

    - mean and std of the input patch,
    - mean and std of the base patch,
    - mean and std of the proposal patch,
    - mean |input - base| (local signal),
    - benefit-fraction prior of the patch row (structural context).
    """
    _validate_grids(input_grid, base_grid, proposal_grid)
    features = np.zeros((PATCH_GRID, PATCH_GRID, 8), dtype=np.float32)
    for row in range(PATCH_GRID):
        for col in range(PATCH_GRID):
            y = _as_float64(
                input_grid[
                    row * PATCH_SIZE : (row + 1) * PATCH_SIZE,
                    col * PATCH_SIZE : (col + 1) * PATCH_SIZE,
                ]
            )
            b = _as_float64(
                base_grid[
                    row * PATCH_SIZE : (row + 1) * PATCH_SIZE,
                    col * PATCH_SIZE : (col + 1) * PATCH_SIZE,
                ]
            )
            d = _as_float64(
                proposal_grid[
                    row * PATCH_SIZE : (row + 1) * PATCH_SIZE,
                    col * PATCH_SIZE : (col + 1) * PATCH_SIZE,
                ]
            )
            features[row, col] = [
                float(np.mean(y)),
                float(np.std(y)),
                float(np.mean(b)),
                float(np.std(b)),
                float(np.mean(d)),
                float(np.std(d)),
                float(np.mean(np.abs(y - b))),
                float(row) / float(PATCH_GRID),
            ]
    return features


class MinimalBenefitPredictor(nn.Module):
    """Small MLP over per-patch features predicting the benefit event.

    Trained separately from the proposal and Base models (two-stage, Phase 5
    box 12). ``forward`` returns per-patch logits on the 16x16 grid.
    """

    def __init__(self, hidden_channels: int = 32, depth: int = 2) -> None:
        super().__init__()
        if hidden_channels < 4:
            raise ValueError("hidden_channels must be >= 4")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = [nn.Linear(8, hidden_channels), nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True)])
        layers.append(nn.Linear(hidden_channels, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, 16, 16, F) -> logits (B, 16, 16, 1)."""
        batch, rows, cols, channels = features.shape
        logits = self.net(features.reshape(batch * rows * cols, channels))
        return logits.reshape(batch, rows, cols, 1)

    @torch.no_grad()
    def score(
        self, input_grid: np.ndarray, base_grid: np.ndarray, proposal_grid: np.ndarray
    ) -> np.ndarray:
        """Per-patch logit score map (16, 16) for one sample."""
        features = patch_features(input_grid, base_grid, proposal_grid)
        tensor = torch.from_numpy(features)[None]
        logits = self.forward(tensor)
        return logits[0, :, :, 0].numpy()


class AttentionGateBaseline(nn.Module):
    """Small reconstruction-trained CNN over (input, base, proposal) grids.

    Predicts the patch gate directly from the concatenated output-grid
    channels; trained with the same composite loss family as the proposal so
    the gate learns where gating helps. Exposed through the ``Predictor``
    interface for the Gate 4 comparison.
    """

    def __init__(self, hidden_channels: int = 16) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.pool = nn.AvgPool2d(kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, channels: torch.Tensor) -> torch.Tensor:
        """channels: (B, 3, H, W) -> patch logits (B, 1, 16, 16)."""
        logits = self.net(channels)
        return self.pool(logits)

    @torch.no_grad()
    def score(
        self, input_grid: np.ndarray, base_grid: np.ndarray, proposal_grid: np.ndarray
    ) -> np.ndarray:
        """Per-patch logit score map (16, 16) for one sample."""
        _validate_grids(input_grid, base_grid, proposal_grid)
        stack = np.stack(
            [
                _as_float64(input_grid),
                _as_float64(base_grid),
                _as_float64(proposal_grid),
            ],
            axis=0,
        )[None]
        tensor = torch.from_numpy(stack.astype(np.float32))
        logits = self.forward(tensor)
        return logits[0, 0].numpy()
