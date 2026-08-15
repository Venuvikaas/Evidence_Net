"""Model factory: build a model from a validated ``ModelConfig``."""

from __future__ import annotations

from torch import nn

from evidence_net.models.base import BaseReconstruction
from evidence_net.models.direct import DirectRestoration
from evidence_net.training.config import ModelConfig


class ModelFactoryError(ValueError):
    """Raised when a model cannot be built from a config."""


def build_model(config: ModelConfig) -> nn.Module:
    """Build the model named by ``config.name`` with its parameters."""
    if config.name == "base":
        return BaseReconstruction(hidden_channels=config.hidden_channels, depth=config.depth)
    if config.name == "direct":
        return DirectRestoration(hidden_channels=config.hidden_channels, depth=config.depth)
    raise ModelFactoryError(f"unknown model name: {config.name}")


def model_summary(model: nn.Module) -> dict[str, int | str]:
    """Architecture summary for provenance (name, parameter count)."""
    return {
        "name": type(model).__name__,
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
    }
