"""Tests for the composite base reconstruction losses."""

from __future__ import annotations

import pytest
import torch

from evidence_net.losses.base_losses import (
    BaseLoss,
    edge_loss,
    frequency_loss,
    pixel_loss,
    structural_loss,
)
from evidence_net.training.config import LossConfig


def constant(value: float, size: int = 16) -> torch.Tensor:
    return torch.full((1, 1, size, size), value)


def random_image(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand(1, 1, 16, 16)


def test_pixel_loss_identical_is_zero() -> None:
    image = random_image()
    assert pixel_loss(image, image) == 0.0
    assert pixel_loss(image, image, kind="l2") == 0.0


def test_pixel_loss_constant_offset_exact() -> None:
    offset = constant(0.25)
    assert pixel_loss(constant(0.0), offset) == pytest.approx(0.25)
    assert pixel_loss(constant(0.0), offset, kind="l2") == pytest.approx(0.0625)


def test_structural_loss_identical_is_zero() -> None:
    image = random_image()
    assert structural_loss(image, image) == pytest.approx(0.0, abs=1e-6)


def test_edge_loss_identical_is_zero() -> None:
    image = random_image()
    assert edge_loss(image, image) == pytest.approx(0.0, abs=1e-6)


def test_frequency_loss_identical_is_zero() -> None:
    image = random_image()
    assert frequency_loss(image, image) == pytest.approx(0.0, abs=1e-6)


def test_composite_identical_is_zero() -> None:
    image = random_image()
    loss = BaseLoss(LossConfig())
    assert loss(image, image) == pytest.approx(0.0, abs=1e-6)


def test_composite_differentiates() -> None:
    loss = BaseLoss(LossConfig())
    prediction = random_image(1).requires_grad_(True)
    target = random_image(2)
    value = loss(prediction, target)
    value.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_composite_components_report() -> None:
    loss = BaseLoss(LossConfig(pixel=1.0, structural=0.5, edge=0.1, frequency=0.0))
    components = loss.components(random_image(1), random_image(2))
    assert set(components) == {"pixel", "structural", "edge", "frequency"}
    assert all(value >= 0.0 for value in components.values())


def test_zero_weight_term_skipped() -> None:
    # Only pixel weight set: structural/edge/frequency must not affect result.
    loss = BaseLoss(LossConfig(pixel=1.0, structural=0.0, edge=0.0, frequency=0.0))
    prediction = random_image(1)
    target = random_image(2)
    assert loss(prediction, target) == pytest.approx(pixel_loss(prediction, target))
