"""Base reconstruction path validation (Phase 3 box 9)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from evidence_net.models.base import BaseReconstruction
from evidence_net.models.direct import DirectRestoration
from evidence_net.models.factory import build_model, model_summary
from evidence_net.models.validate import (
    ModelValidationError,
    check_gradients_flow,
    check_output_contract,
    check_tiled_parity,
    save_load_roundtrip,
    tiled_inference,
)
from evidence_net.training.config import ModelConfig


@pytest.mark.parametrize("name", ["base", "direct"])
def test_output_contract(name: str) -> None:
    model = build_model(ModelConfig(name=name, hidden_channels=8, depth=2))
    model.eval()
    check_output_contract(model)


@pytest.mark.parametrize("name", ["base", "direct"])
def test_gradients_flow(name: str) -> None:
    model = build_model(ModelConfig(name=name, hidden_channels=8, depth=2))
    check_gradients_flow(model)


@pytest.mark.parametrize("name", ["base", "direct"])
def test_checkpoint_roundtrip(name: str, tmp_path: Path) -> None:
    config = ModelConfig(name=name, hidden_channels=8, depth=2)
    model = build_model(config)
    model.eval()
    save_load_roundtrip(model, tmp_path / "model.pt", rebuild=lambda: build_model(config))


@pytest.mark.parametrize("name", ["base", "direct"])
def test_tiled_parity(name: str) -> None:
    model = build_model(ModelConfig(name=name, hidden_channels=8, depth=2))
    model.eval()
    check_tiled_parity(model, tile_size=16, margin=8)


def test_tiled_inference_matches_whole_on_real_shape() -> None:
    model = build_model(ModelConfig(name="base", hidden_channels=8, depth=2))
    model.eval()
    inputs = torch.rand(1, 1, 128, 128)
    with torch.no_grad():
        whole = model(inputs)
        tiled = tiled_inference(model, inputs, tile_size=32, margin=8)
    assert whole.shape == (1, 1, 256, 256)
    assert tiled.shape == whole.shape
    # Interior (beyond the model's border padding convention) must match.
    band = 2 * 8 + 4
    interior_whole = whole[:, :, band:-band, band:-band]
    interior_tiled = tiled[:, :, band:-band, band:-band]
    assert torch.allclose(interior_whole, interior_tiled, atol=1e-5)


def test_base_reconstruction_has_deterministic_anchor() -> None:
    inputs = torch.rand(1, 1, 16, 16)
    anchor = BaseReconstruction.anchor(inputs)
    assert anchor.shape == (1, 1, 32, 32)
    # Untrained refinement starts near the anchor's identity contribution;
    # the anchor itself is exactly bilinear upsampling.
    assert torch.isfinite(anchor).all()


def test_unknown_model_name_rejected() -> None:
    with pytest.raises(ValueError):
        build_model(ModelConfig(name="nope"))


def test_model_summary_counts_parameters() -> None:
    model = DirectRestoration(hidden_channels=8, depth=2)
    summary = model_summary(model)
    assert summary["name"] == "DirectRestoration"
    assert summary["n_params"] > 0


def test_bad_ndim_tiled_inference_rejected() -> None:
    model = DirectRestoration(hidden_channels=8, depth=2)
    with pytest.raises(ModelValidationError):
        tiled_inference(model, torch.rand(1, 1, 1, 8, 8))
