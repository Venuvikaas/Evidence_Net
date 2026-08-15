"""Validation helpers for the base reconstruction path.

Implements the Phase 3 model checks: output dimensions and range, gradient
flow, checkpoint save/restore, and tiled parity (running a fully
convolutional model on overlapping tiles and stitching must equal running it
on the whole input). Tiled parity is what makes large-image inference
feasible later without changing outputs.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class ModelValidationError(AssertionError):
    """Raised when a model fails a path check."""


def check_output_contract(model: nn.Module, input_shape: tuple[int, int] = (128, 128)) -> None:
    """Output must be 1x{2H}x{2W}, finite, and within [0, 1]."""
    height, width = input_shape
    inputs = torch.rand(1, 1, height, width)
    with torch.no_grad():
        output = model(inputs)
    expected = (1, 1, 2 * height, 2 * width)
    if tuple(output.shape) != expected:
        raise ModelValidationError(f"output shape {tuple(output.shape)} != {expected}")
    if not torch.isfinite(output).all():
        raise ModelValidationError("output contains non-finite values")
    if float(output.min()) < 0.0 or float(output.max()) > 1.0:
        raise ModelValidationError(
            f"output range [{float(output.min())}, {float(output.max())}] outside [0, 1]"
        )


def check_gradients_flow(model: nn.Module) -> None:
    """A single loss step must produce finite gradients on all parameters."""
    inputs = torch.rand(1, 1, 16, 16, requires_grad=True)
    targets = torch.rand(1, 1, 32, 32)
    output = model(inputs)
    loss = nn.functional.l1_loss(output, targets)
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise ModelValidationError(f"parameter {name} received no gradient")
        if not torch.isfinite(parameter.grad).all():
            raise ModelValidationError(f"parameter {name} has non-finite gradient")


def save_load_roundtrip(model: nn.Module, path: Path, *, rebuild) -> None:
    """state_dict save/restore must reproduce identical outputs.

    ``rebuild`` recreates a fresh model of the same architecture (the model
    factory passes the config); the restored weights must reproduce the
    original outputs exactly.
    """
    inputs = torch.rand(1, 1, 16, 16)
    with torch.no_grad():
        before = model(inputs)
    state = model.state_dict()
    torch.save(state, path)
    restored = rebuild()
    restored.load_state_dict(torch.load(path, weights_only=True))
    restored.eval()
    with torch.no_grad():
        after = restored(inputs)
    if not torch.allclose(before, after):
        raise ModelValidationError("checkpoint roundtrip changed model outputs")


def tiled_inference(
    model: nn.Module, inputs: torch.Tensor, tile_size: int = 64, margin: int = 8
) -> torch.Tensor:
    """Stitch tile-wise inference over a large input.

    The full input is replicate-padded by ``margin`` (so interpolation inside
    the model sees the same edge content as whole-image inference, whose
    ``align_corners=False`` grids clamp to edge pixels), each tile plus its
    margin context is run through the model, and the margin is cropped from
    the output. For fully convolutional models the result equals whole-image
    inference when ``margin`` covers the receptive field.
    """
    if inputs.ndim != 4:
        raise ModelValidationError("tiled_inference expects a 4D batch tensor")
    _, _, height, width = inputs.shape
    padded_inputs = torch.nn.functional.pad(
        inputs, (margin, margin, margin, margin), mode="replicate"
    )
    output = torch.zeros((inputs.shape[0], 1, height * 2, width * 2))
    for row in range(0, height, tile_size):
        tile_height = min(tile_size, height - row)
        for col in range(0, width, tile_size):
            tile_width = min(tile_size, width - col)
            tile = padded_inputs[
                :,
                :,
                row : row + tile_height + 2 * margin,
                col : col + tile_width + 2 * margin,
            ]
            with torch.no_grad():
                tile_out = model(tile)
            # Interior output region: input rows [row, row + T) map to output
            # rows [2*row, 2*row + 2*T); the margin is cropped from the tile.
            cropped = tile_out[
                :,
                :,
                2 * margin : 2 * margin + 2 * tile_height,
                2 * margin : 2 * margin + 2 * tile_width,
            ]
            output[
                :,
                :,
                2 * row : 2 * row + 2 * tile_height,
                2 * col : 2 * col + 2 * tile_width,
            ] = cropped
    return output


def check_tiled_parity(model: nn.Module, tile_size: int = 32, margin: int = 8) -> None:
    """Interior tiled inference must match whole-image inference (parity).

    The image border band (the model's own zero-padding convention in
    whole-image runs) is excluded: tiled inference provides replicate context
    there, so the two conventions can differ within the receptive field. The
    interior — every pixel beyond the band — must match to float precision.
    """
    inputs = torch.rand(1, 1, 96, 96)
    band = 2 * margin + 4
    with torch.no_grad():
        whole = model(inputs)
        tiled = tiled_inference(model, inputs, tile_size=tile_size, margin=margin)
    interior_whole = whole[:, :, band:-band, band:-band]
    interior_tiled = tiled[:, :, band:-band, band:-band]
    if interior_whole.shape != interior_tiled.shape:
        raise ModelValidationError("tiled inference produced a different shape")
    if not torch.allclose(interior_whole, interior_tiled, atol=1e-5):
        difference = float((interior_whole - interior_tiled).abs().max())
        raise ModelValidationError(f"tiled parity mismatch (max abs diff {difference})")
