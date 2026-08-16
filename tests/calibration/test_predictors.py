"""Benefit predictor behavior tests (support-definition-v1, Phase 5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from evidence_net.benefit.labels import OUTPUT_GRID, PATCH_GRID, PATCH_SIZE
from evidence_net.benefit.predictors import (
    AttentionGateBaseline,
    LocalSignalBaseline,
    MinimalBenefitPredictor,
    PredictorError,
    ResidualMagnitudeBaseline,
    apply_baseline,
    patch_features,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Degraded input, Base output, and proposal grids.

    Left half: degraded input carries structure the Base has not recovered
    (large input-base residual and a large proposal). Right half: stripes the
    Base already resolves (tiny residual, tiny proposal).
    """
    grid = OUTPUT_GRID
    y = np.full((grid, grid), 0.05)
    y[:, : grid // 2] = 0.5  # strong structure the Base misses
    y[:, grid // 2 :] = 0.05
    for column in range(grid // 2, grid, 8):
        y[:, column] = 0.9  # stripes the Base resolves
    base = np.clip(y, 0.0, 1.0)
    base[:, : grid // 2] = 0.2  # Base fails to recover the left half
    proposal = np.zeros((grid, grid))
    proposal[:, : grid // 2] = 0.04
    proposal[:, grid // 2 :] = 0.001
    return y, base, proposal


def test_residual_magnitude_scores_large_proposal() -> None:
    y, b, d = _fixture()
    predictor = ResidualMagnitudeBaseline()
    score = predictor.score(y, b, d)
    assert score.shape == (PATCH_GRID, PATCH_GRID)
    left = score[:, : PATCH_GRID // 2]
    right = score[:, PATCH_GRID // 2 :]
    assert left.mean() > right.mean() * 5


def test_local_signal_scores_structured_input() -> None:
    y, b, d = _fixture()
    predictor = LocalSignalBaseline()
    score = predictor.score(y, b, d)
    left = score[:, : PATCH_GRID // 2]
    right = score[:, PATCH_GRID // 2 :]
    # The left half has higher |input - base| signal (Base misses it).
    assert left.mean() > right.mean()


def test_apply_baseline_composes_gated_output() -> None:
    y, b, d = _fixture()
    result = apply_baseline(ResidualMagnitudeBaseline(), "s1", y, b, d)
    assert result.sample_id == "s1"
    assert result.gate.shape == (PATCH_GRID, PATCH_GRID)
    assert result.gated_output.shape == (OUTPUT_GRID, OUTPUT_GRID)
    assert np.all(result.gate >= 0.0) and np.all(result.gate <= 1.0)
    assert np.all(result.gated_output >= 0.0) and np.all(result.gated_output <= 1.0)
    # The gate is monotone in the score: the strongest patch gets the largest
    # gate and the gated output stays between Base and candidate there.
    strongest = np.unravel_index(np.argmax(result.score), result.score.shape)
    weakest = np.unravel_index(np.argmin(result.score), result.score.shape)
    assert result.gate[strongest] >= result.gate[weakest]
    row, col = strongest
    slice_rows = slice(row * PATCH_SIZE, (row + 1) * PATCH_SIZE)
    slice_cols = slice(col * PATCH_SIZE, (col + 1) * PATCH_SIZE)
    base_patch = b[slice_rows, slice_cols]
    candidate_patch = np.clip(b + d, 0.0, 1.0)[slice_rows, slice_cols]
    gated_patch = result.gated_output[slice_rows, slice_cols]
    between = np.minimum(base_patch, candidate_patch) - 1e-9 <= gated_patch
    assert np.all(between)
    assert np.all(gated_patch <= np.maximum(base_patch, candidate_patch) + 1e-9)


def test_grid_validation() -> None:
    with pytest.raises(PredictorError, match="share one shape"):
        ResidualMagnitudeBaseline().score(
            np.zeros((64, 64)), np.zeros((64, 64)), np.zeros((128, 128))
        )
    with pytest.raises(PredictorError, match="output grid"):
        ResidualMagnitudeBaseline().score(
            np.zeros((32, 32)), np.zeros((32, 32)), np.zeros((32, 32))
        )


def test_patch_features_shape_and_finite() -> None:
    y, b, d = _fixture()
    features = patch_features(y, b, d)
    assert features.shape == (PATCH_GRID, PATCH_GRID, 8)
    assert np.all(np.isfinite(features))


def test_minimal_predictor_forward_and_score() -> None:
    y, b, d = _fixture()
    predictor = MinimalBenefitPredictor(hidden_channels=8, depth=1)
    features = patch_features(y, b, d)
    tensor = torch.from_numpy(features)[None]
    logits = predictor(tensor)
    assert logits.shape == (1, PATCH_GRID, PATCH_GRID, 1)
    score = predictor.score(y, b, d)
    assert score.shape == (PATCH_GRID, PATCH_GRID)
    assert np.all(np.isfinite(score))


def test_minimal_predictor_learns_benefit_pattern() -> None:
    """Trained on calibration-style labels, the MLP separates the halves."""
    rng = np.random.default_rng(0)
    from evidence_net.benefit.labels import patch_benefit_labels

    features_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for _index in range(24):
        y, b, d = _fixture()
        y = y + rng.normal(0.0, 0.005, size=y.shape)
        target = np.clip(b + d, 0.0, 1.0)
        features_list.append(patch_features(y, b, d))
        labels_list.append(patch_benefit_labels(b, d, target).astype(np.float32))
    features = torch.from_numpy(np.stack(features_list))
    labels = torch.from_numpy(np.stack(labels_list))
    predictor = MinimalBenefitPredictor(hidden_channels=16, depth=2)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size=8, shuffle=True
    )
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(8):
        predictor.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(predictor(batch_features), batch_labels.unsqueeze(-1))
            loss.backward()
            optimizer.step()
    predictor.eval()

    y, b, d = _fixture()
    score = predictor.score(y, b, d)
    left = score[:, : PATCH_GRID // 2].mean()
    right = score[:, PATCH_GRID // 2 :].mean()
    assert left > right


def test_attention_gate_forward_shape() -> None:
    y, b, d = _fixture()
    model = AttentionGateBaseline(hidden_channels=8)
    stack = np.stack([y, b, d], axis=0)[None].astype(np.float32)
    logits = model(torch.from_numpy(stack))
    assert logits.shape == (1, 1, PATCH_GRID, PATCH_GRID)
    score = model.score(y, b, d)
    assert score.shape == (PATCH_GRID, PATCH_GRID)
    assert np.all(np.isfinite(score))


def test_attention_gate_rejects_bad_channels() -> None:
    model = AttentionGateBaseline(hidden_channels=8)
    with pytest.raises(PredictorError, match="share one shape"):
        model.score(np.zeros((64, 64)), np.zeros((64, 64)), np.zeros((128, 128)))
