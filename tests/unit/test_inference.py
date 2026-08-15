"""Tests for the baseline inference pipeline (inference/baseline.py)."""

from __future__ import annotations

import numpy as np
import pytest

from evidence_net.inference.baseline import (
    RestorerResult,
    evaluate_restorer,
    evaluate_restorers,
    run_restorer,
)
from evidence_net.models.reference import deterministic_reconstruction


def constant(value: float, size: int = 8) -> np.ndarray:
    return np.full((size, size), value, dtype=np.float64)


def test_run_restorer_clips_outputs() -> None:
    inputs = [np.full((8, 8), 2.0), np.full((8, 8), -1.0)]

    def out_of_range(_array: np.ndarray) -> np.ndarray:
        return _array * 0.5  # 1.0 and -0.5 after scaling

    predictions = run_restorer(inputs, out_of_range)
    assert predictions[0].min() == 1.0
    assert predictions[0].max() == 1.0
    assert predictions[1].min() == 0.0


def test_run_restorer_preserves_in_range() -> None:
    inputs = [constant(0.25), constant(0.75)]
    predictions = run_restorer(inputs, deterministic_reconstruction)
    assert predictions[0].shape == (16, 16)
    assert np.allclose(predictions[0], 0.25)
    assert np.allclose(predictions[1], 0.75)


def test_evaluate_restorer_shape_and_aggregates() -> None:
    # Deterministic reconstruction doubles the grid: 8x8 input -> 16x16.
    inputs = [constant(0.0), constant(0.5)]
    targets = [constant(0.0, 16), constant(0.5, 16)]
    result = evaluate_restorer(
        "deterministic",
        inputs,
        targets,
        ["g0", "g1"],
        deterministic_reconstruction,
        n_boot=100,
        seed=0,
    )
    assert isinstance(result, RestorerResult)
    assert result.name == "deterministic"
    assert set(result.per_group_metrics) == {"g0", "g1"}
    assert "psnr" in result.aggregates
    assert result.aggregates["psnr"]["mean"] == float("inf")
    assert result.aggregates["ssim"]["mean"] == pytest.approx(1.0)
    assert result.aggregates["mae"]["mean"] == 0.0


def test_evaluate_restorer_known_mae() -> None:
    inputs = [constant(0.0)]
    targets = [constant(0.5)]
    result = evaluate_restorer(
        "identity",
        inputs,
        targets,
        ["g0"],
        lambda array: array,
        n_boot=50,
        seed=0,
    )
    assert result.aggregates["mae"]["mean"] == pytest.approx(0.5)
    assert result.aggregates["psnr"]["mean"] == pytest.approx(float(10.0 * np.log10(1.0 / 0.25)))


def test_evaluate_restorers_paired_comparison() -> None:
    inputs = [constant(0.0, 4), constant(0.5, 4)]
    targets = [constant(0.25, 4), constant(0.5, 4)]
    results = evaluate_restorers(
        inputs,
        targets,
        ["g0", "g1"],
        {"identity": lambda array: array, "half": lambda array: array * 0.5},
        n_boot=50,
        seed=1,
    )
    assert set(results) == {"identity", "half"}
    for result in results.values():
        assert set(result.per_group_metrics) == {"g0", "g1"}
    # Identity: MAE 0.25 on g0, 0 on g1 -> mean 0.125.
    assert results["identity"].aggregates["mae"]["mean"] == pytest.approx(0.125)
    # Half: MAE 0.25 on both -> mean 0.25.
    assert results["half"].aggregates["mae"]["mean"] == pytest.approx(0.25)
