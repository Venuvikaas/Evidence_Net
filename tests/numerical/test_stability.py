"""Model stability diagnostic tests (Phase 8, stability-v1).

Analytical fixtures: identity perturbation measures zero deviation, flips
commute with the bilinear anchor, shifts expose subpixel variance, identical
models agree perfectly, error-diversity metrics behave on controlled error
maps, and the diversity guard admits only measured-diverse models.
"""

from __future__ import annotations

import numpy as np
import pytest

from evidence_net.evaluation.statistics import GroupingError
from evidence_net.models.reference import deterministic_reconstruction
from evidence_net.stress_tests.stability import (
    ShiftPerturbation,
    StabilityConfig,
    StabilityError,
    add_if_diverse,
    build_perturbations,
    checkpoint_agreement,
    error_diversity,
    perturbation_stability,
)

GRID = 32


def _inputs(n: int = 4, seed: int = 0) -> tuple[list[np.ndarray], list[str]]:
    rng = np.random.default_rng(seed)
    inputs = [rng.random((GRID, GRID)) for _ in range(n)]
    return inputs, [f"g{i:02d}" for i in range(n)]


def _identity_fn(image: np.ndarray) -> np.ndarray:
    """Deterministic model: bilinear 2x up-sample (shift-variant, flip-equivariant)."""
    return deterministic_reconstruction(image)


# --- Perturbation family ---------------------------------------------------


def test_zero_shift_is_identity_with_zero_deviation() -> None:
    inputs, group_ids = _inputs()
    report = perturbation_stability(
        _identity_fn, inputs, group_ids, [ShiftPerturbation(0, 0)], n_boot=20
    )
    assert report.results[0].aggregate.mean == pytest.approx(0.0, abs=1e-12)
    assert report.results[0].aggregate.n_groups == len(inputs)


def test_shift_exposes_subpixel_variance() -> None:
    # The bilinear anchor is shift-variant: shifting the input and inverting
    # the output changes the result, so the deviation is positive.
    inputs, group_ids = _inputs()
    report = perturbation_stability(
        _identity_fn, inputs, group_ids, [ShiftPerturbation(0, 1)], n_boot=20
    )
    assert report.results[0].aggregate.mean > 0.0


def test_flips_commute_with_bilinear_anchor() -> None:
    # The deterministic bilinear anchor is flip-equivariant, so inverting the
    # flipped output recovers the unperturbed output almost exactly.
    inputs, group_ids = _inputs()
    perturbations = build_perturbations(StabilityConfig(shifts=((0, 0),)))
    report = perturbation_stability(_identity_fn, inputs, group_ids, perturbations, n_boot=20)
    flips = [result for result in report.results if result.perturbation.startswith("flip")]
    assert len(flips) == 2
    for result in flips:
        assert result.aggregate.mean < 1e-6


def test_perturbation_report_shows_distribution() -> None:
    inputs, group_ids = _inputs()
    config = StabilityConfig()
    report = perturbation_stability(
        _identity_fn, inputs, group_ids, build_perturbations(config), n_boot=20
    )
    names = [result.perturbation for result in report.results]
    assert names == ["shift-0-0", "shift-0-1", "shift-1-0", "shift-1-1", "flip-h", "flip-v"]
    assert report.across["n_perturbations"] == len(names)
    assert report.across["max_mean_deviation"] >= 0.0
    assert report.across["argmax_perturbation"] in names


def test_perturbation_report_rejects_duplicate_groups() -> None:
    inputs, group_ids = _inputs()
    with pytest.raises(GroupingError):
        perturbation_stability(
            _identity_fn, inputs, group_ids + group_ids, [ShiftPerturbation(0, 0)], n_boot=10
        )


def test_shift_bounds_are_enforced() -> None:
    with pytest.raises(StabilityError):
        ShiftPerturbation(2, 0)
    with pytest.raises(StabilityError):
        ShiftPerturbation(0, -2)
    with pytest.raises(StabilityError):
        StabilityConfig(shifts=((0, 2),)).validate()
    with pytest.raises(StabilityError):
        StabilityConfig(flips=("z",)).validate()
    with pytest.raises(StabilityError):
        StabilityConfig(min_diversity_threshold=1.5).validate()


# --- Checkpoint agreement ----------------------------------------------------


def test_identical_models_agree_perfectly() -> None:
    inputs, group_ids = _inputs()
    models = [("same-a", _identity_fn), ("same-b", _identity_fn)]
    report = checkpoint_agreement(models, inputs, group_ids, n_boot=20)
    assert report.pairs["same-a"].mean == pytest.approx(0.0, abs=1e-12)
    assert report.pairs["same-a-vs-same-b"].mean == pytest.approx(0.0, abs=1e-12)
    assert report.across["max_pair_agreement"] == pytest.approx(0.0, abs=1e-12)


def _constant_fn(image: np.ndarray) -> np.ndarray:
    return np.full_like(deterministic_reconstruction(image), 0.5)


def test_different_models_disagree() -> None:
    inputs, group_ids = _inputs()
    report = checkpoint_agreement(
        [("anchor", _identity_fn), ("constant", _constant_fn)], inputs, group_ids, n_boot=20
    )
    assert report.pairs["anchor-vs-constant"].mean > 0.0
    assert set(report.pairs) == {"anchor", "anchor-vs-constant", "constant"}


def test_checkpoint_report_is_seeded_and_grouped() -> None:
    inputs, group_ids = _inputs()
    models = [("a", _identity_fn), ("b", _identity_fn)]
    first = checkpoint_agreement(models, inputs, group_ids, n_boot=20, seed=3)
    second = checkpoint_agreement(models, inputs, group_ids, n_boot=20, seed=3)
    assert first.as_dict() == second.as_dict()
    assert first.n_groups == len(group_ids)
    with pytest.raises(StabilityError):
        checkpoint_agreement([], inputs, group_ids, n_boot=10)


# --- Error diversity ----------------------------------------------------------


def _error_grid(values: float) -> np.ndarray:
    return np.full((8, 8), values)


def test_diversity_identical_errors() -> None:
    errors = {"a": _error_grid(0.1), "b": _error_grid(0.1)}
    metrics = error_diversity(errors)["a-vs-b"]
    assert metrics["error_correlation"] == pytest.approx(1.0)
    assert metrics["disagreement_rate"] == pytest.approx(0.0)
    assert metrics["complementarity"] == pytest.approx(0.0)


def test_diversity_opposite_signs() -> None:
    # Opposite signs but identical magnitudes: full sign disagreement, no
    # magnitude complementarity, perfectly (anti)correlated magnitudes.
    errors = {"a": _error_grid(0.2), "b": _error_grid(-0.2)}
    metrics = error_diversity(errors)["a-vs-b"]
    assert metrics["disagreement_rate"] == pytest.approx(1.0)
    assert metrics["complementarity"] == pytest.approx(0.0)
    assert metrics["error_correlation"] == pytest.approx(1.0)


def test_diversity_different_magnitudes_are_complementary() -> None:
    # Same sign, different magnitudes: complementary (different-magnitude
    # mistakes) but no sign disagreement.
    errors = {"a": _error_grid(0.2), "b": _error_grid(0.05)}
    metrics = error_diversity(errors)["a-vs-b"]
    assert metrics["complementarity"] == pytest.approx(1.0)
    assert metrics["disagreement_rate"] == pytest.approx(0.0)


def test_diversity_shape_mismatch_rejected() -> None:
    errors = {"a": np.zeros((4, 4)), "b": np.zeros((8, 8))}
    with pytest.raises(StabilityError):
        error_diversity(errors)


def test_diversity_guard_requires_measured_diversity() -> None:
    # Identical errors: not diverse enough below a 0.2 threshold.
    errors = {"existing": _error_grid(0.1), "candidate": _error_grid(0.1)}
    assert not add_if_diverse("candidate", errors, 0.2)
    # Opposite-sign errors: disagreement rate 1.0 >= 0.2.
    errors["candidate"] = _error_grid(-0.1)
    assert add_if_diverse("candidate", errors, 0.2)
    # A lone model is trivially included.
    assert add_if_diverse("solo", {"solo": _error_grid(0.1)}, 0.2)


def test_diversity_is_not_accuracy() -> None:
    # A perfectly correlated wrong pair is not diverse, even though both are
    # equally confident; the guard stays conservative.
    rng = np.random.default_rng(0)
    grid = rng.normal(size=(16, 16))
    errors = {"existing": grid, "candidate": grid + 0.001}
    metrics = error_diversity(errors)["candidate-vs-existing"]
    assert metrics["disagreement_rate"] < 0.01
    assert not add_if_diverse("candidate", errors, 0.2)
