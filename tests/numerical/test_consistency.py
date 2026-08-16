"""Measurement-consistency report tests (Phase 7, forward-model-v1).

Verifies the compatibility semantics: per-operator residual distribution
(min / median / max, never minimum only), grouped bootstrap discipline,
seeded reproducibility, per-image feature extraction for Lane A, and the
diagnostic's ordering sanity (the true generator operator scores best).
"""

from __future__ import annotations

import numpy as np
import pytest

from evidence_net.evaluation.statistics import GroupingError
from evidence_net.stress_tests.consistency import (
    ConsistencyReport,
    build_consistency_report,
    measure_noise_variance,
    per_image_residuals,
)
from evidence_net.stress_tests.forward import (
    BlurDownsample,
    ForwardConfig,
    NoisyBlurDownsample,
    build_operator_family,
)

GRID = 64


def _sample_set(n: int = 6, seed: int = 0) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    rng = np.random.default_rng(seed)
    # True generator inside the family; observations and anchor restorations.
    generator = NoisyBlurDownsample(blur_sigma=0.6, noise_sigma=0.01, seed=seed)
    observations: list[np.ndarray] = []
    restored: list[np.ndarray] = []
    ids: list[str] = []
    for i in range(n):
        clean = rng.random((GRID, GRID))
        observation = generator.apply(clean, rng)
        observations.append(observation)
        restored.append(clean)  # perfect restoration; only re-degradation noise remains
        ids.append(f"g{i:02d}")
    return restored, observations, ids


def test_report_covers_every_operator_and_kind() -> None:
    restored, observations, ids = _sample_set()
    report = build_consistency_report(
        restored, observations, ids, build_operator_family(ForwardConfig()), n_boot=50
    )
    assert isinstance(report, ConsistencyReport)
    names = [entry.operator for entry in report.operators]
    assert names == ["area", "bilinear", "blur", "noisy-blur"]
    assert {entry.kind for entry in report.operators} == {"deterministic", "stochastic"}
    assert report.n_groups == len(ids)


def test_report_shows_distribution_not_minimum_only() -> None:
    restored, observations, ids = _sample_set()
    report = build_consistency_report(
        restored, observations, ids, build_operator_family(ForwardConfig()), n_boot=50
    )
    across = report.across_operators
    for key in ("min_mae", "median_mae", "max_mae", "argmin_operator", "n_operators"):
        assert key in across
    assert across["n_operators"] == 4
    assert across["min_mae"] <= across["median_mae"] <= across["max_mae"]
    assert across["argmin_operator"] in [entry.operator for entry in report.operators]


def test_report_is_seeded_and_deterministic() -> None:
    restored, observations, ids = _sample_set()
    operators = build_operator_family(ForwardConfig())
    first = build_consistency_report(restored, observations, ids, operators, n_boot=50, seed=3)
    second = build_consistency_report(restored, observations, ids, operators, n_boot=50, seed=3)
    assert first.as_dict() == second.as_dict()


def test_report_aggregates_by_group_not_pixel() -> None:
    restored, observations, ids = _sample_set(n=2)
    operators = build_operator_family(ForwardConfig())
    # One value per group id: fine.
    report = build_consistency_report(restored, observations, ids, operators, n_boot=20)
    assert report.n_groups == 2
    # Duplicate group ids are rejected.
    with pytest.raises(GroupingError):
        build_consistency_report(restored, observations, ids * 2, operators, n_boot=20)
    # Length mismatches are rejected.
    with pytest.raises(GroupingError):
        build_consistency_report(restored, observations, ids[:-1], operators, n_boot=20)


def test_true_generator_scores_best_on_perfect_restoration() -> None:
    # With a perfect restoration the observation re-degrades losslessly through
    # the operator family member that produced it (zero noise): the arg-min
    # operator must be the true path (blur == noisy-blur at zero noise), and
    # the coarser area/bilinear operators must score worse.
    rng = np.random.default_rng(0)
    clean = rng.random((GRID, GRID))
    # The generator uses the family's own parameters at zero noise, so the
    # observation re-degrades losslessly through the true family member.
    config = ForwardConfig()
    generator = NoisyBlurDownsample(blur_sigma=config.blur_sigma, noise_sigma=0.0, seed=0)
    observation = generator.apply(clean, rng)
    operators = build_operator_family(config)
    report = build_consistency_report([clean], [observation], ["g0"], operators, n_boot=10, seed=0)
    assert report.across_operators["argmin_operator"] in ("blur", "noisy-blur")
    assert report.across_operators["min_mae"] == pytest.approx(0.0, abs=1e-9)
    means = {entry.operator: entry.aggregate.mean for entry in report.operators}
    assert means["area"] > report.across_operators["min_mae"]
    assert means["bilinear"] > report.across_operators["min_mae"]


def test_stochastic_report_bias_and_aggregate_fields() -> None:
    restored, observations, ids = _sample_set()
    report = build_consistency_report(
        restored, observations, ids, build_operator_family(ForwardConfig()), n_boot=30
    )
    for entry in report.operators:
        assert set(entry.per_group_mae) == set(ids)
        assert entry.aggregate.n_groups == len(ids)
        assert entry.aggregate.n_boot == 30
        assert isinstance(entry.bias_mean, float)
        assert "interpretation" in report.as_dict()
    assert "compatibility" in report.interpretation.lower()
    assert "not truth" in report.interpretation.lower()


def test_per_image_residuals_are_feature_like() -> None:
    restored, observations, ids = _sample_set()
    operators = build_operator_family(ForwardConfig())
    features = per_image_residuals(restored, observations, ids, operators, seed=0)
    assert set(features) == set(ids)
    for values in features.values():
        assert set(values) == {operator.name for operator in operators}
        assert all(v >= 0.0 for v in values.values())


def test_measure_noise_variance_deterministic_is_zero() -> None:
    operator = BlurDownsample(blur_sigma=0.5)
    image = np.random.default_rng(0).random((GRID, GRID))
    spread = measure_noise_variance(operator, image, n_draws=16)
    assert spread["std"] == 0.0
    assert spread["max"] == 0.0


def test_measure_noise_variance_stochastic_is_positive() -> None:
    operator = NoisyBlurDownsample(blur_sigma=0.0, noise_sigma=0.05, seed=0)
    image = np.zeros((GRID, GRID))
    spread = measure_noise_variance(operator, image, n_draws=16, seed=1)
    assert spread["n_draws"] == 16
    assert spread["max"] >= spread["min"] >= 0.0
    assert spread["std"] > 0.0
