"""Calibration behavior tests (calibration-version-v1, Phase 5)."""

from __future__ import annotations

import numpy as np
import pytest

from evidence_net.benefit.calibration import (
    CalibrationError,
    brier_score,
    expected_calibration_error,
    fit_calibration,
    grouped_brier,
    reliability_curve,
)


def _perfect_scores(n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Well-separated scores with matching binary labels."""
    rng = np.random.default_rng(0)
    half = n // 2
    scores = np.concatenate([rng.normal(-2.0, 0.5, size=half), rng.normal(2.0, 0.5, size=half)])
    labels = np.concatenate([np.zeros(half), np.ones(half)])
    return scores, labels


def test_platt_maps_low_to_low_high_to_high() -> None:
    scores, labels = _perfect_scores()
    mapping = fit_calibration(scores, labels, split="calibration")
    assert mapping.method == "platt"
    assert mapping.split == "calibration"
    low = mapping.apply(np.array([-5.0, -2.0]))
    high = mapping.apply(np.array([2.0, 5.0]))
    assert np.all(low < 0.1)
    assert np.all(high > 0.9)
    assert np.all(mapping.apply(np.array([0.0])) >= 0.0)


def test_calibration_improves_brier_on_separated_data() -> None:
    scores, labels = _perfect_scores()
    mapping = fit_calibration(scores, labels, split="calibration")
    calibrated = mapping.apply(scores)
    raw_prob = 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
    assert brier_score(calibrated, labels) < brier_score(raw_prob, labels)


def test_temperature_method() -> None:
    scores, labels = _perfect_scores()
    mapping = fit_calibration(scores, labels, split="calibration", method="temperature")
    assert mapping.method == "temperature"
    probabilities = mapping.apply(scores)
    assert np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0)


def test_rejects_non_calibration_split() -> None:
    scores, labels = _perfect_scores()
    with pytest.raises(CalibrationError, match="calibration"):
        fit_calibration(scores, labels, split="validation")
    with pytest.raises(CalibrationError, match="calibration"):
        fit_calibration(scores, labels, split="train")


def test_rejects_misaligned_or_nonfinite() -> None:
    scores, labels = _perfect_scores()
    with pytest.raises(CalibrationError, match="aligned"):
        fit_calibration(scores[:10], labels, split="calibration")
    bad = scores.copy()
    bad[0] = np.nan
    with pytest.raises(CalibrationError, match="finite"):
        fit_calibration(bad, labels, split="calibration")


def test_deterministic_fit() -> None:
    scores, labels = _perfect_scores()
    first = fit_calibration(scores, labels, split="calibration")
    second = fit_calibration(scores, labels, split="calibration")
    assert first.params == second.params
    probe = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(first.apply(probe), second.apply(probe))


def test_brier_and_reliability() -> None:
    scores, labels = _perfect_scores()
    mapping = fit_calibration(scores, labels, split="calibration")
    probabilities = mapping.apply(scores)
    centers, mean_pred, mean_label, counts = reliability_curve(probabilities, labels, n_bins=10)
    assert centers.shape == mean_pred.shape == mean_label.shape == counts.shape
    assert counts.sum() == labels.size
    ece = expected_calibration_error(probabilities, labels)
    assert 0.0 <= ece <= 0.5
    assert 0.0 <= brier_score(probabilities, labels) <= 0.5


def test_grouped_brier_uses_groups() -> None:
    rng = np.random.default_rng(1)
    per_group: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for group_id in ("a", "b", "c"):
        scores = rng.normal(0.0, 1.0, size=200)
        labels = (scores > 0.0).astype(np.float64)
        per_group[group_id] = (scores, labels)
    aggregate = grouped_brier(per_group, n_boot=50, seed=0)
    assert aggregate["n_groups"] == 3
    assert aggregate["n_boot"] == 50
    assert aggregate["mean"] >= 0.0
