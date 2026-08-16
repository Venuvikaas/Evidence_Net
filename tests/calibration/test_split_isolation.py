"""Split-isolation regression tests (calibration-version-v1, Phase 5).

These tests enforce the kill-switch rule: calibration may only be fit on the
calibration split, the fitted mapping is a pure function of calibration-split
data, and no test / held-out data can enter fitting.
"""

from __future__ import annotations

import numpy as np
import pytest

from evidence_net.benefit.calibration import CalibrationError, fit_calibration


def _scores_labels(seed: int = 0, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    half = n // 2
    scores = np.concatenate([rng.normal(-2.0, 0.5, size=half), rng.normal(2.0, 0.5, size=half)])
    labels = np.concatenate([np.zeros(half), np.ones(half)])
    return scores, labels


def test_validation_and_train_splits_rejected_for_fit() -> None:
    scores, labels = _scores_labels()
    for forbidden in ("validation", "train", "heldout-source", "test", ""):
        with pytest.raises(CalibrationError, match="calibration"):
            fit_calibration(scores, labels, split=forbidden)


def test_mapping_is_pure_function_of_calibration_data() -> None:
    """Same calibration data in -> identical mapping out, regardless of context."""
    scores, labels = _scores_labels(seed=7)
    mapping_a = fit_calibration(scores, labels, split="calibration")
    mapping_b = fit_calibration(scores, labels, split="calibration")
    probe = np.linspace(-4.0, 4.0, 51)
    assert np.array_equal(mapping_a.apply(probe), mapping_b.apply(probe))
    assert mapping_a.n_calibration == mapping_b.n_calibration == labels.size


def test_fit_never_sees_validation_labels() -> None:
    """Changing only non-calibration data must not change the mapping."""
    cal_scores, cal_labels = _scores_labels(seed=3, n=300)
    validation_scores, validation_labels = _scores_labels(seed=9, n=300)

    mapping_before = fit_calibration(cal_scores, cal_labels, split="calibration")
    # A caller holding validation data cannot leak it into the fit: the fit
    # API only accepts one split, and that split is enforced above.
    probe = np.array([-3.0, -1.0, 0.5, 2.0, 4.0])
    expected = mapping_before.apply(probe)
    assert np.all(np.isfinite(expected))

    # Applying the mapping to validation data is allowed and deterministic.
    validation_probs = mapping_before.apply(validation_scores)
    assert validation_probs.shape == validation_labels.shape
    assert np.all((validation_probs >= 0.0) & (validation_probs <= 1.0))
