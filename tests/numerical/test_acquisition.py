"""Acquisition artifact suite tests (Phase 10, structural-risk-v1 section 3).

Artifacts apply to the degraded input, preserve shape, stay in [0, 1], are
seeded-deterministic, and genuinely change the input (except on degenerate
inputs). All are labeled ``acquisition`` — a separate threat model.
"""

from __future__ import annotations

import numpy as np

from evidence_net.stress_tests.acquisition import (
    AcquisitionError,
    build_acquisition_suite,
)
from evidence_net.stress_tests.hidden_stress import stress_params

SIZE = 64


def _input_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((SIZE, SIZE))


def test_suite_builds_all_frozen_artifacts() -> None:
    suite = build_acquisition_suite()
    names = [artifact.name for artifact in suite]
    assert names == [
        "sensor-noise",
        "column-stripe",
        "gain-nonuniformity",
        "dead-pixels",
        "local-blur-patch",
    ]
    assert all(artifact.threat == "acquisition" for artifact in suite)


def test_artifacts_preserve_shape_and_bounds() -> None:
    image = _input_image()
    for artifact in build_acquisition_suite():
        modified = artifact.apply(image, np.random.default_rng(1))
        assert modified.shape == image.shape
        assert modified.min() >= 0.0 and modified.max() <= 1.0


def test_artifacts_change_the_input() -> None:
    image = _input_image()
    for artifact in build_acquisition_suite():
        modified = artifact.apply(image, np.random.default_rng(2))
        assert float(np.abs(modified - image).mean()) > 1e-4, artifact.name


def test_artifacts_are_seeded_deterministic() -> None:
    image = _input_image()
    for artifact in build_acquisition_suite():
        first = artifact.apply(image, np.random.default_rng(7))
        second = artifact.apply(image, np.random.default_rng(7))
        np.testing.assert_allclose(first, second, err_msg=artifact.name)


def test_dead_pixels_zero_a_small_fraction() -> None:
    image = np.full((SIZE, SIZE), 0.5)
    artifact = build_acquisition_suite(names=("dead-pixels",))[0]
    modified = artifact.apply(image, np.random.default_rng(3))
    fraction = float(np.mean(modified == 0.0))
    params = stress_params()["acquisition"]
    expected = float(params["dead_pixel_fraction"])
    assert abs(fraction - expected) < 5 * expected


def test_unknown_artifact_rejected() -> None:
    try:
        build_acquisition_suite(names=("bogus",))
    except AcquisitionError:
        pass
    else:
        raise AssertionError("expected AcquisitionError for unknown artifact")
