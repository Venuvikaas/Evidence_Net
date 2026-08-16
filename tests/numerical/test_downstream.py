"""Frozen downstream task tests (Phase 10, structural-risk-v1 section 5).

The downstream task is a pure function of outputs and targets: identical
outputs have zero error, structural manipulations increase measurement
error, aggregation follows group discipline, and the task never co-trains.
"""

from __future__ import annotations

import numpy as np
import pytest

from evidence_net.evaluation.statistics import GroupingError
from evidence_net.stress_tests.downstream import (
    MEASUREMENTS,
    DownstreamError,
    downstream_measurements,
    evaluate_downstream,
    measurement_error,
)
from evidence_net.stress_tests.structural import EdgeShift, FalseLineInsertion

SIZE = 64


def _structured_fixture() -> np.ndarray:
    image = np.full((SIZE, SIZE), 0.02)
    for column in (16, 32, 48):
        image[:, column] = 0.9
    return image


def test_identical_output_has_zero_error() -> None:
    image = _structured_fixture()
    for measurement in MEASUREMENTS:
        assert measurement_error(measurement, image, image) == 0.0


def test_measurements_are_pure_and_repeatable() -> None:
    image = _structured_fixture()
    first = downstream_measurements(image, image)
    second = downstream_measurements(image, image)
    assert first == second
    assert set(first) == set(MEASUREMENTS)


def test_edge_shift_increases_displacement_error() -> None:
    image = _structured_fixture()
    shifted = EdgeShift({"edge_shift_px": 2}).apply(image, np.random.default_rng(0))
    assert measurement_error("edge_displacement_px", shifted, image) > 0.0


def test_false_line_increases_component_error() -> None:
    image = _structured_fixture()
    modified = FalseLineInsertion({"line_width": 1}).apply(image, np.random.default_rng(0))
    assert measurement_error("edge_components", modified, image) > 0.0


def test_evaluate_downstream_report_and_discipline() -> None:
    images = [_structured_fixture() for _ in range(4)]
    candidates = [
        FalseLineInsertion({"line_width": 1}).apply(image, np.random.default_rng(i))
        for i, image in enumerate(images)
    ]
    ids = [f"g{i:02d}" for i in range(4)]
    report = evaluate_downstream(
        {"base": images, "candidate": candidates}, images, ids, n_boot=20, seed=0
    )
    assert set(report) == {"base", "candidate"}
    for _output_type, measurements in report.items():
        assert set(measurements) == set(MEASUREMENTS)
        for aggregate in measurements.values():
            assert aggregate["aggregate"]["n_groups"] == 4
            assert aggregate["aggregate"]["n_boot"] == 20
    # Candidate adds structure: its component error exceeds the base's.
    assert (
        report["candidate"]["edge_components"]["aggregate"]["mean"]
        > report["base"]["edge_components"]["aggregate"]["mean"]
    )


def test_evaluate_downstream_is_seeded() -> None:
    images = [_structured_fixture() for _ in range(3)]
    ids = [f"g{i}" for i in range(3)]
    first = evaluate_downstream({"base": images}, images, ids, n_boot=20, seed=5)
    second = evaluate_downstream({"base": images}, images, ids, n_boot=20, seed=5)
    assert first == second


def test_evaluate_downstream_rejects_duplicates_and_mismatches() -> None:
    images = [_structured_fixture() for _ in range(2)]
    with pytest.raises(GroupingError):
        evaluate_downstream({"base": images}, images, ["a", "a"], n_boot=10)
    with pytest.raises(GroupingError):
        evaluate_downstream({"base": images[:1]}, images, ["a", "b"], n_boot=10)
    with pytest.raises(DownstreamError):
        measurement_error("bogus", images[0], images[0])
