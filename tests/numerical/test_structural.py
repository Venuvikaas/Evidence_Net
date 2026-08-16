"""Candidate manipulation suite tests (Phase 10, structural-risk-v1 section 2).

Controlled structural fixtures: inserting a line increases edge components,
deleting one decreases them, shifting an edge moves it, merge/split change
component counts, and defect points toggle isolated components. Parameters
come from the frozen hidden stress definitions.
"""

from __future__ import annotations

import numpy as np
import pytest

from evidence_net.evaluation.metrics import binary_edges, edge_displacement
from evidence_net.evaluation.proposal_metrics import connected_components
from evidence_net.stress_tests.hidden_stress import stress_params
from evidence_net.stress_tests.structural import (
    DefectPoint,
    EdgeShift,
    FalseLineInsertion,
    FalsePeriodicity,
    MergeLines,
    RealLineDeletion,
    SplitLine,
    StructuralError,
    build_candidate_suite,
)

SIZE = 64


def _structured_fixture() -> np.ndarray:
    """Dark background with three bright vertical lines."""
    image = np.full((SIZE, SIZE), 0.02)
    for column in (16, 32, 48):
        image[:, column] = 0.9
    return image


def _components(image: np.ndarray) -> int:
    return connected_components(binary_edges(image, 0.5))


def _bright_components(image: np.ndarray) -> int:
    # Bright structures (lines/points) are better separated by intensity than
    # by edges, whose horizontal ends can reconnect after a split.
    return connected_components(image > 0.5)


def test_false_line_insertion_adds_a_component() -> None:
    original = _structured_fixture()
    manipulation = FalseLineInsertion(stress_params()["perturbation"])
    rng = np.random.default_rng(0)
    modified = manipulation.apply(original, rng)
    assert _components(modified) > _components(original)
    assert modified.min() >= 0.0 and modified.max() <= 1.0


def test_real_line_deletion_removes_a_component() -> None:
    original = _structured_fixture()
    manipulation = RealLineDeletion(stress_params()["perturbation"])
    modified = manipulation.apply(original, np.random.default_rng(0))
    assert _components(modified) < _components(original)


def test_edge_shift_moves_the_edge() -> None:
    original = _structured_fixture()
    manipulation = EdgeShift(stress_params()["perturbation"])
    modified = manipulation.apply(original, np.random.default_rng(0))
    assert edge_displacement(original, modified) > 0.0
    assert manipulation.effect().startswith("shifts")


def test_merge_reduces_component_count() -> None:
    image = np.full((SIZE, SIZE), 0.02)
    image[:, 16] = 0.9
    image[:, 20] = 0.9  # two nearby lines
    before = _bright_components(image)
    manipulation = MergeLines(stress_params()["perturbation"])
    modified = manipulation.apply(image, np.random.default_rng(0))
    assert _bright_components(modified) < before


def test_split_increases_component_count() -> None:
    original = _structured_fixture()
    before = _bright_components(original)
    manipulation = SplitLine(stress_params()["perturbation"])
    modified = manipulation.apply(original, np.random.default_rng(0))
    assert _bright_components(modified) > before


def test_false_periodicity_adds_energy_and_stays_bounded() -> None:
    original = _structured_fixture()
    manipulation = FalsePeriodicity(stress_params()["perturbation"])
    modified = manipulation.apply(original, np.random.default_rng(0))
    assert float(np.abs(modified - original).mean()) > 1e-3
    assert modified.min() >= 0.0 and modified.max() <= 1.0


def test_defect_point_toggles_an_isolated_component() -> None:
    original = np.full((SIZE, SIZE), 0.02)
    manipulation = DefectPoint(stress_params()["perturbation"])
    modified = manipulation.apply(original, np.random.default_rng(0))
    bright_before = connected_components(original > 0.8)
    bright_after = connected_components(modified > 0.8)
    assert bright_after == bright_before + 1


def test_manipulations_are_seeded_deterministic() -> None:
    original = _structured_fixture()
    params = stress_params()["perturbation"]
    for name in ("false-line", "edge-shift", "defect-point"):
        manipulation = build_candidate_suite(params, names=(name,))[0]
        first = manipulation.apply(original, np.random.default_rng(11))
        second = manipulation.apply(original, np.random.default_rng(11))
        np.testing.assert_allclose(first, second)


def test_suite_builds_all_frozen_manipulations() -> None:
    suite = build_candidate_suite()
    names = [manipulation.name for manipulation in suite]
    assert names == [
        "false-line",
        "line-deletion",
        "edge-shift",
        "merge",
        "split",
        "false-periodicity",
        "defect-point",
    ]
    assert all(manipulation.threat == "candidate" for manipulation in suite)


def test_unknown_manipulation_rejected() -> None:
    with pytest.raises(StructuralError):
        build_candidate_suite(names=("bogus",))
