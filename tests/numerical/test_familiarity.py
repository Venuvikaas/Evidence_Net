"""Distribution familiarity tests (Phase 9, familiarity-v1).

Analytical and controlled-population fixtures: the frozen feature vector,
the reference-distance baseline, shift-suite construction, threshold
behavior, report structure and determinism, and the rare-valid gate
mechanism (in-domain rare structures stay familiar; out-of-domain ones are
flagged and reported against the cap).
"""

from __future__ import annotations

import numpy as np
import pytest

from evidence_net.stress_tests.familiarity import (
    RARE_VALID_GROUP,
    REFERENCE_GROUP,
    FamiliarityConfig,
    FamiliarityError,
    ReferenceFamiliarity,
    build_familiarity_report,
    build_shift_suite,
    feature_vector,
)

SIZE = 64


def _noise_images(n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.random((SIZE, SIZE)) for _ in range(n)]


# --- Feature representation -------------------------------------------------


def test_feature_vector_shape_and_names() -> None:
    image = np.random.default_rng(0).random((32, 32))
    vector = feature_vector(image)
    assert vector.shape == (6,)
    assert np.isfinite(vector).all()


def test_feature_vector_is_deterministic() -> None:
    image = np.random.default_rng(1).random((32, 32))
    np.testing.assert_allclose(feature_vector(image), feature_vector(image))


def test_feature_vector_constant_image_is_guarded() -> None:
    vector = feature_vector(np.full((32, 32), 0.5))
    assert vector[1] == 0.0  # std
    np.testing.assert_allclose(vector[2:5], [1.0 / 3.0] * 3)  # neutral energies
    assert np.isfinite(vector).all()


def test_feature_vector_band_energy_fractions_sum_to_one() -> None:
    image = np.random.default_rng(2).random((32, 32))
    vector = feature_vector(image)
    assert np.sum(vector[2:5]) == pytest.approx(1.0, abs=1e-9)
    assert vector[5] >= 0.0  # edge density


# --- Reference-distance baseline ---------------------------------------------


def test_in_distribution_is_familiar() -> None:
    reference = ReferenceFamiliarity.fit(_noise_images(32), threshold=2.0)
    probe = _noise_images(1, seed=5)[0]
    distance = reference.distance(probe)
    assert distance < 2.0
    assert reference.is_familiar(distance)


def test_fit_rejects_empty_or_malformed_reference() -> None:
    with pytest.raises(FamiliarityError):
        ReferenceFamiliarity.fit([])
    with pytest.raises(FamiliarityError):
        ReferenceFamiliarity(np.zeros((4, 5)), threshold=2.0)


def test_threshold_boundary_behavior() -> None:
    reference = ReferenceFamiliarity.fit(_noise_images(16), threshold=2.0)
    probe = _noise_images(1, seed=3)[0]
    distance = reference.distance(probe)
    assert reference.is_familiar(distance + 1e6) is False  # tiny threshold
    tight = ReferenceFamiliarity.fit(_noise_images(16), threshold=1e-9)
    assert tight.is_familiar(distance) is False
    loose = ReferenceFamiliarity.fit(_noise_images(16), threshold=1e9)
    assert loose.is_familiar(distance) is True


# --- Shift suite --------------------------------------------------------------


def test_shift_suite_contains_all_declared_groups() -> None:
    suite = build_shift_suite(n_per_shift=8, size=SIZE, seed=0)
    assert set(suite) == {
        "reference",
        "source",
        "severity",
        "degradation",
        "acquisition",
        "rare-valid",
    }
    for images in suite.values():
        assert len(images) == 8


def test_severity_shift_moves_farther_from_reference() -> None:
    suite = build_shift_suite(n_per_shift=16, size=SIZE, seed=0)
    reference = ReferenceFamiliarity.fit(suite[REFERENCE_GROUP], threshold=2.0)
    reference_mean = float(np.mean([reference.distance(image) for image in suite[REFERENCE_GROUP]]))
    severity_mean = float(np.mean([reference.distance(image) for image in suite["severity"]]))
    assert severity_mean > reference_mean


def test_shift_suite_is_seeded_and_deterministic() -> None:
    first = build_shift_suite(n_per_shift=8, size=SIZE, seed=7)
    second = build_shift_suite(n_per_shift=8, size=SIZE, seed=7)
    for name in first:
        np.testing.assert_allclose(np.stack(first[name]), np.stack(second[name]))


# --- Report -------------------------------------------------------------------


def _probes_from_suite(
    suite: dict[str, list[np.ndarray]],
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[str]]]:
    probes = {name: images for name, images in suite.items() if name != REFERENCE_GROUP}
    ids = {name: [f"{name}-{i:04d}" for i in range(len(images))] for name, images in probes.items()}
    return probes, ids


def test_report_structure_and_determinism() -> None:
    suite = build_shift_suite(n_per_shift=8, size=SIZE, seed=0)
    reference = ReferenceFamiliarity.fit(suite[REFERENCE_GROUP], threshold=2.0)
    probes, ids = _probes_from_suite(suite)
    first = build_familiarity_report(reference, probes, ids, rare_valid_max_false_warning_rate=0.5)
    second = build_familiarity_report(reference, probes, ids, rare_valid_max_false_warning_rate=0.5)
    assert first.as_dict() == second.as_dict()
    for name, group in first.shift_groups.items():
        assert 0.0 <= group["detection_rate"] <= 1.0
        assert group["n"] == len(probes[name])
    assert first.rare_valid["n"] == len(probes[RARE_VALID_GROUP])
    assert "exceeds_cap" in first.rare_valid
    assert "applicability" in first.as_dict()


def test_report_rejects_id_count_mismatch() -> None:
    suite = build_shift_suite(n_per_shift=4, size=SIZE, seed=0)
    reference = ReferenceFamiliarity.fit(suite[REFERENCE_GROUP], threshold=2.0)
    probes, ids = _probes_from_suite(suite)
    ids["severity"] = ids["severity"][:-1]
    with pytest.raises(FamiliarityError):
        build_familiarity_report(reference, probes, ids)


def test_rare_valid_in_domain_stays_familiar() -> None:
    # A reference population that contains thin-line structures: rare-valid
    # probes are in-domain and must not be systematically suppressed.
    suite = build_shift_suite(n_per_shift=16, size=SIZE, seed=0)
    reference_images = suite[REFERENCE_GROUP] + suite[RARE_VALID_GROUP]
    reference = ReferenceFamiliarity.fit(reference_images, threshold=2.0)
    more_rare = build_shift_suite(n_per_shift=16, size=SIZE, seed=1)[RARE_VALID_GROUP]
    probes = {RARE_VALID_GROUP: more_rare}
    ids = {RARE_VALID_GROUP: [f"r-{i:04d}" for i in range(len(more_rare))]}
    report = build_familiarity_report(reference, probes, ids, rare_valid_max_false_warning_rate=0.5)
    assert report.rare_valid["false_warning_rate"] < 0.5
    assert report.rare_valid["exceeds_cap"] is False


def test_rare_valid_out_of_domain_is_reported_against_cap() -> None:
    # A pure-noise reference population: thin-line structures are far from it,
    # so the false-warning rate is high and the cap flag fires — the gate
    # input (Gate 8) is reported rather than hidden.
    suite = build_shift_suite(n_per_shift=16, size=SIZE, seed=0)
    reference = ReferenceFamiliarity.fit(suite[REFERENCE_GROUP], threshold=2.0)
    rare = suite[RARE_VALID_GROUP]
    probes = {RARE_VALID_GROUP: rare}
    ids = {RARE_VALID_GROUP: [f"r-{i:04d}" for i in range(len(rare))]}
    report = build_familiarity_report(reference, probes, ids, rare_valid_max_false_warning_rate=0.5)
    assert report.rare_valid["false_warning_rate"] >= 0.5
    assert report.rare_valid["exceeds_cap"] is True


def test_config_validation() -> None:
    with pytest.raises(FamiliarityError):
        FamiliarityConfig(threshold=0.0).validate()
    with pytest.raises(FamiliarityError):
        FamiliarityConfig(rare_valid_max_false_warning_rate=1.5).validate()
    with pytest.raises(FamiliarityError):
        FamiliarityConfig(n_reference=0).validate()
    with pytest.raises(FamiliarityError):
        FamiliarityConfig(version="old").validate()
