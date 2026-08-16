"""Observation-ambiguity suite tests (Phase 10, structural-risk-v1 section 1).

Distinct clean candidates must re-degrade to near-identical observations
while remaining clearly different as candidates — the ambiguity the forward
family cannot resolve from a single observation.
"""

from __future__ import annotations

import pytest

from evidence_net.evaluation.metrics import mae
from evidence_net.stress_tests.ambiguity import ambiguity_cases


def test_ambiguity_cases_are_built() -> None:
    cases = ambiguity_cases(size=64, sigma=1.5)
    assert [case.case_id for case in cases] == ["stripe", "line-present-absent"]


def test_stripe_case_observations_nearly_identical() -> None:
    (case,) = [case for case in ambiguity_cases(size=64, sigma=1.5) if case.case_id == "stripe"]
    # Candidates differ everywhere ...
    assert case.candidate_mae == pytest.approx(1.0)
    # ... but their observations are near-identical: the family cannot
    # distinguish them from a single observation.
    assert case.observation_mae < 0.01


def test_line_case_observation_difference_below_resolution() -> None:
    (case,) = [
        case
        for case in ambiguity_cases(size=64, sigma=1.5)
        if case.case_id == "line-present-absent"
    ]
    assert case.candidate_mae > 0.0
    assert case.observation_mae < 0.06
    assert case.as_dict()["observation_mae"] == case.observation_mae


def test_ambiguity_is_a_separate_threat_model() -> None:
    # The ambiguity suite never touches the output grid directly: candidates
    # are clean images and observations are on the input grid.
    case = ambiguity_cases(size=64, sigma=1.5)[0]
    assert case.candidate_a.shape == case.candidate_b.shape
    assert case.observation_a.shape != case.candidate_a.shape  # degraded down
    assert mae(case.observation_a, case.observation_b) < mae(case.candidate_a, case.candidate_b)
