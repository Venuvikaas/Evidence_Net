"""Observation-ambiguity suite (Phase 10, structural-risk-v1 section 1).

Distinct clean candidates that re-degrade to nearly identical observations:
the forward family cannot distinguish them, so no output-based diagnostic
can resolve the ambiguity from a single observation. Built from the
``forward-model-v1`` non-identifiability cases (stripe and line pairs).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evidence_net.evaluation.metrics import mae
from evidence_net.stress_tests.forward import (
    non_identifiable_line_pair,
    non_identifiable_stripe_pair,
)


@dataclass(frozen=True)
class AmbiguityCase:
    """Two clean candidates and their near-identical observations."""

    case_id: str
    candidate_a: np.ndarray
    candidate_b: np.ndarray
    observation_a: np.ndarray
    observation_b: np.ndarray
    observation_mae: float
    candidate_mae: float

    def as_dict(self) -> dict[str, float]:
        return {
            "observation_mae": self.observation_mae,
            "candidate_mae": self.candidate_mae,
        }


def ambiguity_cases(size: int = 256, sigma: float = 1.5) -> list[AmbiguityCase]:
    """Build the frozen ambiguity cases (stripe and line pairs)."""
    cases: list[AmbiguityCase] = []
    clean_a, clean_b, obs_a, obs_b = non_identifiable_stripe_pair(size=size, sigma=sigma)
    cases.append(
        AmbiguityCase(
            case_id="stripe",
            candidate_a=clean_a,
            candidate_b=clean_b,
            observation_a=obs_a,
            observation_b=obs_b,
            observation_mae=float(mae(obs_a, obs_b)),
            candidate_mae=float(mae(clean_a, clean_b)),
        )
    )
    clean_a, clean_b, obs_a, obs_b = non_identifiable_line_pair(size=size, sigma=sigma)
    cases.append(
        AmbiguityCase(
            case_id="line-present-absent",
            candidate_a=clean_a,
            candidate_b=clean_b,
            observation_a=obs_a,
            observation_b=obs_b,
            observation_mae=float(mae(obs_a, obs_b)),
            candidate_mae=float(mae(clean_a, clean_b)),
        )
    )
    return cases
