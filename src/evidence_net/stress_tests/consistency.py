"""Measurement-consistency compatibility report (Phase 7, forward-model-v1).

For each operator in the bounded forward family, re-degrades a restored
output to the input grid and compares it with the observed degraded input.
Reports the residual distribution **across the operator family** (min /
median / max and the arg-min operator), never the minimum alone, aggregated
per image / source group with the same grouped-bootstrap discipline as every
other metric (metrics-v1). The diagnostic is compatibility, not truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from evidence_net.evaluation.metrics import mae
from evidence_net.evaluation.statistics import GroupedAggregate, GroupingError, grouped_bootstrap_ci
from evidence_net.stress_tests.forward import ForwardOperator

INTERPRETATION = (
    "Measurement consistency reports how well a restored output re-degrades "
    "through the declared bounded forward family compared with the observed "
    "input. It is compatibility, not truth: it does not identify the true "
    "degradation, and it never certifies that restored detail physically "
    "existed."
)


@dataclass(frozen=True)
class OperatorConsistency:
    """Per-operator residual behavior over the evaluated groups."""

    operator: str
    kind: str
    per_group_mae: dict[str, float]
    aggregate: GroupedAggregate
    bias_mean: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "kind": self.kind,
            "per_group_mae": self.per_group_mae,
            "aggregate": self.aggregate.as_dict(),
            "bias_mean": self.bias_mean,
        }


@dataclass(frozen=True)
class ConsistencyReport:
    """Residual distribution across the operator family for one sample set."""

    n_groups: int
    operators: tuple[OperatorConsistency, ...]
    across_operators: dict[str, float | str]
    interpretation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_groups": self.n_groups,
            "operators": [operator.as_dict() for operator in self.operators],
            "across_operators": self.across_operators,
            "interpretation": self.interpretation,
        }


def re_degrade(
    restored: np.ndarray, operator: ForwardOperator, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Push a restored output back through an operator onto the input grid."""
    return operator.apply(restored, rng)


def _per_group_residuals(
    restored_images: Sequence[np.ndarray],
    observations: Sequence[np.ndarray],
    group_ids: Sequence[str],
    operator: ForwardOperator,
    rng: np.random.Generator,
) -> tuple[dict[str, float], float]:
    """Per-group MAE and mean signed error (bias) for one operator."""
    if not (len(restored_images) == len(observations) == len(group_ids)):
        raise GroupingError(
            "restored_images, observations, and group_ids must have the same length "
            f"({len(restored_images)}, {len(observations)}, {len(group_ids)})"
        )
    per_group: dict[str, float] = {}
    bias_values: list[float] = []
    for restored, observation, group_id in zip(
        restored_images, observations, group_ids, strict=True
    ):
        if group_id in per_group:
            raise GroupingError(f"duplicate group id in consistency report: {group_id}")
        degraded = re_degrade(restored, operator, rng)
        per_group[group_id] = float(mae(observation, degraded))
        residual = degraded - observation
        bias_values.append(float(residual.mean()))
    return per_group, float(np.mean(bias_values))


def build_consistency_report(
    restored_images: Sequence[np.ndarray],
    observations: Sequence[np.ndarray],
    group_ids: Sequence[str],
    operators: Sequence[ForwardOperator],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> ConsistencyReport:
    """Aggregate per-operator residuals with grouped bootstrap CIs.

    Stochastic operators draw one seeded sample per group
    (``seed + group_index``); pass ``seed`` to make the report reproducible.
    Reports the distribution across operators, not the minimum alone.
    """
    if not restored_images or len(restored_images) != len(group_ids):
        raise GroupingError("consistency report requires at least one image per group id")
    rng = np.random.default_rng(seed)
    entries: list[OperatorConsistency] = []
    for index, operator in enumerate(operators):
        operator_rng = np.random.default_rng(seed + index) if operator.is_stochastic else rng
        per_group, bias = _per_group_residuals(
            restored_images, observations, group_ids, operator, operator_rng
        )
        aggregate = grouped_bootstrap_ci(per_group, n_boot=n_boot, seed=seed)
        entries.append(
            OperatorConsistency(
                operator=operator.name,
                kind=operator.kind,
                per_group_mae=per_group,
                aggregate=aggregate,
                bias_mean=bias,
            )
        )
    entries.sort(key=lambda entry: entry.operator)
    means = {entry.operator: entry.aggregate.mean for entry in entries}
    argmin = min(means, key=lambda name: means[name])
    across: dict[str, float | str] = {
        "min_mae": float(min(means.values())),
        "argmin_operator": argmin,
        "median_mae": float(float(np.median(list(means.values())))),
        "max_mae": float(max(means.values())),
        "n_operators": len(entries),
    }
    return ConsistencyReport(
        n_groups=len(group_ids),
        operators=tuple(entries),
        across_operators=across,
        interpretation=INTERPRETATION,
    )


def per_image_residuals(
    restored_images: Sequence[np.ndarray],
    observations: Sequence[np.ndarray],
    group_ids: Sequence[str],
    operators: Sequence[ForwardOperator],
    *,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Per-image per-operator MAE residuals — the candidate consistency features.

    Exposed so Lane A can test whether consistency residuals add incremental
    value beyond simple benefit features (EXP-005, Gate 6). Each group maps to
    ``{operator_name: mae}`` on the input grid.
    """
    if not (len(restored_images) == len(observations) == len(group_ids)):
        raise GroupingError(
            "restored_images, observations, and group_ids must have the same length"
        )
    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, float]] = {}
    for restored, observation, group_id in zip(
        restored_images, observations, group_ids, strict=True
    ):
        result[group_id] = {}
        for operator_index, operator in enumerate(operators):
            operator_rng = (
                np.random.default_rng(seed + operator_index) if operator.is_stochastic else rng
            )
            degraded = re_degrade(restored, operator, operator_rng)
            result[group_id][operator.name] = float(mae(observation, degraded))
    return result


def measure_noise_variance(
    operator: ForwardOperator,
    clean: np.ndarray,
    *,
    n_draws: int = 64,
    seed: int = 0,
) -> dict[str, float]:
    """Quantify stochastic spread: per-draw residual MAE against a reference.

    Applies the operator ``n_draws`` times to ``clean`` with seeded draws and
    reports the spread of the re-degraded outputs (std / min / max of the
    per-draw mean absolute deviation from the seeded first draw). Deterministic
    operators report zero spread.
    """
    if not operator.is_stochastic:
        return {"std": 0.0, "min": 0.0, "max": 0.0, "n_draws": n_draws}
    rng = np.random.default_rng(seed)
    reference = re_degrade(clean, operator, rng)
    spreads: list[float] = []
    for _ in range(n_draws):
        draw = re_degrade(clean, operator, rng)
        spreads.append(float(np.abs(draw - reference).mean()))
    return {
        "std": float(np.std(spreads)),
        "min": float(np.min(spreads)),
        "max": float(np.max(spreads)),
        "n_draws": n_draws,
    }
