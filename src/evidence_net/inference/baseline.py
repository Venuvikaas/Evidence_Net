"""Baseline inference pipeline.

Provides the common inference, evaluation, and artifact interfaces for the
classical reference path (Phase 2). Any restorer with the ``Restorer``
signature can be driven through the same pipeline: apply to each input,
clip to the ``[0, 1]`` tensor contract, compute the contract metrics per
image, and aggregate with grouped bootstrap confidence intervals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evidence_net.evaluation.metrics import all_metrics
from evidence_net.evaluation.statistics import grouped_bootstrap_ci
from evidence_net.models.reference import Restorer


@dataclass(frozen=True)
class RestorerResult:
    """Evaluation results for one restorer over a fixed sample set."""

    name: str
    per_group_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    aggregates: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "per_group_metrics": self.per_group_metrics,
            "aggregates": self.aggregates,
        }


def run_restorer(inputs: Sequence[np.ndarray], restorer: Restorer) -> list[np.ndarray]:
    """Apply a restorer to every input and clip outputs to ``[0, 1]``.

    Outputs are clipped because the tensor contract fixes the comparison
    domain at ``[0, 1]``; raw unclipped values are never reported as if they
    were predictions in range.
    """
    predictions: list[np.ndarray] = []
    for array in inputs:
        restored = np.asarray(restorer(array), dtype=np.float64)
        predictions.append(np.clip(restored, 0.0, 1.0))
    return predictions


def _metric_aggregates(
    per_group_metrics: dict[str, dict[str, Any]], *, n_boot: int, seed: int
) -> dict[str, dict[str, float | int]]:
    """Grouped bootstrap aggregates for every scalar metric and frequency band."""
    first = next(iter(per_group_metrics.values()))
    scalar_metrics = [key for key in first if key != "frequency_bands"]
    bands = sorted(first["frequency_bands"])
    aggregates: dict[str, dict[str, float | int]] = {}
    for metric in scalar_metrics + [f"frequency_bands.{band}" for band in bands]:
        values: dict[str, float] = {}
        for group_id, metrics in per_group_metrics.items():
            if metric.startswith("frequency_bands."):
                band = metric.removeprefix("frequency_bands.")
                values[group_id] = float(metrics["frequency_bands"][band])
            else:
                values[group_id] = float(metrics[metric])
        aggregates[metric] = grouped_bootstrap_ci(values, n_boot=n_boot, seed=seed).as_dict()
    return aggregates


def evaluate_restorer(
    name: str,
    inputs: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    group_ids: Sequence[str],
    restorer: Restorer,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> RestorerResult:
    """Run one restorer and return per-group metrics plus grouped CIs."""
    predictions = run_restorer(inputs, restorer)
    per_group_metrics: dict[str, dict[str, Any]] = {}
    for group_id, prediction, target in zip(group_ids, predictions, targets, strict=True):
        per_group_metrics[group_id] = all_metrics(target, prediction)
    aggregates = _metric_aggregates(per_group_metrics, n_boot=n_boot, seed=seed)
    return RestorerResult(name=name, per_group_metrics=per_group_metrics, aggregates=aggregates)


def evaluate_restorers(
    inputs: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    group_ids: Sequence[str],
    restorers: Mapping[str, Restorer],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, RestorerResult]:
    """Evaluate every named restorer over the same sample set.

    All restorers see the identical inputs/targets so comparisons are paired.
    """
    results: dict[str, RestorerResult] = {}
    for name, restorer in restorers.items():
        results[name] = evaluate_restorer(
            name,
            inputs,
            targets,
            group_ids,
            restorer,
            n_boot=n_boot,
            seed=seed,
        )
    return results
