"""Frozen downstream measurement task (Phase 10, structural-risk-v1 section 5).

The downstream task is *measurement fidelity*: how well a restored output
supports three frozen structural measurements compared with the target
(edge displacement, connected components of binary edges, connected
components of bright structures). It is a pure function of outputs and
targets — never co-trained and never reading the hidden stress definitions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from evidence_net.evaluation.metrics import binary_edges, edge_displacement
from evidence_net.evaluation.proposal_metrics import connected_components
from evidence_net.evaluation.statistics import GroupedAggregate, GroupingError, grouped_bootstrap_ci

MEASUREMENTS = ("edge_displacement_px", "edge_components", "bright_components")
BRIGHT_THRESHOLD = 0.8


class DownstreamError(ValueError):
    """Raised for invalid downstream evaluations."""


def downstream_measurements(output: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Frozen measurements of a restored output relative to the target."""
    output_2d = np.asarray(output, dtype=np.float64)
    target_2d = np.asarray(target, dtype=np.float64)
    if output_2d.ndim == 3 and output_2d.shape[0] == 1:
        output_2d = output_2d[0]
    if target_2d.ndim == 3 and target_2d.shape[0] == 1:
        target_2d = target_2d[0]
    if output_2d.ndim != 2 or target_2d.ndim != 2:
        raise DownstreamError("output and target must be 2D planes")
    edges = binary_edges(output_2d, 0.5)
    bright = output_2d > BRIGHT_THRESHOLD
    return {
        "edge_displacement_px": float(edge_displacement(target_2d, output_2d)),
        "edge_components": float(connected_components(edges)),
        "bright_components": float(connected_components(bright)),
    }


def measurement_error(name: str, output: np.ndarray, target: np.ndarray) -> float:
    """Absolute deviation of one measurement from the target-derived value."""
    if name not in MEASUREMENTS:
        raise DownstreamError(f"unknown downstream measurement: {name}")
    measurements = downstream_measurements(output, target)
    target_measurements = downstream_measurements(target, target)
    return float(abs(measurements[name] - target_measurements[name]))


@dataclass(frozen=True)
class DownstreamAggregate:
    """Group-bootstrapped error of one measurement for one output type."""

    output_type: str
    measurement: str
    per_group: dict[str, float]
    aggregate: GroupedAggregate

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_type": self.output_type,
            "measurement": self.measurement,
            "per_group": self.per_group,
            "aggregate": self.aggregate.as_dict(),
        }


def evaluate_downstream(
    outputs: Mapping[str, Sequence[np.ndarray]],
    targets: Sequence[np.ndarray],
    group_ids: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per-output-type, per-measurement downstream error with group CIs.

    Returns ``{output_type: {measurement: aggregate_dict}}``. Images /
    source groups are the statistical units; duplicate group ids are
    rejected; pixels are never sample counts.
    """
    if len(targets) != len(group_ids) or not targets:
        raise GroupingError("downstream evaluation requires targets and group_ids of equal length")
    if len(set(group_ids)) != len(group_ids):
        duplicates = sorted({group for group in group_ids if group_ids.count(group) > 1})
        raise GroupingError(f"duplicate group ids in downstream evaluation: {duplicates}")
    report: dict[str, dict[str, dict[str, Any]]] = {}
    for output_type, images in outputs.items():
        if len(images) != len(targets):
            raise GroupingError(
                f"output '{output_type}': {len(images)} images for {len(targets)} targets"
            )
        report[output_type] = {}
        for measurement in MEASUREMENTS:
            per_group: dict[str, float] = {}
            for group_id, output, target in zip(group_ids, images, targets, strict=True):
                per_group[group_id] = measurement_error(measurement, output, target)
            aggregate = grouped_bootstrap_ci(per_group, n_boot=n_boot, seed=seed)
            report[output_type][measurement] = DownstreamAggregate(
                output_type=output_type,
                measurement=measurement,
                per_group=per_group,
                aggregate=aggregate,
            ).as_dict()
    return report
