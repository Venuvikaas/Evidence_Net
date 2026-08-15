"""Grouped statistics for evaluation.

Images / source groups are the statistical units. Pixels are never reported
as independent sample counts. This module provides seeded group bootstrap
confidence intervals and guards that reject pixel-level aggregation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from evidence_net.evaluation.metrics import MetricFn


class GroupingError(ValueError):
    """Raised when metrics are aggregated without valid group structure."""


@dataclass(frozen=True)
class GroupedAggregate:
    """Bootstrap aggregate over groups (never over pixels)."""

    mean: float
    ci_lo: float
    ci_hi: float
    n_groups: int
    n_boot: int
    ci_level: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "n_groups": self.n_groups,
            "n_boot": self.n_boot,
            "ci_level": self.ci_level,
        }


def grouped_bootstrap_ci(
    group_values: Mapping[str, float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> GroupedAggregate:
    """Percentile bootstrap CI resampling **groups** with replacement."""
    if not group_values:
        raise GroupingError("grouped_bootstrap_ci requires at least one group")
    values = np.asarray(list(group_values.values()), dtype=float)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    n = len(values)
    for i in range(n_boot):
        boot_means[i] = rng.choice(values, size=n, replace=True).mean()
    alpha = 1.0 - ci_level
    lo, hi = np.percentile(boot_means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return GroupedAggregate(
        mean=float(values.mean()),
        ci_lo=float(lo),
        ci_hi=float(hi),
        n_groups=n,
        n_boot=n_boot,
        ci_level=ci_level,
    )


def reject_pixel_level(values: Sequence[float], images: Sequence[np.ndarray]) -> None:
    """Raise if ``values`` looks like one value per pixel rather than per image."""
    total_pixels = sum(int(image.size) for image in images)
    if len(values) > 1 and len(values) == total_pixels:
        raise GroupingError(
            f"{len(values)} values for {total_pixels} pixels: metrics must be "
            "computed per image / source group, never per pixel"
        )


def aggregate_by_group(
    metric_fn: MetricFn,
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    group_ids: Sequence[str],
    *,
    ci_level: float = 0.95,
    seed: int = 0,
    n_boot: int = 1000,
) -> tuple[dict[str, float], GroupedAggregate]:
    """Compute a metric per group id and aggregate with a group bootstrap CI.

    Requires exactly one prediction/target pair per group id. Pixel-level
    inputs (more values than groups) raise ``GroupingError``.
    """
    if not (len(predictions) == len(targets) == len(group_ids)):
        raise GroupingError(
            "predictions, targets, and group_ids must have the same length "
            f"({len(predictions)}, {len(targets)}, {len(group_ids)})"
        )
    per_group: dict[str, float] = {}
    for group_id, prediction, target in zip(group_ids, predictions, targets, strict=True):
        if group_id in per_group:
            raise GroupingError(f"duplicate group id in aggregation: {group_id}")
        per_group[group_id] = float(metric_fn(target, prediction))
    reject_pixel_level(list(per_group.values()), targets)
    aggregate = grouped_bootstrap_ci(per_group, ci_level=ci_level, seed=seed, n_boot=n_boot)
    return per_group, aggregate
