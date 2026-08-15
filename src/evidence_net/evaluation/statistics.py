"""Grouped statistics for evaluation.

Images / source groups are the statistical units. Pixels are never reported
as independent sample counts. This module provides seeded group bootstrap
confidence intervals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


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
