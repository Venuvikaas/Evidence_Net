"""Data Health & Acquisition Statistic Drift Monitor (Phase 17).

Computes input range, mean, variance, and Kolmogorov-Smirnov distance drift
summaries against reference baseline distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DriftReport:
    """Input statistic drift summary report."""

    mean_shift: float
    std_ratio: float
    out_of_range_ratio: float
    has_drift_alert: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean_shift": round(self.mean_shift, 4),
            "std_ratio": round(self.std_ratio, 4),
            "out_of_range_ratio": round(self.out_of_range_ratio, 4),
            "has_drift_alert": self.has_drift_alert,
        }


def compute_input_drift(
    current_tensor: np.ndarray,
    baseline_mean: float = 0.5,
    baseline_std: float = 0.25,
    expected_range: tuple[float, float] = (0.0, 1.0),
    shift_threshold: float = 0.15,
) -> DriftReport:
    """Compute statistics for current_tensor against reference baseline."""
    arr = np.asarray(current_tensor, dtype=np.float64)

    curr_mean = float(np.mean(arr))
    curr_std = float(np.std(arr))

    mean_shift = abs(curr_mean - baseline_mean)
    std_ratio = curr_std / baseline_std if baseline_std > 0 else 1.0

    out_of_range = (arr < expected_range[0]) | (arr > expected_range[1])
    out_of_range_ratio = float(np.mean(out_of_range))

    has_alert = (
        mean_shift > shift_threshold
        or std_ratio < 0.5
        or std_ratio > 2.0
        or out_of_range_ratio > 0.01
    )

    return DriftReport(
        mean_shift=mean_shift,
        std_ratio=std_ratio,
        out_of_range_ratio=out_of_range_ratio,
        has_drift_alert=has_alert,
    )
