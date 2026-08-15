"""Grouped statistics tests: bootstrap CIs resample groups, and pixel-level
aggregation is rejected (docs/evaluation-protocol.md section 4)."""

from __future__ import annotations

import numpy as np
import pytest

from evidence_net.evaluation.metrics import mae
from evidence_net.evaluation.statistics import (
    GroupedAggregate,
    GroupingError,
    aggregate_by_group,
    grouped_bootstrap_ci,
    reject_pixel_level,
)


def test_bootstrap_ci_constant_values_collapse() -> None:
    groups = {f"g{i}": 0.5 for i in range(20)}
    result = grouped_bootstrap_ci(groups, n_boot=100)
    assert result.mean == pytest.approx(0.5)
    assert result.ci_lo == pytest.approx(0.5)
    assert result.ci_hi == pytest.approx(0.5)
    assert result.n_groups == 20
    assert result.n_boot == 100


def test_bootstrap_ci_contains_mean_and_is_seeded() -> None:
    groups = {f"g{i}": float(i) for i in range(30)}
    result = grouped_bootstrap_ci(groups, n_boot=500, seed=7)
    assert isinstance(result, GroupedAggregate)
    assert result.mean == pytest.approx(np.mean(list(groups.values())))
    assert result.ci_lo <= result.mean <= result.ci_hi
    again = grouped_bootstrap_ci(groups, n_boot=500, seed=7)
    assert result.ci_lo == again.ci_lo
    assert result.ci_hi == again.ci_hi


def test_bootstrap_ci_empty_raises() -> None:
    with pytest.raises(GroupingError):
        grouped_bootstrap_ci({})


def test_reject_pixel_level_detects_pixel_counts() -> None:
    images = [np.zeros((8, 8)), np.zeros((8, 8))]
    with pytest.raises(GroupingError):
        reject_pixel_level([0.0] * 128, images)
    # A per-image list of 2 values passes.
    reject_pixel_level([0.1, 0.2], images)


def test_aggregate_by_group_one_value_per_group() -> None:
    predictions = [np.zeros((4, 4)), np.ones((4, 4))]
    targets = [np.ones((4, 4)), np.ones((4, 4))]
    per_group, aggregate = aggregate_by_group(mae, predictions, targets, ["a", "b"], n_boot=50)
    assert per_group == {"a": 1.0, "b": 0.0}
    assert aggregate.mean == pytest.approx(0.5)
    assert aggregate.n_groups == 2


def test_aggregate_by_group_rejects_pixel_level_values() -> None:
    # 8 one-pixel images: 8 values for 8 pixels is pixel-level aggregation.
    predictions = [np.zeros((1, 1)) for _ in range(8)]
    targets = [np.zeros((1, 1)) for _ in range(8)]
    group_ids = [f"p{i}" for i in range(8)]
    with pytest.raises(GroupingError):
        aggregate_by_group(mae, predictions, targets, group_ids, n_boot=10)


def test_aggregate_by_group_rejects_duplicate_group_ids() -> None:
    predictions = [np.zeros((4, 4)), np.zeros((4, 4))]
    targets = [np.zeros((4, 4)), np.zeros((4, 4))]
    with pytest.raises(GroupingError):
        aggregate_by_group(mae, predictions, targets, ["a", "a"], n_boot=10)


def test_aggregate_by_group_rejects_length_mismatch() -> None:
    with pytest.raises(GroupingError):
        aggregate_by_group(mae, [np.zeros((4, 4))], [], ["a"], n_boot=10)
