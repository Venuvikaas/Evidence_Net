"""Grouped statistics tests: bootstrap CIs resample groups (never pixels)."""

from __future__ import annotations

import numpy as np
import pytest

from evidence_net.evaluation.statistics import (
    GroupedAggregate,
    GroupingError,
    grouped_bootstrap_ci,
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
