"""Benefit evaluation suite tests (EXP-009, Phase 5)."""

from __future__ import annotations

import numpy as np
import pytest

from evidence_net.benefit.evaluate import (
    BenefitEvaluationError,
    _auc,
    _select_top_gate,
    build_benefit_report,
    overall_auc,
    per_group_auc,
    selective_risk_curve,
)
from evidence_net.benefit.labels import OUTPUT_GRID, PATCH_GRID, patch_benefit_labels


def _paired_cases() -> tuple[list, list, list, list, list]:
    """Two samples where the left half is beneficial and the right is not.

    base = 0.5 everywhere; target = 0.6 on the left half, 0.5 on the right.
    proposal = +0.1 everywhere: on the left the candidate reaches the target
    (beneficial), on the right it moves away (harmful).
    """
    bases: list[np.ndarray] = []
    proposals: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    for _ in range(2):
        grid = OUTPUT_GRID
        base = np.full((grid, grid), 0.5)
        proposal = np.full((grid, grid), 0.1)
        target = np.full((grid, grid), 0.5)
        target[:, : grid // 2] = 0.6
        target = np.clip(target + rng.normal(0.0, 0.005, size=(grid, grid)), 0.0, 1.0)
        bases.append(base)
        proposals.append(proposal)
        targets.append(target)
    labels = [
        patch_benefit_labels(b, d, x) for b, d, x in zip(bases, proposals, targets, strict=False)
    ]
    # Scores that perfectly order the left (beneficial) half above the right.
    scores = []
    for _ in range(2):
        score = np.zeros((PATCH_GRID, PATCH_GRID))
        score[:, : PATCH_GRID // 2] = 1.0
        score[:, PATCH_GRID // 2 :] = -1.0
        scores.append(score)
    return scores, labels, bases, proposals, targets


def test_auc_perfect_and_anti() -> None:
    labels = np.array([0, 0, 1, 1])
    assert _auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == pytest.approx(1.0)
    assert _auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == pytest.approx(0.0)
    assert _auc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == pytest.approx(0.5)


def test_per_group_and_overall_auc() -> None:
    scores, labels, bases, proposals, targets = _paired_cases()
    sample_ids = ["a", "b"]
    per_group = per_group_auc(scores, labels, sample_ids)
    assert set(per_group) == {"a", "b"}
    assert all(value > 0.9 for value in per_group.values())
    assert overall_auc(scores, labels, sample_ids) > 0.9


def test_selective_risk_curve_orders() -> None:
    scores, labels, bases, proposals, targets = _paired_cases()
    sample_ids = ["a", "b"]
    curve = selective_risk_curve(scores, labels, bases, proposals, targets, sample_ids)
    # Gated error at high coverage beats ungated error at full coverage.
    assert curve.gated_error["0.90"] < curve.ungated_error["1.00"]
    assert curve.n_patches == 2 * PATCH_GRID * PATCH_GRID


def test_select_top_gate_enforces_coverage() -> None:
    score = np.array([[2.0, 1.0], [0.0, -1.0]])
    gate = _select_top_gate(score, 0.5)
    assert gate.sum() == pytest.approx(2.0)
    assert gate[0, 0] == 1.0 and gate[1, 1] == 0.0


def test_build_benefit_report_alignment() -> None:
    scores, labels, bases, proposals, targets = _paired_cases()
    with pytest.raises(BenefitEvaluationError, match="aligned"):
        build_benefit_report("p", scores[:1], labels, bases, proposals, targets, ["a", "b"])


def test_build_benefit_report_contents() -> None:
    scores, labels, bases, proposals, targets = _paired_cases()
    report = build_benefit_report(
        "test-predictor", scores, labels, bases, proposals, targets, ["a", "b"]
    )
    assert report.predictor == "test-predictor"
    assert report.overall_auc > 0.9
    assert report.selective_risk.n_patches > 0
    assert "group_auc_aggregate" in report.calibration
