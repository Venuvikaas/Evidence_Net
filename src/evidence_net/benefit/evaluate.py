"""Benefit-predictor evaluation suite (Phase 5, EXP-009).

Reports are kept **separate** by design (contract: discrimination, ranking,
selective risk, and calibration are distinct claims):

- **Discrimination / ranking:** AUC per group and over all labeled patches,
  plus rank correlation between score and label.
- **Selective risk:** for each coverage level, the mean error of the
  **gated output** on accepted patches, compared with the ungated and Base
  floors — the EXP-009 primary endpoint ("predicting benefit is useful").
- **Calibration:** see ``calibration.py`` (Brier, reliability, ECE) — the
  report records ranking and calibration conclusions separately.

The statistical unit is the source group; pixels and patches are never
independent sample counts (metrics-v1 discipline).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from evidence_net.benefit.calibration import brier_score, expected_calibration_error
from evidence_net.evaluation.statistics import grouped_bootstrap_ci


class BenefitEvaluationError(ValueError):
    """Raised when benefit evaluation inputs are misaligned."""


def _check_aligned(
    scores: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    sample_ids: Sequence[str],
) -> None:
    if not (len(scores) == len(labels) == len(sample_ids)):
        raise BenefitEvaluationError("scores, labels, and sample ids must be aligned")


def per_group_auc(
    scores: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    sample_ids: Sequence[str],
) -> dict[str, float]:
    """AUC per group (patches within a group are a curve, groups are the units)."""
    _check_aligned(scores, labels, sample_ids)
    result: dict[str, float] = {}
    for sample_id, score, label in zip(sample_ids, scores, labels, strict=True):
        result[sample_id] = _auc(score.reshape(-1), label.reshape(-1))
    return result


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    y = np.asarray(label, dtype=np.float64).reshape(-1)
    positive = s[y == 1.0]
    negative = s[y == 0.0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    # Mann-Whitney U / n_pos / n_neg.
    ranks = _rankdata(np.concatenate([positive, negative]))
    sum_pos = float(ranks[: positive.size].sum())
    u = sum_pos - positive.size * (positive.size + 1.0) / 2.0
    return float(u / (positive.size * negative.size))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    # Average ties.
    sorted_values = values[order]
    index = 0
    while index < values.size:
        end = index + 1
        while end < values.size and sorted_values[end] == sorted_values[index]:
            end += 1
        if end - index > 1:
            ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    return ranks


def overall_auc(
    scores: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    sample_ids: Sequence[str],
) -> float:
    """Pooled AUC over all labeled patches (report-only; groups stay the unit)."""
    _check_aligned(scores, labels, sample_ids)
    all_scores = np.concatenate([score.reshape(-1) for score in scores])
    all_labels = np.concatenate([label.reshape(-1) for label in labels])
    return _auc(all_scores, all_labels)


@dataclass(frozen=True)
class SelectiveRisk:
    """Mean gated error at declared coverage levels, per output family."""

    coverage_levels: tuple[float, ...]
    base_error: dict[str, float]  # coverage -> mean patch MAE of Base output
    ungated_error: dict[str, float]
    gated_error: dict[str, float]  # coverage -> mean patch MAE of gated output
    n_patches: int

    def as_dict(self) -> dict[str, object]:
        return {
            "coverage_levels": list(self.coverage_levels),
            "base_error": self.base_error,
            "ungated_error": self.ungated_error,
            "gated_error": self.gated_error,
            "n_patches": self.n_patches,
        }


def _patch_mae_map(output: np.ndarray, target: np.ndarray, patch_size: int = 16) -> np.ndarray:
    out = np.asarray(output, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    rows = out.shape[0] // patch_size
    cols = out.shape[1] // patch_size
    result = np.zeros((rows, cols), dtype=np.float64)
    for row in range(rows):
        for col in range(cols):
            o = out[
                row * patch_size : (row + 1) * patch_size, col * patch_size : (col + 1) * patch_size
            ]
            t = tgt[
                row * patch_size : (row + 1) * patch_size, col * patch_size : (col + 1) * patch_size
            ]
            result[row, col] = float(np.mean(np.abs(o - t)))
    return result


def selective_risk_curve(
    scores: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    bases: Sequence[np.ndarray],
    proposals: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    sample_ids: Sequence[str],
    *,
    coverage_levels: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    patch_size: int = 16,
) -> SelectiveRisk:
    """Per-coverage mean patch MAE of base, ungated, and gated outputs.

    The gate accepts the highest-scoring patches until the coverage level is
    reached (patches are the units within each coverage sweep; the sweep is a
    report, groups remain the statistical unit elsewhere).
    """
    _check_aligned(scores, labels, sample_ids)
    if not (len(bases) == len(proposals) == len(targets) == len(sample_ids)):
        raise BenefitEvaluationError("bases, proposals, targets must be aligned with samples")

    base_error: dict[str, float] = {}
    ungated_error: dict[str, float] = {}
    gated_error: dict[str, float] = {}

    for level in coverage_levels:
        base_values: list[float] = []
        ungated_values: list[float] = []
        gated_values: list[float] = []
        for score, base, proposal, target in zip(scores, bases, proposals, targets, strict=True):
            gate = _select_top_gate(score, level)
            gate_map = np.repeat(np.repeat(gate, patch_size, axis=0), patch_size, axis=1)
            gated = np.clip(base + gate_map * proposal, 0.0, 1.0)
            base_values.extend(_patch_mae_map(base, target, patch_size).reshape(-1))
            ungated_values.extend(
                _patch_mae_map(np.clip(base + proposal, 0.0, 1.0), target, patch_size).reshape(-1)
            )
            gated_values.extend(_patch_mae_map(gated, target, patch_size).reshape(-1))
        base_error[f"{level:.2f}"] = float(np.mean(base_values))
        ungated_error[f"{level:.2f}"] = float(np.mean(ungated_values))
        gated_error[f"{level:.2f}"] = float(np.mean(gated_values))

    n_patches = int(sum(s.reshape(-1).size for s in scores))
    return SelectiveRisk(
        coverage_levels=coverage_levels,
        base_error=base_error,
        ungated_error=ungated_error,
        gated_error=gated_error,
        n_patches=n_patches,
    )


def _select_top_gate(score: np.ndarray, coverage: float) -> np.ndarray:
    """Accept the top-``coverage`` fraction of patches by score."""
    flat = np.asarray(score, dtype=np.float64).reshape(-1)
    n = flat.size
    if coverage >= 1.0:
        return np.ones(n, dtype=np.float64).reshape(score.shape)
    threshold = np.quantile(flat, 1.0 - coverage)
    gate = (flat >= threshold).astype(np.float64)
    # Enforce exactly the requested count when there are ties.
    accepted = int(round(coverage * n))
    if int(gate.sum()) > accepted:
        indices = np.argsort(-flat)
        gate[:] = 0.0
        gate[indices[:accepted]] = 1.0
    return gate.reshape(score.shape)


@dataclass(frozen=True)
class BenefitReport:
    """Separate ranking and selective-risk evidence for one predictor."""

    predictor: str
    sample_ids: tuple[str, ...]
    per_group_auc: dict[str, float]
    overall_auc: float
    selective_risk: SelectiveRisk
    calibration: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "predictor": self.predictor,
            "sample_ids": list(self.sample_ids),
            "per_group_auc": self.per_group_auc,
            "overall_auc": self.overall_auc,
            "selective_risk": self.selective_risk.as_dict(),
            "calibration": self.calibration,
        }


def build_benefit_report(
    predictor_name: str,
    scores: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    bases: Sequence[np.ndarray],
    proposals: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    sample_ids: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> BenefitReport:
    """Assemble the separate ranking and selective-risk evidence."""
    _check_aligned(scores, labels, sample_ids)
    per_group = per_group_auc(scores, labels, sample_ids)
    finite_groups = {sid: value for sid, value in per_group.items() if value == value}
    if finite_groups:
        auc_aggregate = grouped_bootstrap_ci(finite_groups, n_boot=n_boot, seed=seed).as_dict()
    else:
        auc_aggregate = {
            "mean": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_groups": 0,
            "n_boot": n_boot,
            "ci_level": 0.95,
        }
    selective = selective_risk_curve(scores, labels, bases, proposals, targets, sample_ids)
    return BenefitReport(
        predictor=predictor_name,
        sample_ids=tuple(sample_ids),
        per_group_auc=per_group,
        overall_auc=overall_auc(scores, labels, sample_ids),
        selective_risk=selective,
        calibration={
            "group_auc_aggregate": auc_aggregate,
            "note": "ranking and selective risk are reported separately from calibration",
        },
    )


def grouped_brier_report(
    predictor_name: str,
    probabilities: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    sample_ids: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    n_bins: int = 10,
) -> dict[str, object]:
    """Group-bootstrapped Brier and ECE for a calibrated predictor."""
    _check_aligned(probabilities, labels, sample_ids)
    per_group: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sample_id, probability, label in zip(sample_ids, probabilities, labels, strict=True):
        per_group[sample_id] = (
            np.asarray(probability).reshape(-1),
            np.asarray(label).reshape(-1),
        )
    group_scores = {
        sid: brier_score(probabilities, labels)
        for sid, (probabilities, labels) in per_group.items()
    }
    aggregate = grouped_bootstrap_ci(group_scores, n_boot=n_boot, seed=seed).as_dict()
    pooled_p = np.concatenate([probs for probs, _labels in per_group.values()])
    pooled_l = np.concatenate([labels for _probs, labels in per_group.values()])
    return {
        "predictor": predictor_name,
        "group_brier_aggregate": aggregate,
        "ece": expected_calibration_error(pooled_p, pooled_l, n_bins=n_bins),
        "n_bins": n_bins,
    }
