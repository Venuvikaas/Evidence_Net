"""Oracle headroom reports (Phase 4, box 8).

Aggregates oracle decisions over a paired sample into coverage-risk and
structural-impact reports with group-bootstrap confidence intervals, so the
headroom of selective acceptance is stated with the same statistical
discipline as every other metric in the harness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from evidence_net.evaluation.metrics import edge_displacement, structural_error
from evidence_net.evaluation.oracle import OracleDecision, oracle_output
from evidence_net.evaluation.statistics import grouped_bootstrap_ci

_PRIMARY = ("psnr", "ssim", "mae")


@dataclass(frozen=True)
class HeadroomReport:
    """Aggregated oracle headroom over a paired sample."""

    n_groups: int
    coverage: dict[str, dict[str, float]]
    risk: dict[str, dict[str, float]]
    base_metrics: dict[str, dict[str, float]]
    candidate_metrics: dict[str, dict[str, float]]
    oracle_pixel_metrics: dict[str, dict[str, float]]
    oracle_patch_metrics: dict[str, dict[str, float]]
    structural_impact: dict[str, dict[str, float]]

    def as_dict(self) -> dict[str, object]:
        return {
            "n_groups": self.n_groups,
            "coverage": self.coverage,
            "risk": self.risk,
            "base_metrics": self.base_metrics,
            "candidate_metrics": self.candidate_metrics,
            "oracle_pixel_metrics": self.oracle_pixel_metrics,
            "oracle_patch_metrics": self.oracle_patch_metrics,
            "structural_impact": self.structural_impact,
        }


def _bootstrap(values: dict[str, float], *, n_boot: int, seed: int) -> dict[str, float]:
    return grouped_bootstrap_ci(values, n_boot=n_boot, seed=seed).as_dict()


def _metric_values(
    decisions: Sequence[OracleDecision], field: str, metric: str
) -> dict[str, float]:
    return {decision.sample_id: float(getattr(decision, field)[metric]) for decision in decisions}


def _coverage_values(decisions: Sequence[OracleDecision], granularity: str) -> dict[str, float]:
    return {
        decision.sample_id: (
            decision.pixel_coverage if granularity == "pixel" else decision.patch_coverage
        )
        for decision in decisions
    }


def _structural_impact(
    decisions: Sequence[OracleDecision],
    *,
    bases: Sequence[np.ndarray],
    proposals: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
) -> dict[str, dict[str, float]]:
    """Structural metrics of base, candidate, and oracle-patch outputs."""
    fields = ("edge_displacement_px", "structural_error")
    results: dict[str, dict[str, float]] = {field: {} for field in fields}
    for decision, base, proposal, target in zip(decisions, bases, proposals, targets, strict=True):
        oracle_patch = oracle_output(base, proposal, decision.patch_gate)
        results["edge_displacement_px"][decision.sample_id] = edge_displacement(
            target, oracle_patch
        )
        results["structural_error"][decision.sample_id] = structural_error(target, oracle_patch)
    return results


def build_headroom_report(
    decisions: Sequence[OracleDecision],
    *,
    bases: Sequence[np.ndarray] | None = None,
    proposals: Sequence[np.ndarray] | None = None,
    targets: Sequence[np.ndarray] | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> HeadroomReport:
    """Aggregate oracle decisions into a coverage-risk / structural report."""
    coverage_report: dict[str, dict[str, float]] = {}
    risk_report: dict[str, dict[str, float]] = {}
    for granularity in ("pixel", "patch"):
        values = _coverage_values(decisions, granularity)
        coverage_report[granularity] = _bootstrap(values, n_boot=n_boot, seed=seed)
        risk_report[granularity] = _bootstrap(
            {sid: 1.0 - value for sid, value in values.items()},
            n_boot=n_boot,
            seed=seed,
        )

    base_metrics: dict[str, dict[str, float]] = {}
    candidate_metrics: dict[str, dict[str, float]] = {}
    oracle_pixel_metrics: dict[str, dict[str, float]] = {}
    oracle_patch_metrics: dict[str, dict[str, float]] = {}
    for metric in _PRIMARY:
        base_metrics[metric] = _bootstrap(
            _metric_values(decisions, "base_metrics", metric), n_boot=n_boot, seed=seed
        )
        candidate_metrics[metric] = _bootstrap(
            _metric_values(decisions, "candidate_metrics", metric), n_boot=n_boot, seed=seed
        )
        oracle_pixel_metrics[metric] = _bootstrap(
            _metric_values(decisions, "oracle_pixel_metrics", metric), n_boot=n_boot, seed=seed
        )
        oracle_patch_metrics[metric] = _bootstrap(
            _metric_values(decisions, "oracle_patch_metrics", metric), n_boot=n_boot, seed=seed
        )

    structural: dict[str, dict[str, float]] = {}
    if bases is not None and proposals is not None and targets is not None:
        impact = _structural_impact(decisions, bases=bases, proposals=proposals, targets=targets)
        for field, values in impact.items():
            structural[field] = _bootstrap(values, n_boot=n_boot, seed=seed)

    return HeadroomReport(
        n_groups=len(decisions),
        coverage=coverage_report,
        risk=risk_report,
        base_metrics=base_metrics,
        candidate_metrics=candidate_metrics,
        oracle_pixel_metrics=oracle_pixel_metrics,
        oracle_patch_metrics=oracle_patch_metrics,
        structural_impact=structural,
    )


def headroom_gain(report: HeadroomReport) -> dict[str, float]:
    """Oracle-patch gain over Base and over the ungated candidate.

    Returns mean improvements (psnr dB, ssim, mae) of the oracle-patch output
    versus the Base, and the same versus the ungated candidate.
    """
    gain: dict[str, float] = {}
    for metric in _PRIMARY:
        gain[f"oracle_vs_base_{metric}"] = (
            report.oracle_patch_metrics[metric]["mean"] - report.base_metrics[metric]["mean"]
        )
        gain[f"oracle_vs_candidate_{metric}"] = (
            report.oracle_patch_metrics[metric]["mean"] - report.candidate_metrics[metric]["mean"]
        )
    return gain
