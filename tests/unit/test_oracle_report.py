"""Oracle headroom report tests (Phase 4 box 8)."""

from __future__ import annotations

import numpy as np

from evidence_net.evaluation.oracle import oracle_decisions
from evidence_net.evaluation.oracle_report import build_headroom_report, headroom_gain


def _sample(n: int = 4, value: float = 0.5) -> tuple[list, list, list, list, list]:
    targets = [np.full((32, 32), value + 0.1 * i) for i in range(n)]
    bases = [np.zeros_like(target) for target in targets]
    proposals = [target - base for target, base in zip(targets, bases, strict=True)]
    candidates = [
        np.clip(base + proposal, 0, 1) for base, proposal in zip(bases, proposals, strict=True)
    ]
    ids = [f"s{i}" for i in range(n)]
    return ids, bases, proposals, candidates, targets


def test_report_aggregates_coverage_and_metrics() -> None:
    ids, bases, proposals, candidates, targets = _sample()
    decisions = oracle_decisions(ids, bases, proposals, candidates, targets)
    report = build_headroom_report(decisions, bases=bases, proposals=proposals, targets=targets)
    assert report.n_groups == 4
    assert report.coverage["pixel"]["mean"] == 1.0
    assert report.coverage["patch"]["mean"] == 1.0
    assert report.risk["patch"]["mean"] == 0.0
    # Oracle-patch output equals the (perfect) candidate here.
    assert report.oracle_patch_metrics["mae"]["mean"] == report.candidate_metrics["mae"]["mean"]
    assert report.base_metrics["mae"]["mean"] > report.candidate_metrics["mae"]["mean"]


def test_report_structural_impact_present() -> None:
    ids, bases, proposals, candidates, targets = _sample()
    decisions = oracle_decisions(ids, bases, proposals, candidates, targets)
    report = build_headroom_report(decisions, bases=bases, proposals=proposals, targets=targets)
    assert "edge_displacement_px" in report.structural_impact
    assert "structural_error" in report.structural_impact


def test_headroom_gain_signs() -> None:
    ids, bases, proposals, candidates, targets = _sample()
    decisions = oracle_decisions(ids, bases, proposals, candidates, targets)
    report = build_headroom_report(decisions, bases=bases, proposals=proposals, targets=targets)
    gain = headroom_gain(report)
    assert gain["oracle_vs_base_mae"] < 0.0  # oracle lowers MAE vs Base
    assert gain["oracle_vs_base_ssim"] > 0.0
    assert gain["oracle_vs_candidate_mae"] == 0.0  # perfect oracle == candidate


def test_report_as_dict_roundtrip() -> None:
    ids, bases, proposals, candidates, targets = _sample(n=2)
    decisions = oracle_decisions(ids, bases, proposals, candidates, targets)
    report = build_headroom_report(decisions, bases=bases, proposals=proposals, targets=targets)
    data = report.as_dict()
    assert data["n_groups"] == 2
    assert set(data) == {
        "n_groups",
        "coverage",
        "risk",
        "base_metrics",
        "candidate_metrics",
        "oracle_pixel_metrics",
        "oracle_patch_metrics",
        "structural_impact",
    }
