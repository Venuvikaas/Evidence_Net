"""Oracle gating and headroom tests (Phase 4 boxes 7-8)."""

from __future__ import annotations

import numpy as np

from evidence_net.evaluation.oracle import (
    PATCH_SIZE,
    OracleDecision,
    coverage,
    oracle_decide,
    oracle_decisions,
    oracle_output,
    patch_gate,
    pixel_gate,
    risk,
)


def _bar_image(size: int = 32, lo: int = 8, hi: int = 24, value: float = 1.0) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.float64)
    image[:, lo:hi] = value
    return image


def test_pixel_gate_accepts_strict_improvement() -> None:
    target = np.full((32, 32), 0.5)
    base = np.zeros((32, 32))  # error 0.5 everywhere
    candidate = np.full((32, 32), 0.5)  # error 0 everywhere
    gate = pixel_gate(base, candidate, target)
    assert gate.mean() == 1.0


def test_pixel_gate_rejects_worse_candidate() -> None:
    target = np.full((32, 32), 0.5)
    base = np.full((32, 32), 0.5)  # error 0
    candidate = np.zeros((32, 32))  # error 0.5
    assert pixel_gate(base, candidate, target).mean() == 0.0


def test_pixel_gate_rejects_ties() -> None:
    target = np.full((32, 32), 0.5)
    base = np.zeros((32, 32))
    assert pixel_gate(base, base.copy(), target).mean() == 0.0  # strict


def test_pixel_gate_mixed_region() -> None:
    # Candidate fixes the left half and breaks the right half.
    target = np.full((64, 64), 0.5)
    base = np.zeros((64, 64))
    candidate = np.full((64, 64), 0.5)
    candidate[:, 32:] = 0.0  # worse on the right
    gate = pixel_gate(base, candidate, target)
    assert abs(gate[:, :32].mean() - 1.0) < 1e-9
    assert abs(gate[:, 32:].mean() - 0.0) < 1e-9


def test_patch_gate_accepts_fixed_patch_rejects_broken_patch() -> None:
    size = PATCH_SIZE * 4
    target = np.full((size, size), 0.5)
    base = np.zeros((size, size))
    candidate = np.full((size, size), 0.5)
    # Break the top-right patch quadrant only.
    candidate[:PATCH_SIZE, PATCH_SIZE * 2 :] = 0.0
    gate = patch_gate(base, candidate, target)
    assert gate[:PATCH_SIZE, :PATCH_SIZE].mean() == 1.0  # fixed patch
    assert gate[:PATCH_SIZE, PATCH_SIZE * 2 :].mean() == 0.0  # broken patch


def test_patch_gate_map_shape_matches_grid() -> None:
    gate = patch_gate(np.zeros((32, 32)), np.ones((32, 32)), np.full((32, 32), 0.5))
    assert gate.shape == (32, 32)
    assert set(np.unique(gate)).issubset({0, 1})


def test_oracle_output_composes_gate() -> None:
    base = np.zeros((16, 16))
    proposal = np.full((16, 16), 0.1)
    gate = np.zeros((16, 16), dtype=np.uint8)
    gate[8:, 8:] = 1
    output = oracle_output(base, proposal, gate)
    assert np.allclose(output[:8, :8], 0.0)
    assert np.allclose(output[8:, 8:], 0.1)


def test_coverage_and_risk_complement() -> None:
    gate = np.zeros((10, 10), dtype=np.uint8)
    gate[:4] = 1
    assert coverage(gate) == 0.4
    assert risk(gate) == 0.6


def test_oracle_decide_metrics_ordering() -> None:
    # Candidate strictly improves over base everywhere -> oracle beats base.
    target = np.full((32, 32), 0.5)
    base = np.zeros_like(target)  # error 0.5
    candidate = target.copy()  # error 0
    decision = oracle_decide("s1", base, candidate - base, candidate, target)
    assert isinstance(decision, OracleDecision)
    assert decision.pixel_coverage == 1.0
    assert decision.patch_coverage == 1.0
    # Oracle-patch output equals the candidate when every patch is accepted.
    assert np.allclose(decision.oracle_patch_metrics["psnr"], decision.candidate_metrics["psnr"])


def test_oracle_decisions_aligned() -> None:
    sample_ids = ["a", "b"]
    targets = [np.full((32, 32), 0.5), np.full((32, 32), 0.3)]
    bases = [np.zeros_like(t) for t in targets]
    candidates = [t.copy() for t in targets]
    decisions = oracle_decisions(sample_ids, bases, candidates, candidates, targets)
    assert [decision.sample_id for decision in decisions] == sample_ids
    assert all(decision.patch_coverage == 1.0 for decision in decisions)
