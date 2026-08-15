"""Oracle gating (Phase 4, boxes 7-8).

The oracle is a study tool that sees the ground truth: for each pixel and for
each fixed-size patch it decides whether accepting the proposal reduces error
vs the Base. Per ``docs/proposal-contract.md`` section 4:

- **Pixel decision:** accept at pixel ``p`` when
  ``|c_p - x_p| < |b_p - x_p|`` (strict; ties and increases are rejected).
- **Patch decision:** on a ``PATCH_SIZE x PATCH_SIZE`` grid (16 for the
  official 256x256 output), accept patch ``r`` when the patch-level MAE of
  the candidate is strictly lower than that of the Base.

Coverage is the accepted fraction; risk is the fraction where accepting the
proposal increases error (the oracle rejects these; an ungated system takes
the harm). The oracle is never used at inference — it only measures headroom.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from evidence_net.evaluation.metrics import all_metrics, mae

PATCH_SIZE = 16


def _as_float64(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)


def pixel_gate(base: np.ndarray, candidate: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Binary per-pixel gate: 1 where the candidate strictly beats the Base."""
    b = _as_float64(base)
    c = _as_float64(candidate)
    x = _as_float64(target)
    return (np.abs(c - x) < np.abs(b - x)).astype(np.uint8)


def _patch_grid(shape: tuple[int, int]) -> np.ndarray:
    """Integer patch-index map over the output grid (H, W) -> (rows, cols)."""
    height, width = shape
    rows = np.arange(height) // PATCH_SIZE
    cols = np.arange(width) // PATCH_SIZE
    return rows[:, None] + cols[None, :] * (height // PATCH_SIZE + 1)


def patch_mae_map(base: np.ndarray, candidate: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-patch MAE of Base and candidate vs the target.

    Returns ``(base_patch_mae, candidate_patch_mae)`` float arrays indexed by
    patch id (row-major over the patch grid, excluding partial edge patches).
    """
    b = _as_float64(base)
    c = _as_float64(candidate)
    x = _as_float64(target)
    height, width = b.shape
    grid = _patch_grid((height, width))
    patch_ids = np.unique(grid)
    base_mae = np.zeros(patch_ids.max() + 1, dtype=np.float64)
    candidate_mae = np.zeros(patch_ids.max() + 1, dtype=np.float64)
    for patch_id in patch_ids:
        mask = grid == patch_id
        base_mae[patch_id] = mae(x[mask], b[mask])
        candidate_mae[patch_id] = mae(x[mask], c[mask])
    return base_mae, candidate_mae


def patch_gate(base: np.ndarray, candidate: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Binary per-patch gate map on the pixel grid (values in {0, 1})."""
    b = _as_float64(base)
    c = _as_float64(candidate)
    x = _as_float64(target)
    height, width = b.shape
    grid = _patch_grid((height, width))
    base_mae, candidate_mae = patch_mae_map(b, c, x)
    accept = (candidate_mae[grid] < base_mae[grid]).astype(np.uint8)
    return accept


def oracle_output(
    base: np.ndarray, proposal: np.ndarray, gate: np.ndarray
) -> np.ndarray:
    """Compose the gated output ``x_hat = b + g * d`` from a gate map."""
    b = _as_float64(base)
    d = _as_float64(proposal)
    g = _as_float64(gate)
    return np.clip(b + g * d, 0.0, 1.0)


def coverage(gate: np.ndarray) -> float:
    """Fraction of gated units (pixels or patches) that accept the proposal."""
    g = _as_float64(gate)
    return float(g.mean()) if g.size else 0.0


def risk(gate: np.ndarray) -> float:
    """Fraction of units where accepting the proposal would increase error.

    The oracle rejects those units; an ungated system would take the harm,
    so risk is ``1 - coverage`` under the strict binary oracle.
    """
    return 1.0 - coverage(gate)


@dataclass(frozen=True)
class OracleDecision:
    """Oracle decisions and metrics for one image."""

    sample_id: str
    pixel_gate: np.ndarray
    patch_gate: np.ndarray
    pixel_coverage: float
    patch_coverage: float
    pixel_risk: float
    patch_risk: float
    base_metrics: dict[str, float | dict[str, float]]
    candidate_metrics: dict[str, float | dict[str, float]]
    oracle_pixel_metrics: dict[str, float | dict[str, float]]
    oracle_patch_metrics: dict[str, float | dict[str, float]]


def oracle_decide(
    sample_id: str,
    base: np.ndarray,
    proposal: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
) -> OracleDecision:
    """Run pixel and patch oracle decisions plus metrics for one image."""
    pixel_gate_map = pixel_gate(base, candidate, target)
    patch_gate_map = patch_gate(base, candidate, target)
    oracle_pixel = oracle_output(base, proposal, pixel_gate_map)
    oracle_patch = oracle_output(base, proposal, patch_gate_map)
    return OracleDecision(
        sample_id=sample_id,
        pixel_gate=pixel_gate_map,
        patch_gate=patch_gate_map,
        pixel_coverage=coverage(pixel_gate_map),
        patch_coverage=coverage(patch_gate_map),
        pixel_risk=risk(pixel_gate_map),
        patch_risk=risk(patch_gate_map),
        base_metrics=all_metrics(target, base),
        candidate_metrics=all_metrics(target, candidate),
        oracle_pixel_metrics=all_metrics(target, oracle_pixel),
        oracle_patch_metrics=all_metrics(target, oracle_patch),
    )


def oracle_decisions(
    sample_ids: Sequence[str],
    bases: Sequence[np.ndarray],
    proposals: Sequence[np.ndarray],
    candidates: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
) -> list[OracleDecision]:
    """Oracle decisions over paired sample sets (all sequences aligned)."""
    return [
        oracle_decide(sample_id, base, proposal, candidate, target)
        for sample_id, base, proposal, candidate, target in zip(
            sample_ids, bases, proposals, candidates, targets, strict=True
        )
    ]
