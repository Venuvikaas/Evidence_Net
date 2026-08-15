"""Structural summary checks on controlled fixtures (Phase 4 box 6).

Verifies connected-component counting and edge-displacement behavior on
hand-built fixtures, plus the magnitude/energy/structural-change summaries
against analytic expectations.
"""

from __future__ import annotations

import numpy as np

from evidence_net.evaluation.metrics import edge_displacement, ssim
from evidence_net.evaluation.proposal_metrics import (
    connected_components,
    proposal_effect_summary,
    proposal_energy,
    proposal_magnitude,
    structural_change,
)


def test_connected_components_separated_blobs() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[6:8, 6:8] = True
    assert connected_components(mask) == 2


def test_connected_components_connected_shape() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:5, 1] = True
    mask[1, 1:5] = True  # L-shape, connected
    assert connected_components(mask) == 1


def test_connected_components_empty() -> None:
    assert connected_components(np.zeros((5, 5), dtype=bool)) == 0


def test_connected_components_diagonal_not_connected() -> None:
    # 4-connectivity: diagonal touch is two components.
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    assert connected_components(mask) == 2


def test_connected_components_full_mask_single() -> None:
    assert connected_components(np.ones((8, 8), dtype=bool)) == 1


def test_proposal_magnitude_scales_with_amplitude() -> None:
    base = np.zeros((16, 16), dtype=np.float64)
    small = 0.01 * np.ones((16, 16))
    large = 0.05 * np.ones((16, 16))
    small_summary = proposal_magnitude(base, small)
    large_summary = proposal_magnitude(base, large)
    assert abs(large_summary["mean_abs"] - 5 * small_summary["mean_abs"]) < 1e-12
    assert abs(large_summary["max_abs"] - 0.05) < 1e-12
    # Flat base: range is zero, so the relative mean uses the epsilon guard.
    assert np.isfinite(large_summary["relative_mean"])


def test_proposal_magnitude_relative_to_base_range() -> None:
    base = np.zeros((16, 16), dtype=np.float64)
    base[8:, :] = 1.0  # range = 1
    proposal = 0.1 * np.ones((16, 16))
    summary = proposal_magnitude(base, proposal)
    assert abs(summary["relative_mean"] - 0.1) < 1e-9


def test_proposal_energy_is_normalized() -> None:
    rng = np.random.default_rng(0)
    for shape in [(16, 16), (32, 32)]:
        energy = proposal_energy(rng.normal(size=shape))
        assert abs(sum(energy.values()) - 1.0) < 1e-9


def test_structural_change_shifted_edge() -> None:
    # Base has a vertical edge at column 4; candidate at column 8.
    base = np.zeros((16, 16), dtype=np.float64)
    base[:, 4:] = 1.0
    candidate = np.zeros((16, 16), dtype=np.float64)
    candidate[:, 8:] = 1.0
    change = structural_change(base, candidate)
    displacement = edge_displacement(base, candidate)
    assert change["edge_displacement_px"] == displacement
    assert change["edge_displacement_px"] > 0.0
    assert change["ssim"] == ssim(base, candidate)
    # A pure translation preserves the edge magnitude map, so the delta is ~0.
    assert abs(change["edge_magnitude_delta"]) < 1e-9


def test_structural_change_identical_is_zero() -> None:
    image = np.zeros((16, 16), dtype=np.float64)
    image[:, 4:] = 1.0
    change = structural_change(image, image)
    assert change["edge_displacement_px"] == 0.0
    assert change["ssim"] == 1.0
    assert change["edge_magnitude_delta"] == 0.0


def test_proposal_effect_summary_flat() -> None:
    base = np.zeros((16, 16), dtype=np.float64)
    base[:, 4:] = 1.0
    zero_proposal = np.zeros((16, 16))
    summary = proposal_effect_summary(base, zero_proposal, base)
    assert summary["mean_abs"] == 0.0
    assert summary["max_abs"] == 0.0
    assert summary["edge_displacement_px"] == 0.0
    assert summary["ssim"] == 1.0
    # A zero proposal has no energy anywhere; every band fraction is 0.
    assert all(v == 0.0 for k, v in summary.items() if k.startswith("energy_"))
