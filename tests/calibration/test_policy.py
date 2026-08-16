"""Decision policy tests (decision-policy-v1, Phase 6)."""

from __future__ import annotations

import numpy as np
import pytest

from evidence_net.benefit.labels import OUTPUT_GRID, PATCH_GRID
from evidence_net.decision.policy import (
    ACCEPT,
    ATTENUATE,
    REJECT,
    PolicyConfig,
    PolicyError,
    action_fractions,
    apply_policy,
    attenuation_gate,
    coverage_risk_report,
    fit_policy_thresholds,
    policy_outputs,
)


def _grids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = OUTPUT_GRID
    y = np.full((grid, grid), 0.05)
    y[:, : grid // 2] = 0.6
    y[:, grid // 2 :] = 0.05
    for column in range(grid // 2, grid, 8):
        y[:, column] = 0.9
    base = np.clip(y, 0.0, 1.0)
    base[:, : grid // 2] = 0.2
    proposal = np.zeros((grid, grid))
    proposal[:, : grid // 2] = 0.4
    proposal[:, grid // 2 :] = 0.001
    return y, base, proposal


def test_config_validation() -> None:
    with pytest.raises(PolicyError, match="thresholds"):
        PolicyConfig(accept_threshold=0.3, reject_threshold=0.5).validate()
    with pytest.raises(PolicyError, match="unresolved_edge_density"):
        PolicyConfig(unresolved_edge_density=1.5).validate()
    PolicyConfig().validate()


def test_apply_policy_assigns_actions() -> None:
    y, b, d = _grids()
    probability = np.full((PATCH_GRID, PATCH_GRID), 0.5)
    probability[:, : PATCH_GRID // 2] = 0.9  # accept
    probability[:, PATCH_GRID // 2 :] = 0.2  # reject
    action_map = apply_policy("s1", probability, y, b, d, PolicyConfig())
    assert np.all(action_map.actions[:, : PATCH_GRID // 2] == ACCEPT)
    assert np.all(action_map.actions[:, PATCH_GRID // 2 :] == REJECT)
    assert np.all(action_map.gates[:, : PATCH_GRID // 2] == 1.0)
    assert np.all(action_map.gates[:, PATCH_GRID // 2 :] == 0.0)


def test_attenuate_band_uses_linear_gate() -> None:
    y, b, d = _grids()
    probability = np.full((PATCH_GRID, PATCH_GRID), 0.55)
    action_map = apply_policy("s1", probability, y, b, d, PolicyConfig())
    assert np.all(action_map.actions == ATTENUATE)
    expected = attenuation_gate(np.full((PATCH_GRID, PATCH_GRID), 0.55), PolicyConfig())
    assert np.allclose(action_map.gates, expected)
    assert np.all(action_map.gates > 0.0) and np.all(action_map.gates < 1.0)


def test_unresolved_mask_is_orthogonal_to_action() -> None:
    """A rejected patch can be unresolved: rejection never certifies Base."""
    y, b, d = _grids()
    # Right half: reject (low probability) AND high edge density (stripes) ->
    # unresolved. Left half: accept, low edge density -> resolved.
    probability = np.full((PATCH_GRID, PATCH_GRID), 0.9)
    probability[:, PATCH_GRID // 2 :] = 0.1
    config = PolicyConfig(unresolved_edge_density=0.2)
    action_map = apply_policy("s1", probability, y, b, d, config)
    assert np.all(action_map.actions[:, : PATCH_GRID // 2] == ACCEPT)
    assert np.all(action_map.actions[:, PATCH_GRID // 2 :] == REJECT)
    # The rejected (striped) right half is mostly unresolved; the accepted
    # (flat) left half is resolved.
    assert action_map.unresolved[:, PATCH_GRID // 2 :].mean() > 0.8
    assert action_map.unresolved[:, : PATCH_GRID // 2].mean() < 0.1
    # Both facts coexist: rejected and unresolved simultaneously.
    rejected_unresolved = (
        action_map.actions[:, PATCH_GRID // 2 :] == REJECT
    ) & action_map.unresolved[:, PATCH_GRID // 2 :]
    assert rejected_unresolved.any()


def test_policy_outputs_compose() -> None:
    y, b, d = _grids()
    probability = np.full((PATCH_GRID, PATCH_GRID), 0.9)
    action_map = apply_policy("s1", probability, y, b, d, PolicyConfig())
    output = policy_outputs(action_map, b, d)
    assert output.shape == (OUTPUT_GRID, OUTPUT_GRID)
    candidate = np.clip(b + d, 0.0, 1.0)
    assert np.allclose(output, candidate, atol=1e-9)


def test_grid_validation() -> None:
    with pytest.raises(PolicyError, match="patch grid"):
        apply_policy(
            "s1",
            np.zeros((8, 8)),
            np.zeros((64, 64)),
            np.zeros((64, 64)),
            np.zeros((64, 64)),
            PolicyConfig(),
        )
    with pytest.raises(PolicyError, match="share one shape"):
        apply_policy(
            "s1",
            np.zeros((PATCH_GRID, PATCH_GRID)),
            np.zeros((64, 64)),
            np.zeros((64, 64)),
            np.zeros((128, 128)),
            PolicyConfig(),
        )


def test_fit_policy_rejects_non_fit_splits() -> None:
    p = np.linspace(0.05, 0.95, 64)
    labels = (p > 0.5).astype(np.float64)
    for forbidden in ("train", "heldout-source", "test"):
        with pytest.raises(PolicyError, match="validation"):
            fit_policy_thresholds(p, labels, split=forbidden)


def test_fit_policy_requires_nonempty_bands() -> None:
    p = np.full(64, 0.9)
    labels = np.ones(64)
    with pytest.raises(PolicyError, match="reject band is empty"):
        fit_policy_thresholds(p, labels, split="calibration")


def test_fit_policy_freeze_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=128)
    labels = (p > 0.5).astype(np.float64)
    first = fit_policy_thresholds(p, labels, split="calibration")
    second = fit_policy_thresholds(p, labels, split="calibration")
    assert first == second


def test_coverage_risk_report() -> None:
    y, b, d = _grids()
    x = np.clip(b + d, 0.0, 1.0)
    x[:, :] = b  # target equals Base: only accept where the proposal helps
    x[:, : OUTPUT_GRID // 2] = np.clip(b + d, 0.0, 1.0)[:, : OUTPUT_GRID // 2]
    probability = np.full((PATCH_GRID, PATCH_GRID), 0.9)
    action_map = apply_policy("s1", probability, y, b, d, PolicyConfig())
    report = coverage_risk_report([action_map], [b], [d], [x])
    assert report["n_samples"] == 1
    assert report["mean_coverage"] == pytest.approx(1.0)
    assert set(report["action_patch_mae"]) == {ACCEPT, ATTENUATE, REJECT}
    assert 0.0 <= report["mean_unresolved_fraction"] <= 1.0


def test_action_fractions() -> None:
    y, b, d = _grids()
    probability = np.full((PATCH_GRID, PATCH_GRID), 0.5)
    probability[:, : PATCH_GRID // 2] = 0.9
    probability[:, PATCH_GRID // 2 :] = 0.2
    action_map = apply_policy("s1", probability, y, b, d, PolicyConfig())
    fractions = action_fractions([action_map])
    assert fractions[ACCEPT] == pytest.approx(0.5)
    assert fractions[REJECT] == pytest.approx(0.5)
    assert fractions[ATTENUATE] == pytest.approx(0.0)
