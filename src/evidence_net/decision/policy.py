"""Selective action policy (Phase 6, decision-policy-v1).

Each 16x16 patch gets exactly one action from the calibrated benefit
probability ``p``:

- **accept** (``p >= accept_threshold``): emit the ungated candidate.
- **attenuate** (``reject_threshold <= p < accept_threshold``): emit
  ``b + g(p) * d`` with the documented linear mapping.
- **reject** (``p < reject_threshold``): emit the frozen Base.

The **unresolved mask** is separate and orthogonal: a patch is unresolved
when the policy has no basis to trust either output (v1: patch edge density
above a declared threshold, from the EXP-004 failure catalogue). **Rejection
emits the Base but never certifies it** — rejected patches may be
unresolved, and the mask is reported separately (kill-switch rule).

Thresholds are chosen on validation/calibration data only and then frozen;
``fit_policy`` rejects any other split (same isolation rule as
``calibration-version-v1``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from evidence_net.benefit.labels import PATCH_GRID, PATCH_SIZE
from evidence_net.evaluation.metrics import edge_magnitude

POLICY_VERSION = "policy-v1"

# Action codes (stable, versioned).
ACCEPT = "accept"
ATTENUATE = "attenuate"
REJECT = "reject"
ACTIONS = (ACCEPT, ATTENUATE, REJECT)

# Allowed splits for threshold selection (dataset-splits-v1.json).
POLICY_FIT_SPLITS = ("validation", "calibration")


class PolicyError(ValueError):
    """Raised when the policy cannot be applied to the given inputs."""


@dataclass(frozen=True)
class PolicyConfig:
    """Frozen policy parameters (decision-policy-v1)."""

    accept_threshold: float = 0.75
    reject_threshold: float = 0.35
    unresolved_edge_density: float = 0.35
    version: str = POLICY_VERSION

    def validate(self) -> None:
        if not 0.0 < self.reject_threshold < self.accept_threshold < 1.0:
            raise PolicyError(
                "thresholds must satisfy 0 < reject < accept < 1, got "
                f"reject={self.reject_threshold}, accept={self.accept_threshold}"
            )
        if not 0.0 <= self.unresolved_edge_density <= 1.0:
            raise PolicyError(
                f"unresolved_edge_density must be in [0, 1], got {self.unresolved_edge_density}"
            )


def attenuation_gate(probability: np.ndarray, config: PolicyConfig) -> np.ndarray:
    """Linear ``g(p)`` in [0, 1] for the attenuate band (documented mapping)."""
    p = np.asarray(probability, dtype=np.float64)
    numerator = p - config.reject_threshold
    denominator = config.accept_threshold - config.reject_threshold
    return np.clip(numerator / denominator, 0.0, 1.0)


@dataclass(frozen=True)
class ActionMap:
    """Per-patch actions, gates, and the orthogonal unresolved mask."""

    sample_id: str
    actions: np.ndarray  # (16, 16) string array
    gates: np.ndarray  # (16, 16) in [0, 1]; 1 for accept, g(p) for attenuate, 0 for reject
    unresolved: np.ndarray  # (16, 16) bool mask
    probabilities: np.ndarray  # (16, 16) calibrated benefit probability

    def counts(self) -> dict[str, int]:
        return {action: int((self.actions == action).sum()) for action in ACTIONS}

    def fractions(self) -> dict[str, float]:
        total = int(self.actions.size)
        return {action: count / total for action, count in self.counts().items()}

    def coverage(self) -> float:
        """Fraction of patches where the proposal is (fully or partially) applied."""
        return float((self.gates > 0.0).mean()) if self.gates.size else 0.0

    def unresolved_fraction(self) -> float:
        return float(self.unresolved.mean()) if self.unresolved.size else 0.0


def _patch_edge_density(input_grid: np.ndarray) -> np.ndarray:
    """Mean normalized edge magnitude per patch on the output grid."""
    magnitude = edge_magnitude(np.asarray(input_grid, dtype=np.float64))
    density = np.zeros((PATCH_GRID, PATCH_GRID), dtype=np.float64)
    for row in range(PATCH_GRID):
        for col in range(PATCH_GRID):
            patch = magnitude[
                row * PATCH_SIZE : (row + 1) * PATCH_SIZE,
                col * PATCH_SIZE : (col + 1) * PATCH_SIZE,
            ]
            density[row, col] = float(patch.mean())
    return density


def _validate_grids(
    probability: np.ndarray,
    input_grid: np.ndarray,
    base_grid: np.ndarray,
    proposal_grid: np.ndarray,
) -> None:
    if probability.shape != (PATCH_GRID, PATCH_GRID):
        raise PolicyError(
            f"probability must be the {PATCH_GRID}x{PATCH_GRID} patch grid, got {probability.shape}"
        )
    grids = (input_grid, base_grid, proposal_grid)
    shapes = {array.shape for array in grids}
    if len(shapes) != 1:
        raise PolicyError(f"input/base/proposal grids must share one shape, got {shapes}")
    if next(iter(shapes)) != (PATCH_GRID * PATCH_SIZE, PATCH_GRID * PATCH_SIZE):
        raise PolicyError(
            f"grids must be on the {PATCH_GRID * PATCH_SIZE}x"
            f"{PATCH_GRID * PATCH_SIZE} output grid, got {next(iter(shapes))}"
        )


def apply_policy(
    sample_id: str,
    probability: np.ndarray,
    input_grid: np.ndarray,
    base_grid: np.ndarray,
    proposal_grid: np.ndarray,
    config: PolicyConfig,
) -> ActionMap:
    """Assign actions, gates, and the unresolved mask for one sample."""
    _validate_grids(probability, input_grid, base_grid, proposal_grid)
    p = np.asarray(probability, dtype=np.float64)
    actions = np.full((PATCH_GRID, PATCH_GRID), REJECT, dtype=object)
    actions[p >= config.accept_threshold] = ACCEPT
    in_band = (p >= config.reject_threshold) & (p < config.accept_threshold)
    actions[in_band] = ATTENUATE
    gates = np.zeros((PATCH_GRID, PATCH_GRID), dtype=np.float64)
    gates[actions == ACCEPT] = 1.0
    gates[actions == ATTENUATE] = attenuation_gate(p, config)[in_band]
    density = _patch_edge_density(input_grid)
    unresolved = density >= config.unresolved_edge_density
    return ActionMap(
        sample_id=sample_id,
        actions=actions,
        gates=gates,
        unresolved=unresolved,
        probabilities=p,
    )


def policy_outputs(
    action_map: ActionMap,
    base_grid: np.ndarray,
    proposal_grid: np.ndarray,
) -> np.ndarray:
    """Compose the gated output grid from the action map's gates."""
    gate_map = np.repeat(np.repeat(action_map.gates, PATCH_SIZE, axis=0), PATCH_SIZE, axis=1)
    return np.clip(base_grid + gate_map * proposal_grid, 0.0, 1.0)


def _validate_threshold_data(probabilities: np.ndarray, labels: np.ndarray, *, split: str) -> None:
    if split not in POLICY_FIT_SPLITS:
        raise PolicyError(
            f"policy thresholds may only be fit on {POLICY_FIT_SPLITS}, got split={split!r}"
        )
    if probabilities.shape != labels.shape:
        raise PolicyError("probabilities and labels must be aligned")
    flat = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(flat)):
        raise PolicyError("policy probabilities must be finite")


def fit_policy_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    split: str,
    accept_threshold: float = 0.75,
    reject_threshold: float = 0.35,
    unresolved_edge_density: float = 0.35,
) -> PolicyConfig:
    """Freeze a policy from validation/calibration data only.

    ``probabilities`` are calibrated benefit probabilities and ``labels`` the
    deterministic benefit labels; the thresholds are checked for
    consistency with the data (e.g. the accept band must not be empty) but
    are otherwise declared, per decision-policy-v1. Rejecting any other
    split is a regression-tested kill switch.
    """
    _validate_threshold_data(probabilities, labels, split=split)
    config = PolicyConfig(
        accept_threshold=accept_threshold,
        reject_threshold=reject_threshold,
        unresolved_edge_density=unresolved_edge_density,
    )
    config.validate()
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    np.asarray(labels, dtype=np.float64).reshape(-1)
    accepted = p >= config.accept_threshold
    rejected = p < config.reject_threshold
    if not accepted.any():
        raise PolicyError("accept band is empty on the fit split; thresholds unusable")
    if not rejected.any():
        raise PolicyError("reject band is empty on the fit split; thresholds unusable")
    return config


def action_fractions(maps: Sequence[ActionMap]) -> dict[str, float]:
    """Aggregate action fractions over samples (patches pooled for the report)."""
    totals = {action: 0 for action in ACTIONS}
    total = 0
    for action_map in maps:
        for action, count in action_map.counts().items():
            totals[action] += count
        total += int(action_map.actions.size)
    if total == 0:
        return {action: 0.0 for action in ACTIONS}
    return {action: count / total for action, count in totals.items()}


def coverage_risk_report(
    maps: Sequence[ActionMap],
    bases: Sequence[np.ndarray],
    proposals: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
) -> dict[str, object]:
    """Coverage, unresolved area, and gated-error risk over samples.

    Returns pooled patch-level aggregates plus per-action patch MAE vs the
    target, and the unresolved-area fraction. Groups remain the statistical
    unit elsewhere; this report is the action-map summary.
    """
    if not (len(maps) == len(bases) == len(proposals) == len(targets)):
        raise PolicyError("maps, bases, proposals, and targets must be aligned")
    coverages: list[float] = []
    unresolved: list[float] = []
    action_mae: dict[str, list[float]] = {action: [] for action in ACTIONS}
    for action_map, base, proposal, target in zip(maps, bases, proposals, targets, strict=True):
        coverages.append(action_map.coverage())
        unresolved.append(action_map.unresolved_fraction())
        gated = policy_outputs(action_map, base, proposal)
        for row in range(PATCH_GRID):
            for col in range(PATCH_GRID):
                action = str(action_map.actions[row, col])
                rows = slice(row * PATCH_SIZE, (row + 1) * PATCH_SIZE)
                cols = slice(col * PATCH_SIZE, (col + 1) * PATCH_SIZE)
                error = float(np.mean(np.abs(target[rows, cols] - gated[rows, cols])))
                action_mae[action].append(error)
    return {
        "mean_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "mean_unresolved_fraction": float(np.mean(unresolved)) if unresolved else 0.0,
        "action_patch_mae": {
            action: float(np.mean(errors)) if errors else float("nan")
            for action, errors in action_mae.items()
        },
        "action_fractions": action_fractions(maps),
        "n_samples": len(maps),
    }
