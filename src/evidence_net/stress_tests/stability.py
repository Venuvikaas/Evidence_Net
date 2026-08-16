"""Model stability diagnostics (Phase 8, stability-v1 draft).

Stability means *agreement*: how much a model's output changes when the input
is perturbed by an invertible spatial transform (and the output inverted
back), how much checkpoints of the same architecture agree, and how diverse
model errors are before models are combined. Agreement is stability, never
correctness, never a probability of truth, and never calibration
(stability-v1, Gate 7).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evidence_net.evaluation.metrics import mae
from evidence_net.evaluation.statistics import GroupedAggregate, GroupingError, grouped_bootstrap_ci

ModelFn = Callable[[np.ndarray], np.ndarray]

# Frozen bounds from stability-v1.
MAX_SHIFT = 1  # |dy|, |dx| <= 1 on the input grid
DEFAULT_SHIFTS = ((0, 0), (0, 1), (1, 0), (1, 1))
DEFAULT_FLIPS = ("h", "v")


class StabilityError(ValueError):
    """Raised for invalid stability configurations or diagnostics."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StabilityConfig:
    """Validated stability-diagnostic configuration (stability-v1)."""

    version: str = "stability-v1"
    shifts: tuple[tuple[int, int], ...] = DEFAULT_SHIFTS
    flips: tuple[str, ...] = DEFAULT_FLIPS
    min_diversity_threshold: float = 0.2
    n_boot: int = 1000
    seed: int = 0

    def validate(self) -> None:
        if self.version != "stability-v1":
            raise StabilityError(f"version must be stability-v1, got {self.version}")
        for dy, dx in self.shifts:
            if abs(dy) > MAX_SHIFT or abs(dx) > MAX_SHIFT:
                raise StabilityError(
                    f"shifts must satisfy |dy|,|dx| <= {MAX_SHIFT}, got ({dy}, {dx})"
                )
        unknown = set(self.flips) - {"h", "v"}
        if unknown:
            raise StabilityError(f"unknown flips: {sorted(unknown)}")
        if not 0.0 <= self.min_diversity_threshold <= 1.0:
            raise StabilityError(
                f"min_diversity_threshold must be in [0, 1], got {self.min_diversity_threshold}"
            )
        if self.n_boot < 1:
            raise StabilityError(f"n_boot must be >= 1, got {self.n_boot}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "shifts": [list(shift) for shift in self.shifts],
            "flips": list(self.flips),
            "min_diversity_threshold": self.min_diversity_threshold,
            "n_boot": self.n_boot,
            "seed": self.seed,
        }


def load_stability_config(path: Path) -> StabilityConfig:
    """Load and validate a stability config from YAML (stability-v1)."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise StabilityError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise StabilityError(f"config root must be a mapping: {path}")
    allowed = {"version", "shifts", "flips", "min_diversity_threshold", "n_boot", "seed"}
    unknown = set(raw) - allowed
    if unknown:
        raise StabilityError(f"unknown config keys: {sorted(unknown)}")

    def pick(key: str, default: Any) -> Any:
        return raw.get(key, default)

    shifts = pick("shifts", [list(shift) for shift in DEFAULT_SHIFTS])
    if not isinstance(shifts, list) or not all(
        isinstance(shift, list) and len(shift) == 2 and all(isinstance(v, int) for v in shift)
        for shift in shifts
    ):
        raise StabilityError("shifts must be a list of [dy, dx] int pairs")
    flips = pick("flips", list(DEFAULT_FLIPS))
    if not isinstance(flips, list) or not all(isinstance(f, str) for f in flips):
        raise StabilityError("flips must be a list of strings")
    threshold = pick("min_diversity_threshold", 0.2)
    if not isinstance(threshold, int | float) or isinstance(threshold, bool):
        raise StabilityError("min_diversity_threshold must be a number")
    n_boot = pick("n_boot", 1000)
    if not isinstance(n_boot, int) or isinstance(n_boot, bool):
        raise StabilityError("n_boot must be an int")
    seed = pick("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise StabilityError("seed must be an int")

    config = StabilityConfig(
        version=str(pick("version", "stability-v1")),
        shifts=tuple(tuple(shift) for shift in shifts),
        flips=tuple(flips),
        min_diversity_threshold=float(threshold),
        n_boot=n_boot,
        seed=seed,
    )
    config.validate()
    return config


# ---------------------------------------------------------------------------
# Invertible perturbations (stability-v1 section 1.1)
# ---------------------------------------------------------------------------


class Perturbation:
    """An invertible spatial perturbation with a known output-grid inverse."""

    name: str

    def perturb(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def invert(self, image: np.ndarray) -> np.ndarray:
        """Invert on the output grid (scale 2 for shifts; flips are self-inverse)."""
        raise NotImplementedError


class ShiftPerturbation(Perturbation):
    """Pixel shift on the input grid; inverse shifts the output grid by 2x."""

    def __init__(self, dy: int, dx: int) -> None:
        if abs(dy) > MAX_SHIFT or abs(dx) > MAX_SHIFT:
            raise StabilityError(f"shifts must satisfy |dy|,|dx| <= {MAX_SHIFT}, got ({dy}, {dx})")
        self.dy = dy
        self.dx = dx
        self.name = f"shift-{dy}-{dx}"

    def perturb(self, image: np.ndarray) -> np.ndarray:
        return np.roll(image, (self.dy, self.dx), axis=(0, 1))

    def invert(self, image: np.ndarray) -> np.ndarray:
        return np.roll(image, (-2 * self.dy, -2 * self.dx), axis=(0, 1))


class FlipPerturbation(Perturbation):
    """Horizontal or vertical flip (self-inverse on both grids)."""

    def __init__(self, axis: str) -> None:
        if axis not in ("h", "v"):
            raise StabilityError(f"flip axis must be 'h' or 'v', got {axis}")
        self.axis = axis
        self.name = f"flip-{axis}"

    def perturb(self, image: np.ndarray) -> np.ndarray:
        return np.fliplr(image) if self.axis == "h" else np.flipud(image)

    def invert(self, image: np.ndarray) -> np.ndarray:
        return self.perturb(image)


def build_perturbations(config: StabilityConfig) -> list[Perturbation]:
    """Build the declared invertible perturbation family from a validated config."""
    config.validate()
    perturbations: list[Perturbation] = [ShiftPerturbation(dy, dx) for dy, dx in config.shifts]
    perturbations.extend(FlipPerturbation(axis) for axis in config.flips)
    return perturbations


def _require_unique_groups(group_ids: Sequence[str]) -> None:
    """Reject duplicate group ids so per-group statistics stay one value per group."""
    if len(set(group_ids)) != len(group_ids):
        duplicates = sorted({group for group in group_ids if group_ids.count(group) > 1})
        raise GroupingError(f"duplicate group ids in stability diagnostic: {duplicates}")


# ---------------------------------------------------------------------------
# Perturbation stability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerturbationResult:
    """Deviation of one perturbation over the evaluated groups."""

    perturbation: str
    per_group: dict[str, float]
    aggregate: GroupedAggregate

    def as_dict(self) -> dict[str, Any]:
        return {
            "perturbation": self.perturbation,
            "per_group": self.per_group,
            "aggregate": self.aggregate.as_dict(),
        }


@dataclass(frozen=True)
class PerturbationStability:
    """Deviation distribution across the perturbation family."""

    n_groups: int
    results: tuple[PerturbationResult, ...]
    across: dict[str, float | str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_groups": self.n_groups,
            "perturbations": [result.as_dict() for result in self.results],
            "across": self.across,
        }


def perturbation_stability(
    model_fn: ModelFn,
    inputs: Sequence[np.ndarray],
    group_ids: Sequence[str],
    perturbations: Sequence[Perturbation],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> PerturbationStability:
    """Per-perturbation deviation of ``invert(model(perturb(x)))`` vs ``model(x)``.

    The zero shift is the identity and must measure zero deviation (sanity
    check). Groups are the statistical units (metrics-v1 discipline).
    """
    if len(inputs) != len(group_ids) or not inputs:
        raise GroupingError("perturbation stability requires inputs and group_ids of equal length")
    _require_unique_groups(group_ids)
    baseline = [model_fn(input_) for input_ in inputs]
    results: list[PerturbationResult] = []
    for perturbation in perturbations:
        per_group: dict[str, float] = {}
        for group_id, input_, output in zip(group_ids, inputs, baseline, strict=True):
            inverted = perturbation.invert(model_fn(perturbation.perturb(input_)))
            per_group[group_id] = float(mae(output, inverted))
        results.append(
            PerturbationResult(
                perturbation=perturbation.name,
                per_group=per_group,
                aggregate=grouped_bootstrap_ci(per_group, n_boot=n_boot, seed=seed),
            )
        )
    means = {result.perturbation: result.aggregate.mean for result in results}
    argmax = max(means, key=lambda name: means[name])
    across: dict[str, float | str] = {
        "max_mean_deviation": float(max(means.values())),
        "argmax_perturbation": argmax,
        "n_perturbations": len(results),
    }
    return PerturbationStability(
        n_groups=len(group_ids),
        results=tuple(results),
        across=across,
    )


# ---------------------------------------------------------------------------
# Checkpoint agreement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointStability:
    """Pairwise output agreement of same-architecture checkpoints."""

    n_groups: int
    pairs: dict[str, GroupedAggregate]
    across: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_groups": self.n_groups,
            "pairs": {name: aggregate.as_dict() for name, aggregate in self.pairs.items()},
            "across": self.across,
        }


def checkpoint_agreement(
    models: Sequence[tuple[str, ModelFn]],
    inputs: Sequence[np.ndarray],
    group_ids: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> CheckpointStability:
    """Per-group MAE between every pair of model outputs on identical inputs.

    A model compared with itself must report zero agreement deviation
    (sanity). Pairs are keyed ``a-vs-b`` in sorted order.
    """
    if len(inputs) != len(group_ids) or not inputs:
        raise GroupingError("checkpoint agreement requires inputs and group_ids of equal length")
    if not models:
        raise StabilityError("checkpoint agreement requires at least one model")
    _require_unique_groups(group_ids)
    outputs: dict[str, list[np.ndarray]] = {}
    for name, model_fn in models:
        outputs[name] = [model_fn(input_) for input_ in inputs]
    names = sorted(outputs)
    pairs: dict[str, GroupedAggregate] = {}
    for i, name_a in enumerate(names):
        for name_b in names[i:]:
            key = name_a if name_a == name_b else f"{name_a}-vs-{name_b}"
            per_group = {
                group_id: float(mae(output_a, output_b))
                for group_id, output_a, output_b in zip(
                    group_ids, outputs[name_a], outputs[name_b], strict=True
                )
            }
            pairs[key] = grouped_bootstrap_ci(per_group, n_boot=n_boot, seed=seed)
    means = [aggregate.mean for aggregate in pairs.values()]
    return CheckpointStability(
        n_groups=len(group_ids),
        pairs=pairs,
        across={
            "mean_pair_agreement": float(np.mean(means)),
            "max_pair_agreement": float(np.max(means)),
        },
    )


# ---------------------------------------------------------------------------
# Error diversity (stability-v1 section 1.3)
# ---------------------------------------------------------------------------


def _pairwise_diversity(
    errors_a: np.ndarray, errors_b: np.ndarray, *, tolerance: float
) -> dict[str, float]:
    """Pairwise error-diversity metrics between two signed error maps."""
    signed_a = np.asarray(errors_a, dtype=np.float64)
    signed_b = np.asarray(errors_b, dtype=np.float64)
    if signed_a.shape != signed_b.shape:
        raise StabilityError(
            f"error maps must share shape, got {signed_a.shape} vs {signed_b.shape}"
        )
    magnitude_a = np.abs(signed_a)
    magnitude_b = np.abs(signed_b)
    std_a = float(magnitude_a.std())
    std_b = float(magnitude_b.std())
    if std_a == 0.0 and std_b == 0.0:
        correlation = 1.0 if np.array_equal(magnitude_a, magnitude_b) else 0.0
    elif std_a == 0.0 or std_b == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(magnitude_a.ravel(), magnitude_b.ravel())[0, 1])
    significant = (magnitude_a > tolerance) | (magnitude_b > tolerance)
    disagreement_rate = (
        float(np.mean((np.sign(signed_a) != np.sign(signed_b)) & significant))
        if significant.any()
        else 0.0
    )
    complementarity = float(np.mean(np.abs(magnitude_a - magnitude_b) > tolerance))
    return {
        "error_correlation": correlation,
        "disagreement_rate": disagreement_rate,
        "complementarity": complementarity,
    }


def error_diversity(
    signed_errors: Mapping[str, np.ndarray], *, tolerance: float = 1e-6
) -> dict[str, dict[str, float]]:
    """Pairwise error-diversity metrics over signed per-pixel error maps.

    ``signed_errors`` maps a model name to its signed error map
    ``output - target`` on the same grid. Diversity is never accuracy.
    """
    names = sorted(signed_errors)
    result: dict[str, dict[str, float]] = {}
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            result[f"{name_a}-vs-{name_b}"] = _pairwise_diversity(
                signed_errors[name_a], signed_errors[name_b], tolerance=tolerance
            )
    return result


def add_if_diverse(
    candidate: str,
    signed_errors: Mapping[str, np.ndarray],
    threshold: float,
    *,
    tolerance: float = 1e-6,
) -> bool:
    """True when the candidate's mean disagreement rate vs the set meets ``threshold``.

    The diversity guard from stability-v1 section 1.3: a model may join a
    comparison set only when its measured error diversity is sufficient.
    """
    others = {name: error for name, error in signed_errors.items() if name != candidate}
    if not others:
        return True
    rates: list[float] = []
    for error in others.values():
        pair = _pairwise_diversity(signed_errors[candidate], error, tolerance=tolerance)
        rates.append(pair["disagreement_rate"])
    return float(np.mean(rates)) >= threshold
