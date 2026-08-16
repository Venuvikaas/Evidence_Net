"""Candidate manipulation suite (Phase 10, structural-risk-v1 section 2).

Seven frozen manipulations of the **restored output** that probe structural
errors: false-line insertion, real-line deletion, edge shift, merge, split,
false periodicity, and defect point. Geometry and amplitudes come from the
frozen hidden stress definitions (`data/stress/hidden-stress-v1.json`).
All manipulations are labeled ``candidate`` — a separate threat model from
ambiguity, acquisition, natural failures, and downstream evidence
(no hallucination-resistance claim from this suite alone).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np

from evidence_net.stress_tests.hidden_stress import HiddenStressError, stress_params


class StructuralError(ValueError):
    """Raised for invalid candidate manipulations or parameters."""


def _as_2d(candidate: np.ndarray) -> np.ndarray:
    array = np.asarray(candidate)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise StructuralError(f"candidate must be (H, W), got shape {array.shape}")
    return array.astype(np.float64, copy=True)


def _clip(array: np.ndarray) -> np.ndarray:
    return np.clip(array, 0.0, 1.0)


def _bright_column(image: np.ndarray) -> int:
    """Column index with the highest total brightness (dominant vertical line)."""
    return int(np.argmax(image.sum(axis=0)))


def _bright_columns(image: np.ndarray, n: int = 2, min_sep: int = 4) -> list[int]:
    """Top-n bright columns (relative to the median), kept ``min_sep`` apart.

    Only columns whose total brightness is at least twice the median column
    sum qualify, so a fall-through can never pick a background column.
    """
    sums = image.sum(axis=0)
    median = float(np.median(sums))
    threshold = max(2.0 * median, 1e-9)
    candidates = [int(column) for column in np.argsort(sums)[::-1] if sums[column] >= threshold]
    picked: list[int] = []
    for column in candidates:
        if all(abs(column - p) >= min_sep for p in picked):
            picked.append(column)
        if len(picked) >= n:
            break
    return picked


class CandidateManipulation(ABC):
    """A frozen structural manipulation of a restored output."""

    name: str
    threat = "candidate"

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    @abstractmethod
    def apply(self, candidate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return a modified copy of the candidate (values clipped to [0, 1])."""

    @abstractmethod
    def effect(self) -> str:
        """Human description of the intended local structural effect."""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "threat": self.threat, "effect": self.effect()}


class FalseLineInsertion(CandidateManipulation):
    """Insert a thin bright line that may not exist in the target."""

    name = "false-line"

    def apply(self, candidate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        image = _as_2d(candidate)
        height, width = image.shape
        line_width = int(self.params.get("line_width", 1))
        length = min(int(self.params.get("min_line_length", 16)), height)
        start_row = rng.integers(0, height - length + 1)
        column = rng.integers(0, width)
        value = float(np.clip(max(np.percentile(image, 95), 0.9), 0.0, 1.0))
        image[start_row : start_row + length, column : column + line_width] = value
        return _clip(image)

    def effect(self) -> str:
        return "inserts a thin line that may not exist in the target (hallucination probe)"


class RealLineDeletion(CandidateManipulation):
    """Remove the dominant bright line (real structure loss)."""

    name = "line-deletion"

    def apply(self, candidate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        image = _as_2d(candidate)
        line_width = int(self.params.get("line_width", 1))
        column = _bright_column(image)
        background = float(np.median(image))
        image[:, column : column + line_width] = background
        return _clip(image)

    def effect(self) -> str:
        return "removes the dominant bright line (real structure loss)"


class EdgeShift(CandidateManipulation):
    """Move the dominant vertical edge by a frozen pixel amount."""

    name = "edge-shift"

    def apply(self, candidate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        image = _as_2d(candidate)
        height, width = image.shape
        shift = int(self.params.get("edge_shift_px", 2))
        gradient = np.abs(np.diff(image, axis=1))
        edge_column = int(np.argmax(gradient.sum(axis=0)))
        if shift <= 0 or edge_column >= width:
            return _clip(image)
        right = image[:, edge_column + 1 :]
        image[:, edge_column + 1 :] = np.roll(right, shift, axis=1)
        return _clip(image)

    def effect(self) -> str:
        return f"shifts the dominant edge by {self.params.get('edge_shift_px', 2)} px"


class MergeLines(CandidateManipulation):
    """Join two nearby bright structures into one."""

    name = "merge"

    def apply(self, candidate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        image = _as_2d(candidate)
        gap = int(self.params.get("merge_gap", 4))
        columns = _bright_columns(image, 2, min_sep=gap)
        if len(columns) < 2:
            # No pair to merge: fabricate one from the brightest column.
            column = _bright_column(image)
            width = image.shape[1]
            columns = [max(0, min(column, width - gap - 1)), column]
        columns.sort()
        left, right = columns
        fill = image[:, left]
        image[:, left : right + 1] = fill[:, None]
        return _clip(image)

    def effect(self) -> str:
        return "merges two nearby structures into one"


class SplitLine(CandidateManipulation):
    """Break one bright structure into two components."""

    name = "split"

    def apply(self, candidate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        image = _as_2d(candidate)
        height, width = image.shape
        gap = int(self.params.get("split_gap", 4))
        column = _bright_column(image)
        background = float(np.median(image))
        start = max(0, height // 2 - gap // 2)
        image[start : start + gap, column] = background
        return _clip(image)

    def effect(self) -> str:
        return "splits one structure into two components"


class FalsePeriodicity(CandidateManipulation):
    """Add a periodic stripe pattern (false periodicity)."""

    name = "false-periodicity"

    def apply(self, candidate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        image = _as_2d(candidate)
        period = int(self.params.get("period", 4))
        amplitude = float(self.params.get("stripe_amplitude", 0.05))
        columns = np.arange(image.shape[1])
        pattern = np.sin(2.0 * np.pi * columns / period) * amplitude
        return _clip(image + pattern[None, :])

    def effect(self) -> str:
        return "adds a periodic pattern that may be hallucinated"


class DefectPoint(CandidateManipulation):
    """Add or remove an isolated defect-like point."""

    name = "defect-point"

    def apply(self, candidate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        image = _as_2d(candidate)
        height, width = image.shape
        value = float(self.params.get("point_value", 0.9))
        row = int(rng.integers(0, height))
        column = int(rng.integers(0, width))
        image[row, column] = 0.0 if image[row, column] >= value else value
        return _clip(image)

    def effect(self) -> str:
        return "adds or removes an isolated defect-like point"


MANIPULATION_TYPES: dict[str, type[CandidateManipulation]] = {
    "false-line": FalseLineInsertion,
    "line-deletion": RealLineDeletion,
    "edge-shift": EdgeShift,
    "merge": MergeLines,
    "split": SplitLine,
    "false-periodicity": FalsePeriodicity,
    "defect-point": DefectPoint,
}


def build_candidate_suite(
    params: Mapping[str, Any] | None = None,
    names: tuple[str, ...] | None = None,
) -> list[CandidateManipulation]:
    """Build the frozen candidate suite from hidden-stress perturbation params.

    ``params`` should be the ``perturbation`` section of the hidden stress
    definitions; when omitted they are loaded from the frozen file.
    """
    if params is None:
        try:
            loaded = stress_params()
        except HiddenStressError as exc:
            raise StructuralError(f"cannot build candidate suite: {exc}") from exc
        perturbation = loaded["perturbation"]
        assert isinstance(perturbation, dict)
        params = perturbation
    selected = names or tuple(MANIPULATION_TYPES)
    unknown = set(selected) - set(MANIPULATION_TYPES)
    if unknown:
        raise StructuralError(f"unknown manipulations: {sorted(unknown)}")
    return [MANIPULATION_TYPES[name](params) for name in selected]
