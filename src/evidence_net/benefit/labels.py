"""Deterministic proposal-benefit labels (Phase 5, support-definition-v1).

The benefit event is a patch-level strict comparison, identical to the
Phase 4 oracle patch rule (``docs/proposal-contract.md``): a 16x16 patch on
the 256x256 output grid is **beneficial** when the patch MAE of the ungated
candidate is strictly lower than the patch MAE of the frozen Base output.

    beneficial(r)  <=>  MAE(x_r, c_r) < MAE(x_r, b_r)

Ties and increases are not beneficial. Labels are a pure, deterministic
function of ``(base, proposal, target)`` on the output grid — versioned as
``labels-v1`` — and are written as versioned JSON artifacts so the predictor
and the decision policy can consume an immutable event definition.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evidence_net.evaluation.metrics import mae

LABELS_VERSION = "labels-v1"
PATCH_SIZE = 16  # must match evaluation.oracle.PATCH_SIZE

# Patch-grid shape on the official 256x256 output grid.
OUTPUT_GRID = 256
PATCH_GRID = OUTPUT_GRID // PATCH_SIZE  # 16x16 patch grid


class BenefitLabelsError(ValueError):
    """Raised when benefit labels cannot be computed from the inputs."""


def _as_float64(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)


def patch_benefit_labels(
    base: np.ndarray,
    proposal: np.ndarray,
    target: np.ndarray,
    *,
    margin: float = 0.0,
) -> np.ndarray:
    """Binary per-patch benefit labels on the patch grid (16x16).

    Returns a ``(16, 16)`` uint8 array: 1 where the ungated candidate patch
    improves on the Base patch by **more than ``margin``** MAE, else 0:

        beneficial(r)  <=>  MAE(x, c) + margin < MAE(x, b)

    ``labels-v1`` uses ``margin = 0`` (strict). ``labels-v2`` declares a
    meaningful margin (e.g. 0.005): Gate 4 evidence (EXP-009, ADR-016)
    showed the strict event is dominated by sub-margin noise (mean delta
    0.0026, AUC at chance), while the meaningful-benefit event is
    predictable (AUC 0.91-0.99 for simple features). Requires a 256x256
    output grid (partial edge patches are not labeled).
    """
    if margin < 0.0:
        raise BenefitLabelsError(f"margin must be >= 0, got {margin}")
    b = _as_float64(base)
    d = _as_float64(proposal)
    x = _as_float64(target)
    if b.shape != x.shape or d.shape != x.shape:
        raise BenefitLabelsError(
            f"base/proposal/target must share the output grid, got {b.shape}, {d.shape}, {x.shape}"
        )
    height, width = x.shape
    if height != OUTPUT_GRID or width != OUTPUT_GRID:
        raise BenefitLabelsError(
            f"benefit labels are defined on the {OUTPUT_GRID}x{OUTPUT_GRID} "
            f"output grid, got {x.shape}"
        )
    candidate = np.clip(b + d, 0.0, 1.0)
    labels = np.zeros((PATCH_GRID, PATCH_GRID), dtype=np.uint8)
    for row in range(PATCH_GRID):
        for col in range(PATCH_GRID):
            rows = slice(row * PATCH_SIZE, (row + 1) * PATCH_SIZE)
            cols = slice(col * PATCH_SIZE, (col + 1) * PATCH_SIZE)
            labels[row, col] = int(
                mae(x[rows, cols], candidate[rows, cols]) + margin
                < mae(x[rows, cols], b[rows, cols])
            )
    return labels


def benefit_fraction(labels: np.ndarray) -> float:
    """Fraction of labeled patches that are beneficial."""
    flat = np.asarray(labels, dtype=np.float64)
    return float(flat.mean()) if flat.size else 0.0


@dataclass(frozen=True)
class LabeledSample:
    """Deterministic benefit labels for one output-grid sample."""

    sample_id: str
    labels: np.ndarray  # (16, 16) uint8
    benefit_fraction: float
    n_patches: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "labels_version": LABELS_VERSION,
            "patch_grid": [int(self.labels.shape[0]), int(self.labels.shape[1])],
            "benefit_fraction": self.benefit_fraction,
            "n_patches": self.n_patches,
            "labels": self.labels.astype(int).tolist(),
        }


def label_samples(
    sample_ids: Sequence[str],
    bases: Sequence[np.ndarray],
    proposals: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
) -> list[LabeledSample]:
    """Deterministic labels over aligned sample sets (all sequences aligned)."""
    if not (len(sample_ids) == len(bases) == len(proposals) == len(targets)):
        raise BenefitLabelsError("sample ids, bases, proposals, targets must be aligned")
    return [
        _label_one(sample_id, base, proposal, target)
        for sample_id, base, proposal, target in zip(
            sample_ids, bases, proposals, targets, strict=True
        )
    ]


def _label_one(
    sample_id: str, base: np.ndarray, proposal: np.ndarray, target: np.ndarray
) -> LabeledSample:
    labels = patch_benefit_labels(base, proposal, target)
    return LabeledSample(
        sample_id=sample_id,
        labels=labels,
        benefit_fraction=benefit_fraction(labels),
        n_patches=int(labels.size),
    )


def write_label_manifest(path: Path, samples: Sequence[LabeledSample]) -> Path:
    """Write the versioned label artifact (``benefit-labels-v1.json``)."""
    payload = {
        "schema": "benefit-labels-v1",
        "labels_version": LABELS_VERSION,
        "event": "patch MAE(candidate) < patch MAE(base), strict, 16x16 grid",
        "samples": [sample.as_dict() for sample in samples],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
