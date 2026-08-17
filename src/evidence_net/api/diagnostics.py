"""Per-run diagnostics for the review API (Phase 11/13 integration).

The review UI must display only backend-computed scientific values. This
module computes the run-level diagnostics that are cheap, deterministic
functions of the run's own tensors, on the frozen 256x256 output grid:

- ``proposal_benefit.npy``: benefit **ranking score** map (per-patch,
  upsampled to the output grid) from the ``residual-magnitude`` predictor
  on the ``labels-v2`` margin event (ADR-016; group-bootstrapped AUC 0.889
  [0.870, 0.907]). It is a ranking signal only — it never gates outputs.
- ``decision_map.npy``: gate map of the promoted ``decision-policy-v1``
  (ADR-010): default-accept everywhere (gate 1) with **unresolved** patches
  gated to 0 (fall back to the frozen Base). Reported as a layer; the
  released final output remains the default-accept candidate.
- ``unresolved.npy``: the orthogonal unresolved mask from patch edge density
  (policy config, EXP-004 periodic-region evidence).

The remaining optional diagnostics — measurement consistency, model
stability, distribution familiarity — are **not** computed by this service:
they require operators/ensembles/a reference population that belong to the
dedicated measurement scripts. The UI renders them as ``not-defined`` with
their exact legends.
"""

from __future__ import annotations

import numpy as np
import torch

from evidence_net.benefit.predictors import ResidualMagnitudeBaseline
from evidence_net.decision.policy import PolicyConfig
from evidence_net.evaluation.metrics import edge_magnitude

OUTPUT_GRID = 256
PATCH_SIZE = 16
PATCH_GRID = OUTPUT_GRID // PATCH_SIZE

BENEFIT_PREDICTOR = "residual-magnitude"
BENEFIT_EVENT = "labels-v2 margin event: MAE(candidate) + 0.005 < MAE(base), per 16x16 patch"
DECISION_POLICY = "decision-policy-v1 (default-accept + unresolved abstention)"


def _to_2d(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    return arr[0] if arr.ndim == 3 else arr


def _to_output_grid(array: np.ndarray) -> np.ndarray:
    """Bilinear-upsample any 2D grid to the 256x256 output grid."""
    arr = _to_2d(array)
    if arr.shape == (OUTPUT_GRID, OUTPUT_GRID):
        return arr
    tensor = torch.from_numpy(arr)[None, None]
    up = torch.nn.functional.interpolate(
        tensor, size=(OUTPUT_GRID, OUTPUT_GRID), mode="bilinear", align_corners=False
    )
    return up[0, 0].numpy()


def _patch_edge_density(input_grid: np.ndarray) -> np.ndarray:
    """Mean normalized edge magnitude per 16x16 patch (input at output grid)."""
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


def _upsample_patch_map(patch_map: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(patch_map, PATCH_SIZE, axis=0), PATCH_SIZE, axis=1)


def compute_run_diagnostics(
    input_grid: np.ndarray,
    base_grid: np.ndarray,
    proposal_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Optional artifacts for one run, or ``{}`` when the grid contract differs.

    All three diagnostics are contract-defined on the 256x256 output grid.
    Runs on other grids (e.g. non-128x128 uploads) return no diagnostics and
    the corresponding artifacts stay ``not-defined``.
    """
    inp = _to_output_grid(input_grid)
    base = _to_2d(base_grid)
    proposal = _to_2d(proposal_grid)
    if not (base.shape == proposal.shape == (OUTPUT_GRID, OUTPUT_GRID)):
        return {}

    # Benefit ranking score (16x16) -> output grid (nearest, per-patch blocks).
    score = ResidualMagnitudeBaseline().score(inp, base, proposal)
    benefit = _upsample_patch_map(score).astype(np.float32)

    # Promoted policy: default-accept + unresolved abstention (ADR-010).
    config = PolicyConfig()
    density = _patch_edge_density(inp)
    unresolved = density >= config.unresolved_edge_density
    gates = np.where(unresolved, 0.0, 1.0)

    return {
        "proposal_benefit.npy": benefit,
        "decision_map.npy": _upsample_patch_map(gates).astype(np.float32),
        "unresolved.npy": _upsample_patch_map(unresolved.astype(np.float32)).astype(np.float32),
    }
