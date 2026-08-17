"""ONNX vs PyTorch decision-parity test (Phase 15).

Validates the Phase 15 export gate on the promoted 128x128 -> 256x256 grid:

- **tensor parity**: Base output ``b`` and Proposal output ``d`` match
  between PyTorch and ONNX Runtime within tolerance;
- **spatial parity**: output grids are 256x256 (2x up-scale contract);
- **ranking parity**: the residual-magnitude benefit score map computed from
  ONNX tensors matches the one from PyTorch tensors;
- **action parity**: the promoted decision gate map (default-accept with
  unresolved abstention) computed from ONNX tensors matches PyTorch;
- **abstention parity**: the unresolved-region mask matches as well;
- **calibration**: recorded honestly as ``not-defined`` — the service does
  not re-fit a calibration artifact at inference time (calibration-v1 is a
  recorded version, not a served tensor).

The test uses the frozen promoted checkpoints when they are present locally;
without them (CI) it falls back to default-architecture models so the parity
gate still runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from deploy.export_onnx import (
    BASE_CHECKPOINT,
    PROPOSAL_CHECKPOINT,
    export_base_model,
    export_proposal_model,
)

from evidence_net.api.diagnostics import compute_run_diagnostics
from evidence_net.models.base import BaseReconstruction
from evidence_net.models.proposal import DetailProposer

INPUT_GRID = 128
OUTPUT_GRID = 256
RTOL = 1e-5
ATOL = 1e-5

onnxruntime = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")


def _input_grid(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(1, 1, INPUT_GRID, INPUT_GRID)).astype(np.float32)


def _load_models():
    """Promoted checkpoints when present, default architectures otherwise."""
    if BASE_CHECKPOINT.is_file():
        from deploy.export_onnx import load_checkpoint

        base = load_checkpoint(BASE_CHECKPOINT, BaseReconstruction)
    else:
        base = BaseReconstruction().eval()
    if PROPOSAL_CHECKPOINT.is_file():
        from deploy.export_onnx import load_checkpoint

        proposer = load_checkpoint(PROPOSAL_CHECKPOINT, DetailProposer)
    else:
        proposer = DetailProposer().eval()
    return base, proposer


def test_base_onnx_parity(tmp_path: Path) -> None:
    """Tensor + spatial parity of the Base reconstruction."""
    onnx_path = tmp_path / "base.onnx"
    export_base_model(onnx_path)
    assert onnx_path.is_file()

    base, _ = _load_models()
    y = _input_grid()
    with torch.no_grad():
        py_output = base(torch.from_numpy(y)).numpy()

    session = onnxruntime.InferenceSession(str(onnx_path))
    onnx_output = session.run(None, {"input": y})[0]

    # Spatial parity: frozen 2x up-scale contract.
    assert onnx_output.shape == (1, 1, OUTPUT_GRID, OUTPUT_GRID)
    assert py_output.shape == onnx_output.shape
    # Tensor parity within tolerance.
    np.testing.assert_allclose(py_output, onnx_output, rtol=RTOL, atol=ATOL)


def test_proposal_onnx_parity(tmp_path: Path) -> None:
    """Tensor + spatial parity of the bounded detail proposal."""
    onnx_path = tmp_path / "proposal.onnx"
    export_proposal_model(onnx_path)
    assert onnx_path.is_file()

    base, proposer = _load_models()
    y = _input_grid()
    with torch.no_grad():
        b_torch = base(torch.from_numpy(y))
        py_output = proposer(torch.from_numpy(y), b_torch).numpy()

    session = onnxruntime.InferenceSession(str(onnx_path))
    onnx_output = session.run(None, {"y": y, "b": b_torch.numpy()})[0]

    assert onnx_output.shape == (1, 1, OUTPUT_GRID, OUTPUT_GRID)
    assert py_output.shape == onnx_output.shape
    np.testing.assert_allclose(py_output, onnx_output, rtol=RTOL, atol=ATOL)


def test_full_decision_parity(tmp_path: Path) -> None:
    """Ranking, action, and abstention parity of the full promoted pipeline."""
    base_path = tmp_path / "base.onnx"
    proposal_path = tmp_path / "proposal.onnx"
    export_base_model(base_path)
    export_proposal_model(proposal_path)

    base, proposer = _load_models()
    y = _input_grid()
    y_t = torch.from_numpy(y)

    # --- PyTorch side of the promoted pipeline ---------------------------
    with torch.no_grad():
        b_t = base(y_t)
        d_t = proposer(y_t, b_t)
    c_t = torch.clamp(b_t + d_t, 0.0, 1.0)  # ungated candidate
    final_t = c_t  # promoted policy default-accepts the candidate (ADR-010)

    # --- ONNX Runtime side ------------------------------------------------
    b_s = onnxruntime.InferenceSession(str(base_path))
    d_s = onnxruntime.InferenceSession(str(proposal_path))
    b_np = b_s.run(None, {"input": y})[0]
    d_np = d_s.run(None, {"y": y, "b": b_np})[0]
    c_np = np.clip(b_np + d_np, 0.0, 1.0)
    final_np = c_np

    # Tensor parity for every frozen output of the pipeline.
    for name, py, onx in [
        ("base", b_t.numpy(), b_np),
        ("proposal", d_t.numpy(), d_np),
        ("candidate", c_t.numpy(), c_np),
        ("final", final_t.numpy(), final_np),
    ]:
        np.testing.assert_allclose(py, onx, rtol=RTOL, atol=ATOL, err_msg=f"{name} parity")

    # Ranking / action / abstention parity: the served diagnostics must be
    # identical whether computed from PyTorch or ONNX tensors.
    diag_py = compute_run_diagnostics(
        input_grid=y[0],
        base_grid=b_t.numpy()[0],
        proposal_grid=d_t.numpy()[0],
    )
    diag_onx = compute_run_diagnostics(
        input_grid=y[0],
        base_grid=b_np[0],
        proposal_grid=d_np[0],
    )
    assert set(diag_py) == set(diag_onx), "diagnostic artifact sets must match"
    for key in diag_py:
        np.testing.assert_allclose(
            diag_py[key], diag_onx[key], rtol=RTOL, atol=ATOL, err_msg=f"{key} parity"
        )

    # Spatial parity of the diagnostic layers.
    assert diag_py["proposal_benefit.npy"].shape == (OUTPUT_GRID, OUTPUT_GRID)
    assert diag_py["decision_map.npy"].shape == (OUTPUT_GRID, OUTPUT_GRID)
    assert diag_py["unresolved.npy"].shape == (OUTPUT_GRID, OUTPUT_GRID)

    # Calibration: honestly not-defined. The promoted pipeline records
    # calibration-v1 as a version but never serves a calibration tensor, so
    # there is nothing to compare — this assertion documents that decision.
    assert "calibration" not in {k.rsplit(".", 1)[0] for k in diag_py}
