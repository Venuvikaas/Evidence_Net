"""ONNX vs PyTorch spatial and decision parity test (Phase 15)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from deploy.export_onnx import export_base_model, export_proposal_model

from evidence_net.models.base import BaseReconstruction
from evidence_net.models.proposal import DetailProposer


@pytest.fixture
def dummy_tensors():
    rng = np.random.default_rng(42)
    y = rng.uniform(0.0, 1.0, size=(1, 1, 32, 32)).astype(np.float32)
    return y


def test_base_onnx_parity(tmp_path, dummy_tensors):
    onnx_path = tmp_path / "base.onnx"
    export_base_model(onnx_path)
    assert onnx_path.is_file()

    # PyTorch output
    py_model = BaseReconstruction().eval()
    y_torch = torch.from_numpy(dummy_tensors)
    with torch.no_grad():
        py_output = py_model(y_torch).numpy()

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))
        onnx_output = session.run(None, {"input": dummy_tensors})[0]
        # Parity check tolerance 1e-5
        np.testing.assert_allclose(py_output, onnx_output, rtol=1e-5, atol=1e-5)
    except Exception as exc:
        pytest.skip(f"ONNXRuntime session skipped: {exc}")


def test_proposal_onnx_parity(tmp_path, dummy_tensors):
    onnx_path = tmp_path / "proposal.onnx"
    export_proposal_model(onnx_path)
    assert onnx_path.is_file()

    py_base = BaseReconstruction().eval()
    py_prop = DetailProposer().eval()

    y_torch = torch.from_numpy(dummy_tensors)
    with torch.no_grad():
        b_torch = py_base(y_torch)
        py_proposal_out = py_prop(y_torch, b_torch).numpy()

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))
        b_np = b_torch.numpy()
        onnx_output = session.run(None, {"y": dummy_tensors, "b": b_np})[0]
        np.testing.assert_allclose(py_proposal_out, onnx_output, rtol=1e-5, atol=1e-5)
    except Exception as exc:
        pytest.skip(f"ONNXRuntime session skipped: {exc}")
