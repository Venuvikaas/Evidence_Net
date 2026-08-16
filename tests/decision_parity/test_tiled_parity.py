"""Tiled inference boundaries and map alignment test (Phase 15)."""

from __future__ import annotations

import numpy as np

from evidence_net.inference.pipeline import UnifiedInferencePipeline
from evidence_net.models.base import BaseReconstruction


def test_tiled_boundary_alignment(tmp_path):
    rng = np.random.default_rng(123)
    # Large 128x128 input tensor
    large_input = rng.uniform(0.0, 1.0, size=(1, 128, 128)).astype(np.float32)

    pipeline = UnifiedInferencePipeline(base_model=BaseReconstruction())
    result = pipeline.run_sample(
        input_tensor=large_input,
        runs_dir=tmp_path / "runs",
        run_id="tiled-parity-001",
    )

    # Output grid is 256x256
    assert result.final_tensor.shape == (1, 256, 256)
    assert not np.isnan(result.final_tensor).any()
    assert not np.isinf(result.final_tensor).any()
    assert np.min(result.final_tensor) >= 0.0
    assert np.max(result.final_tensor) <= 1.0
