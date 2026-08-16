"""Integration tests for Unified Inference Pipeline (Phase 11)."""

from __future__ import annotations

import numpy as np
import pytest

from evidence_net.inference.package import generate_review_package
from evidence_net.inference.pipeline import UnifiedInferencePipeline
from evidence_net.inference.provenance import build_provenance_record
from evidence_net.models.base import BaseReconstruction
from evidence_net.models.proposal import DetailProposer


@pytest.fixture
def sample_tensors():
    rng = np.random.default_rng(42)
    inp = rng.uniform(0.0, 1.0, size=(1, 64, 64)).astype(np.float32)
    tgt = rng.uniform(0.0, 1.0, size=(1, 64, 64)).astype(np.float32)
    return inp, tgt


def test_pipeline_execution(tmp_path, sample_tensors):
    inp, tgt = sample_tensors
    base_model = BaseReconstruction()
    proposal_model = DetailProposer()
    provenance = build_provenance_record(dataset_manifest_hash="sha256:test1234")

    pipeline = UnifiedInferencePipeline(
        base_model=base_model,
        proposal_model=proposal_model,
        provenance=provenance,
    )

    res = pipeline.run_sample(
        input_tensor=inp,
        target_tensor=tgt,
        runs_dir=tmp_path / "runs",
        run_id="exp-test-001",
    )

    assert res.run_id == "exp-test-001"
    assert (res.run_dir / "artifacts" / "input.npy").is_file()
    assert (res.run_dir / "artifacts" / "base.npy").is_file()
    assert (res.run_dir / "artifacts" / "proposal.npy").is_file()
    assert (res.run_dir / "artifacts" / "candidate.npy").is_file()
    assert (res.run_dir / "artifacts" / "final.npy").is_file()
    assert (res.run_dir / "artifacts" / "optional_fields_manifest.json").is_file()

    # Check metrics computed
    assert "base" in res.metrics
    assert "candidate" in res.metrics
    assert "final" in res.metrics


def test_optional_fields_handling(tmp_path, sample_tensors):
    inp, _ = sample_tensors
    pipeline = UnifiedInferencePipeline()

    benefit = np.ones((1, 64, 64), dtype=np.float32)
    res = pipeline.run_sample(
        input_tensor=inp,
        runs_dir=tmp_path / "runs",
        run_id="exp-test-opts",
        optional_tensors={"proposal_benefit.npy": benefit},
    )

    assert "proposal_benefit.npy" in res.artifact_metadata
    assert (res.run_dir / "artifacts" / "proposal_benefit.npy").is_file()


def test_review_package_generation(tmp_path):
    prov = build_provenance_record()
    art_meta = {
        "input.npy": {
            "dtype": "float32",
            "shape": [1, 64, 64],
            "range": [0.0, 1.0],
            "hash": "sha256:0",
        }
    }
    metrics = {"base": {"mae": {"value": 0.05, "unit": "pixel"}}}

    md_path, json_path = generate_review_package(
        output_dir=tmp_path,
        run_id="run-review-01",
        provenance=prov,
        metrics=metrics,
        artifact_metadata=art_meta,
    )

    assert md_path.is_file()
    assert json_path.is_file()
    assert "EVIDENCE-Net Technical Review Package" in md_path.read_text(encoding="utf-8")
