"""Full-loop regression test for EVIDENCE-Net sample inference to review package."""

from __future__ import annotations

import json

import numpy as np

from evidence_net.inference.package import generate_review_package
from evidence_net.inference.pipeline import UnifiedInferencePipeline
from evidence_net.inference.provenance import build_provenance_record


def test_full_loop_regression(tmp_path):
    rng = np.random.default_rng(12345)
    inp = rng.uniform(0.0, 1.0, size=(1, 32, 32)).astype(np.float32)
    tgt = rng.uniform(0.0, 1.0, size=(1, 32, 32)).astype(np.float32)

    prov = build_provenance_record(
        dataset_manifest_hash="sha256:1111222233334444",
        base_model_version="base-v1-golden",
        proposal_model_version="proposal-v1-golden",
    )

    pipeline = UnifiedInferencePipeline(provenance=prov)
    result = pipeline.run_sample(
        input_tensor=inp,
        target_tensor=tgt,
        runs_dir=tmp_path / "runs",
        run_id="golden-loop-001",
    )

    md_path, json_path = generate_review_package(
        output_dir=result.run_dir,
        run_id=result.run_id,
        provenance=result.provenance,
        metrics=result.metrics,
        artifact_metadata=result.artifact_metadata,
    )

    # Check files produced in run bundle
    assert (result.run_dir / "config.yaml").is_file()
    assert (result.run_dir / "manifest.json").is_file()
    assert (result.run_dir / "metrics.json").is_file()
    assert (result.run_dir / "environment.txt").is_file()
    assert (result.run_dir / "summary.md").is_file()
    assert md_path.is_file()
    assert json_path.is_file()

    # Validate JSON payload structure
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "review-package-v1"
    assert payload["run_id"] == "golden-loop-001"
    assert payload["provenance"]["dataset_manifest_hash"] == "sha256:1111222233334444"
    assert "input.npy" in payload["artifacts"]
    assert "base.npy" in payload["artifacts"]
    assert "proposal.npy" in payload["artifacts"]
    assert "candidate.npy" in payload["artifacts"]
    assert "final.npy" in payload["artifacts"]
