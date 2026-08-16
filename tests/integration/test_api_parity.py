"""Integration test verifying CLI vs FastAPI endpoint equivalence (Phase 12)."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from evidence_net.api.app import create_app
from evidence_net.inference.pipeline import UnifiedInferencePipeline


def test_api_cli_restoration_parity(tmp_path):
    app = create_app(db_path=tmp_path / "test_metadata.db")
    client = TestClient(app)

    rng = np.random.default_rng(42)
    inp = rng.uniform(0.0, 1.0, size=(1, 16, 16)).astype(np.float32)

    # 1. CLI Pipeline execution
    cli_pipeline = UnifiedInferencePipeline()
    cli_res = cli_pipeline.run_sample(
        input_tensor=inp,
        runs_dir=tmp_path / "runs",
        run_id="cli-run-001",
    )

    # 2. API endpoint execution
    response = client.post(
        "/api/v1/restoration",
        json={
            "input_values": inp.flatten().tolist(),
            "shape": list(inp.shape),
        },
    )

    assert response.status_code == 200
    api_data = response.json()

    assert api_data["status"] == "completed"
    assert "provenance" in api_data
    assert "artifacts" in api_data
    assert api_data["provenance"]["pipeline_version"] == cli_res.provenance.pipeline_version


def test_api_health_and_version(tmp_path):
    app = create_app(db_path=tmp_path / "test_metadata.db")
    client = TestClient(app)

    res_health = client.get("/api/v1/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"

    res_ver = client.get("/api/v1/version")
    assert res_ver.status_code == 200
    assert "versions" in res_ver.json()


def test_api_upload_validation(tmp_path):
    app = create_app(db_path=tmp_path / "test_metadata.db")
    client = TestClient(app)

    # Reject disallowed file extension
    res_bad = client.post(
        "/api/v1/upload",
        files={"file": ("test.exe", b"binary content", "application/octet-stream")},
    )
    assert res_bad.status_code == 400
    assert res_bad.json()["error_code"] == "INVALID_FILE_EXTENSION"

    # Accept valid .npy file
    res_good = client.post(
        "/api/v1/upload",
        files={"file": ("test.npy", b"dummy npy bytes", "application/octet-stream")},
    )
    assert res_good.status_code == 200
    assert res_good.json()["status"] == "success"
