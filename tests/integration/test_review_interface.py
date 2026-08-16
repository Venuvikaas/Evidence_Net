"""Integration test verifying Review Interface contract endpoints (Phase 13 & 14)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from evidence_net.api.app import create_app


def test_review_interface_endpoints(tmp_path):
    app = create_app(db_path=tmp_path / "test_review_interface.db")
    client = TestClient(app)

    # 1. Test version response includes expected component keys
    ver_res = client.get("/api/v1/version")
    assert ver_res.status_code == 200
    versions = ver_res.json()["versions"]
    assert "pipeline_version" in versions
    assert "base_model_version" in versions
    assert "proposal_model_version" in versions

    # 2. Test metadata response
    meta_res = client.get("/api/v1/metadata")
    assert meta_res.status_code == 200
    meta_data = meta_res.json()
    assert meta_data["service"] == "EVIDENCE-Net API"
    assert meta_data["api_contract_version"] == "v1"

    # 3. Test recording review event (Phase 14 human interpretation workflow)
    rev_res = client.post(
        "/api/v1/review/events",
        json={
            "run_id": "run-test-123",
            "reviewer_id": "expert-01",
            "action_type": "accept_proposal",
            "decision_reason": "Verified positive proposal continuity gain",
        },
    )
    assert rev_res.status_code == 200
    assert rev_res.json()["status"] == "recorded"
