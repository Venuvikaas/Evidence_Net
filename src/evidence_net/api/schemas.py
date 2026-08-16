"""Pydantic API request/response schemas for EVIDENCE-Net (Phase 12).

Defines validation models for health, metadata, version, restoration inference,
comparison, stress tests, and upload endpoints matching docs/contracts/.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    timestamp: str


class VersionInfo(BaseModel):
    component: str
    version: str
    description: str = ""


class VersionResponse(BaseModel):
    versions: dict[str, str]


class ErrorPayload(BaseModel):
    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class RestorationRequest(BaseModel):
    input_values: list[list[float]] | list[float] | None = None
    shape: list[int] | None = None
    has_target: bool = False
    target_values: list[list[float]] | list[float] | None = None
    optional_fields: dict[str, Any] = Field(default_factory=dict)


class RestorationResponse(BaseModel):
    run_id: str
    status: str = "completed"
    provenance: dict[str, str]
    metrics: dict[str, Any]
    artifacts: dict[str, dict[str, Any]]
    run_dir: str


class ArtifactInfo(BaseModel):
    name: str
    dtype: str
    shape: list[int]
    range: list[float]
    hash: str
    is_optional: bool = False


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    created_at: str
    artifacts: list[ArtifactInfo]


class ComparisonRequest(BaseModel):
    run_ids: list[str] = Field(min_length=1)


class ComparisonResponse(BaseModel):
    comparison_id: str
    run_ids: list[str]
    metrics_summary: dict[str, Any]


class StressTestRequest(BaseModel):
    perturbation_type: str = Field(default="gaussian_noise", description="Noise or shift type")
    severity_levels: list[float] = Field(default_factory=lambda: [0.01, 0.05, 0.1])


class StressTestResponse(BaseModel):
    test_id: str
    perturbation_type: str
    results: list[dict[str, Any]]


class ReviewEventRequest(BaseModel):
    run_id: str
    reviewer_id: str = "expert-reviewer"
    action_type: str
    decision_reason: str = ""
    event_details: dict[str, Any] = Field(default_factory=dict)


class ReviewEventResponse(BaseModel):
    event_id: int
    run_id: str
    status: str = "recorded"
