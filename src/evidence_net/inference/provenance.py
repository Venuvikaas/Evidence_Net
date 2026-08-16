"""Provenance tracking for EVIDENCE-Net inference runs (Phase 11).

Records explicit semantic version identifiers for dataset manifest, base model,
detail proposal model, support definition, calibration, forward model, and
decision policy. Any unpromoted or unavailable version is recorded as "not-defined"
per contract artifacts-v1 and error-and-optional-fields-v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProvenanceRecord:
    """Immutable provenance record for a run or artifact."""

    dataset_manifest_hash: str = "not-defined"
    base_model_version: str = "base-model-v1"
    proposal_model_version: str = "proposal-model-v1"
    support_definition_version: str = "not-defined"
    calibration_version: str = "not-defined"
    forward_model_version: str = "not-defined"
    decision_policy_version: str = "not-defined"
    pipeline_version: str = "unified-inference-v1"

    def as_dict(self) -> dict[str, str]:
        return {
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "base_model_version": self.base_model_version,
            "proposal_model_version": self.proposal_model_version,
            "support_definition_version": self.support_definition_version,
            "calibration_version": self.calibration_version,
            "forward_model_version": self.forward_model_version,
            "decision_policy_version": self.decision_policy_version,
            "pipeline_version": self.pipeline_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceRecord:
        return cls(
            dataset_manifest_hash=data.get("dataset_manifest_hash", "not-defined"),
            base_model_version=data.get("base_model_version", "base-model-v1"),
            proposal_model_version=data.get("proposal_model_version", "proposal-model-v1"),
            support_definition_version=data.get("support_definition_version", "not-defined"),
            calibration_version=data.get("calibration_version", "not-defined"),
            forward_model_version=data.get("forward_model_version", "not-defined"),
            decision_policy_version=data.get("decision_policy_version", "not-defined"),
            pipeline_version=data.get("pipeline_version", "unified-inference-v1"),
        )


def build_provenance_record(
    dataset_manifest_hash: str = "not-defined",
    base_model_version: str = "base-model-v1",
    proposal_model_version: str = "proposal-model-v1",
    support_definition_version: str = "not-defined",
    calibration_version: str = "not-defined",
    forward_model_version: str = "not-defined",
    decision_policy_version: str = "not-defined",
) -> ProvenanceRecord:
    """Build a validated ProvenanceRecord guaranteed to mark missing fields as 'not-defined'."""
    return ProvenanceRecord(
        dataset_manifest_hash=dataset_manifest_hash or "not-defined",
        base_model_version=base_model_version or "not-defined",
        proposal_model_version=proposal_model_version or "not-defined",
        support_definition_version=support_definition_version or "not-defined",
        calibration_version=calibration_version or "not-defined",
        forward_model_version=forward_model_version or "not-defined",
        decision_policy_version=decision_policy_version or "not-defined",
    )
