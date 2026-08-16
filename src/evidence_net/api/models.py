"""SQLAlchemy metadata models for EVIDENCE-Net Metadata Store (Phase 12 & Phase 14).

Defines persistence schemas for versions, runs, artifacts, metrics, policies,
and human review events.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from evidence_net.api.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class VersionModel(Base):
    """Tracks dataset, model, policy, and contract versions."""

    __tablename__ = "versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    component_name: Mapped[str] = mapped_column(String(64), index=True)
    version_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class RunModel(Base):
    """Tracks execution run metadata and bundle status."""

    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), default="restoration")
    status: Mapped[str] = mapped_column(String(32), default="completed")
    dataset_manifest_hash: Mapped[str] = mapped_column(String(64), default="not-defined")
    base_model_version: Mapped[str] = mapped_column(String(64), default="base-model-v1")
    proposal_model_version: Mapped[str] = mapped_column(String(64), default="proposal-model-v1")
    run_dir: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    artifacts: Mapped[list[ArtifactModel]] = relationship(
        "ArtifactModel", back_populates="run", cascade="all, delete-orphan"
    )
    metrics: Mapped[list[MetricModel]] = relationship(
        "MetricModel", back_populates="run", cascade="all, delete-orphan"
    )
    review_events: Mapped[list[ReviewEventModel]] = relationship(
        "ReviewEventModel", back_populates="run", cascade="all, delete-orphan"
    )


class ArtifactModel(Base):
    """Tracks tensor and report artifact files and contract metadata."""

    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("runs.run_id"), index=True)
    artifact_name: Mapped[str] = mapped_column(String(128))
    file_path: Mapped[str] = mapped_column(Text)
    dtype: Mapped[str] = mapped_column(String(32), default="float32")
    shape: Mapped[list[int]] = mapped_column(JSON, default=list)
    range_min: Mapped[float] = mapped_column(Float, default=0.0)
    range_max: Mapped[float] = mapped_column(Float, default=1.0)
    hash_sha256: Mapped[str] = mapped_column(String(128), default="")
    is_optional: Mapped[bool] = mapped_column(default=False)

    run: Mapped[RunModel] = relationship("RunModel", back_populates="artifacts")


class MetricModel(Base):
    """Tracks scalar metrics per stage and group."""

    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("runs.run_id"), index=True)
    stage: Mapped[str] = mapped_column(String(32))
    metric_name: Mapped[str] = mapped_column(String(64))
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(32), default="-")

    run: Mapped[RunModel] = relationship("RunModel", back_populates="metrics")


class PolicyModel(Base):
    """Tracks decision policy parameters and thresholds."""

    __tablename__ = "policies"

    policy_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    description: Mapped[str] = mapped_column(Text, default="")
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class ReviewEventModel(Base):
    """Tracks human reviewer events and expert interaction study logs (Phase 14)."""

    __tablename__ = "review_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("runs.run_id"), index=True)
    reviewer_id: Mapped[str] = mapped_column(String(64), default="expert-reviewer")
    action_type: Mapped[str] = mapped_column(String(64))  # e.g., accept_proposal / reject_proposal
    decision_reason: Mapped[str] = mapped_column(Text, default="")
    event_details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    run: Mapped[RunModel] = relationship("RunModel", back_populates="review_events")
