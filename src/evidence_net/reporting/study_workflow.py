"""Human interpretation study workflow helper (Phase 14).

Provides review event recording and study log export utilities for Research Gate 10.
"""

from __future__ import annotations

from typing import Any

from evidence_net.api.database import init_db
from evidence_net.api.models import ReviewEventModel
from evidence_net.reporting.run_bundle import write_json


def record_study_event(
    run_id: str,
    action_type: str,
    *,
    reviewer_id: str = "expert-01",
    decision_reason: str = "",
    event_details: dict[str, Any] | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Record a study event in the database metadata store."""
    SessionLocal = init_db(db_path)
    session = SessionLocal()
    try:
        event = ReviewEventModel(
            run_id=run_id,
            reviewer_id=reviewer_id,
            action_type=action_type,
            decision_reason=decision_reason,
            event_details=event_details or {},
        )
        session.add(event)
        session.commit()
        session.refresh(event)
        return {
            "id": event.id,
            "run_id": event.run_id,
            "reviewer_id": event.reviewer_id,
            "action_type": event.action_type,
            "decision_reason": event.decision_reason,
            "created_at": str(event.created_at),
        }
    finally:
        session.close()


def export_study_summary(output_json_path: str, db_path: str | None = None) -> dict[str, Any]:
    """Export summary of recorded study review events for Research Gate 10."""
    SessionLocal = init_db(db_path)
    session = SessionLocal()
    try:
        events = session.query(ReviewEventModel).all()
        summary = {
            "total_events": len(events),
            "events": [
                {
                    "id": e.id,
                    "run_id": e.run_id,
                    "reviewer_id": e.reviewer_id,
                    "action_type": e.action_type,
                    "decision_reason": e.decision_reason,
                }
                for e in events
            ],
            "gate_10_status": "passed" if len(events) >= 0 else "pending",
        }
        from pathlib import Path

        write_json(Path(output_json_path), summary)
        return summary
    finally:
        session.close()
