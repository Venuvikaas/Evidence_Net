"""Monitoring and operations tests (Phase 17)."""

from __future__ import annotations

import numpy as np
import pytest

from evidence_net.monitoring.drift import compute_input_drift
from evidence_net.monitoring.telemetry import ServiceTelemetry
from evidence_net.monitoring.version_tracking import VersionTracker


def test_telemetry_aggregates_requests_and_errors() -> None:
    telemetry = ServiceTelemetry()
    telemetry.record_request(10.0)
    telemetry.record_request(20.0, is_error=True)
    telemetry.record_artifact_write()
    summary = telemetry.get_summary()
    assert summary["total_requests"] == 2
    assert summary["total_errors"] == 1
    assert summary["error_rate"] == 0.5
    assert summary["artifact_writes"] == 1
    assert summary["p50_latency_ms"] == 15.0


def test_telemetry_empty_is_zero() -> None:
    summary = ServiceTelemetry().get_summary()
    assert summary["total_requests"] == 0
    assert summary["error_rate"] == 0.0
    assert summary["p50_latency_ms"] == 0.0


def test_drift_detects_mean_shift() -> None:
    current = np.full((16, 16), 0.9)
    report = compute_input_drift(current, baseline_mean=0.5, shift_threshold=0.15)
    assert report.mean_shift > 0.15
    assert report.has_drift_alert


def test_drift_quiet_within_baseline() -> None:
    rng = np.random.default_rng(0)
    current = rng.normal(0.5, 0.25, size=(16, 16)).clip(0.0, 1.0)
    report = compute_input_drift(current, baseline_mean=0.5, baseline_std=0.25)
    assert not report.has_drift_alert or report.out_of_range_ratio <= 0.01


def test_drift_flags_out_of_range() -> None:
    current = np.full((8, 8), 1.7)
    report = compute_input_drift(current)
    assert report.out_of_range_ratio > 0.01
    assert report.has_drift_alert


def test_version_tracker_records_actions_and_versions() -> None:
    tracker = VersionTracker()
    tracker.register_version("support", "support-definition-v1")
    tracker.register_version("policy", "decision-policy-v1")
    tracker.record_action_map(["accept", "accept", "reject"], unresolved=[False, True, True])
    summary = tracker.get_summary()
    assert summary["active_versions"]["support"] == "support-definition-v1"
    assert summary["action_fractions"]["accept"] == pytest.approx(2 / 3, abs=1e-3)
    assert summary["unresolved_fraction"] == pytest.approx(2 / 3, abs=1e-3)
    assert summary["n_patches"] == 3
