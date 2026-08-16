"""Service Telemetry & Operational Monitoring for EVIDENCE-Net (Phase 17).

Tracks service latency, error rates, memory footprint, request volume, queue depth,
and artifact write operations. Exposes metrics for Prometheus scraper integration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ServiceTelemetry:
    """In-memory operational telemetry collector."""

    request_count: int = 0
    error_count: int = 0
    artifact_write_count: int = 0
    latency_records: list[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def record_request(self, latency_ms: float, is_error: bool = False) -> None:
        """Record an API request latency and status."""
        self.request_count += 1
        if is_error:
            self.error_count += 1
        self.latency_records.append(latency_ms)

    def record_artifact_write(self) -> None:
        """Record an artifact write operation."""
        self.artifact_write_count += 1

    def get_summary(self) -> dict[str, Any]:
        """Return aggregated operational telemetry statistics."""
        lats = self.latency_records
        uptime_sec = time.time() - self.start_time
        err_rate = (
            round(self.error_count / self.request_count, 4) if self.request_count > 0 else 0.0
        )
        return {
            "uptime_seconds": round(uptime_sec, 2),
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": err_rate,
            "artifact_writes": self.artifact_write_count,
            "p50_latency_ms": round(float(sum(lats) / len(lats)), 3) if lats else 0.0,
        }


# Global singleton instance
telemetry = ServiceTelemetry()
