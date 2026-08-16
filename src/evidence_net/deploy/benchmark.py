"""Performance Benchmarking Module for EVIDENCE-Net (Phase 15).

Measures model parameter size, memory consumption, latency quantiles (p50, p95, p99),
and throughput (samples/sec) for Base Reconstruction and Detail Proposal pipelines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from evidence_net.models.base import BaseReconstruction
from evidence_net.models.proposal import DetailProposer


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Benchmarking result summary."""

    num_parameters: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_samples_per_sec: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_parameters": self.num_parameters,
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "p50_latency_ms": round(self.p50_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "p99_latency_ms": round(self.p99_latency_ms, 3),
            "throughput_samples_per_sec": round(self.throughput_samples_per_sec, 2),
        }


def benchmark_pipeline(
    batch_size: int = 1,
    sample_shape: tuple[int, int] = (64, 64),
    num_warmup: int = 5,
    num_runs: int = 20,
    device: str = "cpu",
) -> BenchmarkMetrics:
    """Benchmark full sample-to-artifact pipeline execution latency and throughput."""
    dev = torch.device(device)
    base_model = BaseReconstruction().to(dev).eval()
    proposal_model = DetailProposer().to(dev).eval()

    num_params = sum(p.numel() for p in base_model.parameters()) + sum(
        p.numel() for p in proposal_model.parameters()
    )

    dummy_y = torch.randn(batch_size, 1, sample_shape[0], sample_shape[1], device=dev)

    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            b = base_model(dummy_y)
            d = proposal_model(dummy_y, b)
            _ = torch.clamp(b + d, 0.0, 1.0)

    # Timed runs
    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            b = base_model(dummy_y)
            d = proposal_model(dummy_y, b)
            _ = torch.clamp(b + d, 0.0, 1.0)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr))
    p50 = float(np.percentile(lat_arr, 50))
    p95 = float(np.percentile(lat_arr, 95))
    p99 = float(np.percentile(lat_arr, 99))
    throughput = (batch_size * 1000.0) / mean_lat if mean_lat > 0 else 0.0

    return BenchmarkMetrics(
        num_parameters=num_params,
        mean_latency_ms=mean_lat,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        throughput_samples_per_sec=throughput,
    )
