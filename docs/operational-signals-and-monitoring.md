# EVIDENCE-Net Operational Signals and Monitoring Runbook (Phase 17)

- **Version:** v1
- **Status:** frozen
- **Owner:** Developer D

---

## 1. Overview

This document specifies operational telemetry signals, Prometheus metric names, alert thresholds, and data health drift indicators.

---

## 2. Telemetry Signals & Metrics

| Metric Name | Type | Description | Target / Alert Threshold |
| --- | --- | --- | --- |
| `evidence_net_requests_total` | Counter | Total inference and API request count | N/A |
| `evidence_net_errors_total` | Counter | Total 4xx/5xx API error count | Alert if $> 1\%$ over 5m |
| `evidence_net_inference_latency_ms` | Histogram | End-to-end sample restoration latency | Alert if $p95 > 250\text{ms}$ |
| `evidence_net_input_drift_shift` | Gauge | Shift in input mean vs development baseline | Alert if $> 0.15$ |
| `evidence_net_artifact_writes` | Counter | Artifact write operations to `runs/` | Alert on filesystem write failure |

---

## 3. Data Health & Drift Monitoring

Input tensors are evaluated continuously against baseline distribution parameters:
- **Range Violation:** Values outside $[0.0, 1.0]$ indicate acquisition distortion.
- **Mean Shift:** Mean shift $> 0.15$ triggers a data drift alert.
- **Action Required:** Flag input sample, leave optional fields as `"not-defined"`, and inform reviewer via telemetry dashboard.
