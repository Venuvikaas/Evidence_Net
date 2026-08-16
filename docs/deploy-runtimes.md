# EVIDENCE-Net Deployment Runtimes & Hardware Specifications (Phase 15)

- **Version:** v1
- **Status:** frozen
- **Owner:** Developer D

---

## 1. Overview

This document specifies runtime container requirements, hardware recommendations, and ONNX Runtime performance profiles for deploying EVIDENCE-Net.

---

## 2. Hardware Recommendations

| Deployment Tier | Minimum CPU | Recommended GPU | RAM | Storage |
| --- | --- | --- | --- | --- |
| **Local Dev / Edge** | 4-core x86_64 / ARM64 | Optional (CPU fallback) | 8 GB | 20 GB SSD |
| **Production Review Service** | 8-core x86_64 | NVIDIA T4 / A10G (16GB VRAM) | 16 GB | 100 GB NVMe |
| **Batch Processing Pipeline** | 16-core x86_64 | NVIDIA L4 / A100 | 32 GB | 500 GB NVMe |

---

## 3. Container & Memory Profile

- **Base Container Footprint:** ~350 MB RAM
- **Model Graph Memory:** Base (~45 MB), Detail Proposal (~28 MB)
- **Peak Batch Inference RAM:** ~1.2 GB per 64x64 batch of 32
- **Throughput Spec:** $\ge 150$ samples/sec on GPU, $\ge 35$ samples/sec on CPU.
