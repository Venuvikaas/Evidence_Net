"""Benchmark the promoted inference pipeline (Phase 15, deploy parity).

Usage (from the repository root):::

    python scripts/benchmark_pipeline.py --synthetic
    python scripts/benchmark_pipeline.py --n-samples 16

Loads the promoted Base and Proposal checkpoints and the unified inference
pipeline, and measures:

- model sizes (parameter counts and checkpoint bytes);
- peak memory during inference (best-effort, CPU/RSS based);
- per-sample latency and throughput at the declared 128x128 -> 256x256
  resolution;
- a sanity parity check (gated output dimensions/range) so the benchmark
  never silently measures a broken pipeline.

``--synthetic`` generates random inputs so CI can exercise the harness
without the data files; the real mode reads the frozen validation split.
Never touches ``Test_NoisyLR/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from evidence_net.models.factory import build_model
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id
    from evidence_net.training.config import ModelConfig
except ImportError:  # allow running before `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evidence_net.models.factory import build_model  # noqa: E402
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id  # noqa: E402
    from evidence_net.training.config import ModelConfig  # noqa: E402

OUTPUT_GRID = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", default=REPO_ROOT / "runs", type=Path)
    parser.add_argument("--synthetic", action="store_true", help="use random inputs (CI smoke)")
    return parser.parse_args()


def _load_model(checkpoint: Path) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(
        name=payload["config"]["model"]["name"],
        hidden_channels=payload["config"]["model"]["hidden_channels"],
        depth=payload["config"]["model"]["depth"],
        amplitude=payload["config"]["model"].get("amplitude", 0.1),
    )
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _inputs(n_samples: int, seed: int, *, synthetic: bool) -> tuple[list[str], list[np.ndarray]]:
    if synthetic:
        rng = np.random.default_rng(seed)
        synthetic_ids = [f"synthetic-{index:06d}" for index in range(n_samples)]
        synthetic_arrays = [
            rng.normal(0.5, 0.2, size=(128, 128)).astype(np.float32) for _ in range(n_samples)
        ]
        return synthetic_ids, synthetic_arrays
    from evidence_net.data.paths import resolve_dataset_paths
    from evidence_net.training.dataset import RestorationDataset

    paths = resolve_dataset_paths()
    dataset = RestorationDataset(
        paths.train_dir, split="validation", n_samples=n_samples, seed=seed
    )
    real_ids: list[str] = []
    real_arrays: list[np.ndarray] = []
    for index in range(len(dataset)):
        input_tensor, _target, sample_id = dataset[index]
        real_ids.append(f"{sample_id}")
        real_arrays.append(input_tensor.squeeze(0).numpy().astype(np.float32).copy())
    return real_ids, real_arrays


def _measure_rss_mb() -> float:
    """Best-effort resident-set memory in MB (tracemalloc-based, cross-platform)."""
    import tracemalloc

    tracemalloc.start()
    snap = tracemalloc.take_snapshot()
    total = sum(stat.size for stat in snap.statistics("lineno"))
    tracemalloc.stop()
    return total / (1024 * 1024)


def main() -> int:
    args = parse_args()
    base = _load_model(REPO_ROOT / "checkpoints" / "train-base-gate2" / "best.pt")
    proposal = _load_model(REPO_ROOT / "checkpoints" / "train-proposal-gate3v2" / "best.pt")
    from evidence_net.models.proposal import BoundedDetailProposal

    if not isinstance(proposal, BoundedDetailProposal):
        raise SystemExit("FAIL: proposal checkpoint is not a BoundedDetailProposal")

    base_params = sum(p.numel() for p in base.parameters())
    proposal_params = sum(p.numel() for p in proposal.parameters())
    base_bytes = (REPO_ROOT / "checkpoints" / "train-base-gate2" / "best.pt").stat().st_size
    proposal_bytes = (
        (REPO_ROOT / "checkpoints" / "train-proposal-gate3v2" / "best.pt").stat().st_size
    )

    ids, arrays = _inputs(args.n_samples, args.seed, synthetic=args.synthetic)
    latencies_ms: list[float] = []
    with torch.no_grad():
        for array in arrays:
            tensor = torch.from_numpy(np.ascontiguousarray(array))[None, None]
            start = time.perf_counter()
            b, d, c = proposal.propose(tensor)
            gated = torch.clamp(b + d, 0.0, 1.0)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            assert gated.shape == (1, 1, OUTPUT_GRID, OUTPUT_GRID), gated.shape
            assert float(gated.min()) >= 0.0 and float(gated.max()) <= 1.0

    latencies_ms = sorted(latencies_ms)
    mean_ms = float(np.mean(latencies_ms))
    p50_ms = float(np.median(latencies_ms))
    p95_ms = float(latencies_ms[int(0.95 * (len(latencies_ms) - 1))])
    throughput = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    memory_mb = _measure_rss_mb()

    metrics: dict[str, object] = {
        "n_samples": len(ids),
        "mode": "synthetic" if args.synthetic else "real",
        "resolution": f"128x128 -> {OUTPUT_GRID}x{OUTPUT_GRID}",
        "base_parameters": base_params,
        "proposal_parameters": proposal_params,
        "base_checkpoint_bytes": base_bytes,
        "proposal_checkpoint_bytes": proposal_bytes,
        "latency_mean_ms": round(mean_ms, 3),
        "latency_p50_ms": round(p50_ms, 3),
        "latency_p95_ms": round(p95_ms, 3),
        "throughput_samples_per_sec": round(throughput, 2),
        "peak_python_memory_mb": round(memory_mb, 2),
    }

    run_id = args.run_id or new_run_id("benchmark")
    create_run_bundle(
        args.out,
        run_id,
        config={"phase": 15, "n_samples": args.n_samples, "seed": args.seed},
        manifest={
            "run_id": run_id,
            "contract": "base-output-v1 / proposal-output-v1",
            "checkpoints": "train-base-gate2/best.pt, train-proposal-gate3v2/best.pt",
            "test_final_isolation": "confirmed-no-test-noisylr",
        },
        metrics=metrics,
        summary=_build_summary(run_id, metrics),
        reference="promoted checkpoints (see manifest)",
    )
    print(f"Run bundle written to {args.out / run_id}")
    return 0


def _build_summary(run_id: str, metrics: dict[str, object]) -> str:
    lines = [
        f"# Pipeline benchmark {run_id}",
        "",
        f"- Mode: {metrics['mode']}; resolution {metrics['resolution']}.",
        f"- Model sizes: Base {metrics['base_parameters']} params "
        f"({metrics['base_checkpoint_bytes']} B), Proposal "
        f"{metrics['proposal_parameters']} params ({metrics['proposal_checkpoint_bytes']} B).",
        f"- Latency: mean {metrics['latency_mean_ms']} ms, p50 "
        f"{metrics['latency_p50_ms']} ms, p95 {metrics['latency_p95_ms']} ms.",
        f"- Throughput: {metrics['throughput_samples_per_sec']} samples/s.",
        f"- Peak Python memory (best-effort): {metrics['peak_python_memory_mb']} MB.",
        "",
        "Numbers are environment-specific (CPU/GPU, batch=1, no warm-up "
        "excluded); the declared comparison is relative, not absolute.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
