"""Smoke pipeline: fixture -> sample -> infer -> evaluate -> report.

Usage (from the repository root)::

    python scripts/smoke.py
    python scripts/smoke.py --run-id smoke-20260815-test --out runs

The smoke path is the minimal vertical slice
``manifest -> sample -> preprocess -> infer -> evaluate -> save artifacts ->
generate report`` and must keep working as later phases extend it. Since
Phase 2 it runs the classical baselines over a fixture pair and writes a
comparison report alongside the run bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from evidence_net.inference.baseline import evaluate_restorers
    from evidence_net.models.reference import (
        classical_restoration,
        deterministic_reconstruction,
    )
    from evidence_net.reporting.comparison_report import (
        write_comparison_report,
        write_comparison_sheet,
    )
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id
except ImportError:  # allow running before `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evidence_net.inference.baseline import evaluate_restorers  # noqa: E402
    from evidence_net.models.reference import (  # noqa: E402
        classical_restoration,
        deterministic_reconstruction,
    )
    from evidence_net.reporting.comparison_report import (  # noqa: E402
        write_comparison_report,
        write_comparison_sheet,
    )
    from evidence_net.reporting.run_bundle import (  # noqa: E402
        create_run_bundle,
        new_run_id,
    )

FIXTURE_RELATIVE = Path("data/fixtures/sample_8x8.npy")


PIPELINE = (
    "manifest -> sample -> preprocess -> infer -> evaluate -> save artifacts -> generate report"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id", default=None, help="explicit run id (default: smoke-<timestamp>)"
    )
    parser.add_argument("--out", default=REPO_ROOT / "runs", type=Path, help="runs directory")
    return parser.parse_args()


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_fixture(path: Path) -> np.ndarray:
    """Load the fixture, preserving raw values and metadata."""
    return np.load(path)


def make_smoke_pair(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic fixture pair: mean-pooled input and original target.

    Mirrors the 2x super-resolution contract (128x128 input -> 256x256
    target) at fixture scale: 4x4 input -> 8x8 target.
    """
    input_ = array.reshape(4, 2, 4, 2).mean(axis=(1, 3))
    return input_, array


def build_summary(
    run_id: str,
    fixture_path: Path,
    metrics: dict[str, float | list[int] | str],
    results: dict,
    sample_ids: list[str],
) -> str:
    lines = [
        f"# Smoke run {run_id}",
        "",
        f"- Fixture: `{fixture_path}`",
        "- Outcome: fixture loaded; baselines evaluated and run bundle "
        "written (Phase 2 smoke path).",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in metrics.items())
    lines.extend(["", "## Baseline aggregates", ""])
    for name, result in sorted(results.items()):
        agg = result.aggregates
        lines.append(
            f"- **{name}**: PSNR {agg['psnr']['mean']:.4f} dB, "
            f"SSIM {agg['ssim']['mean']:.4f}, MAE {agg['mae']['mean']:.4f}"
        )
    lines.append("")
    lines.append(f"Groups: {', '.join(sample_ids)}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    fixture_path = REPO_ROOT / FIXTURE_RELATIVE
    if not fixture_path.exists():
        print(f"FAIL fixture missing: {fixture_path}", file=sys.stderr)
        return 1

    array = load_fixture(fixture_path)
    run_id = args.run_id or new_run_id("smoke")

    config = {
        "phase": 2,
        "pipeline": PIPELINE,
        "fixture": FIXTURE_RELATIVE.as_posix(),
        "seed": None,
        "device": "cpu",
    }
    fixture_bytes = fixture_path.read_bytes()
    manifest = {
        "run_id": run_id,
        "dataset_manifest": "not-defined",
        "sample": {
            "relative_path": FIXTURE_RELATIVE.as_posix(),
            "byte_size": len(fixture_bytes),
            "sha256": sha256_of_bytes(fixture_bytes),
        },
        "config_sha256": sha256_of_bytes(json.dumps(config, sort_keys=True).encode("utf-8")),
    }
    input_, target = make_smoke_pair(array)
    results = evaluate_restorers(
        [input_],
        [target],
        ["fixture-000000"],
        {
            "deterministic-bilinear": deterministic_reconstruction,
            "classical-median5-bilinear": classical_restoration,
        },
        n_boot=100,
        seed=0,
    )
    metrics: dict[str, float | list[int] | str] = {
        "sample_shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "n_files_processed": 1,
        "n_restorers": len(results),
    }
    summary = build_summary(run_id, fixture_path, metrics, results, ["fixture-000000"])

    run_dir = create_run_bundle(
        args.out,
        run_id,
        config=config,
        manifest=manifest,
        metrics={name: result.aggregates for name, result in sorted(results.items())},
        summary=summary,
        reference=f"reference-input: {FIXTURE_RELATIVE.as_posix()}",
    )
    prediction = np.clip(deterministic_reconstruction(input_), 0.0, 1.0)
    write_comparison_sheet(
        run_dir / "artifacts",
        0,
        input_,
        prediction,
        target,
        results["deterministic-bilinear"].per_group_metrics["fixture-000000"],
    )
    write_comparison_report(
        run_dir,
        results,
        sample_ids=["fixture-000000"],
        n_samples=1,
        split_label="smoke",
    )
    print(f"Run bundle written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
