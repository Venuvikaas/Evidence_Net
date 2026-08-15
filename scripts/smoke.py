"""Initial smoke pipeline: load a fixture and write a run bundle.

Usage (from the repository root)::

    python scripts/smoke.py
    python scripts/smoke.py --run-id smoke-20260815-test --out runs

The smoke path is the minimal vertical slice
``manifest -> sample -> preprocess -> infer -> evaluate -> save artifacts ->
generate report`` and must keep working as later phases extend it.
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
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id
except ImportError:  # allow running before `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id  # noqa: E402

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


def build_summary(
    run_id: str, fixture_path: Path, metrics: dict[str, float | list[int] | str]
) -> str:
    return (
        f"# Smoke run {run_id}\n\n"
        f"- Fixture: `{fixture_path}`\n"
        f"- Outcome: fixture loaded and run bundle written (Phase 0 skeleton).\n\n"
        "## Metrics\n\n| metric | value |\n| --- | --- |\n"
        + "\n".join(f"| {key} | {value} |" for key, value in metrics.items())
        + "\n"
    )


def main() -> int:
    args = parse_args()
    fixture_path = REPO_ROOT / FIXTURE_RELATIVE
    if not fixture_path.exists():
        print(f"FAIL fixture missing: {fixture_path}", file=sys.stderr)
        return 1

    array = load_fixture(fixture_path)
    run_id = args.run_id or new_run_id("smoke")

    config = {
        "phase": 0,
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
    metrics: dict[str, float | list[int] | str] = {
        "sample_shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "n_files_processed": 1,
    }
    summary = build_summary(run_id, fixture_path, metrics)

    run_dir = create_run_bundle(
        args.out,
        run_id,
        config=config,
        manifest=manifest,
        metrics=metrics,
        summary=summary,
        reference=f"reference-input: {FIXTURE_RELATIVE.as_posix()}",
    )
    print(f"Run bundle written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
