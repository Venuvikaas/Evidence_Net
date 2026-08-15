"""Run-bundle creation helpers.

A run bundle is the reproducible unit of governed work defined in
``docs/run-and-artifact-contract.md``. Every smoke run, experiment, and
evaluation writes the full bundle so its configuration, data manifest,
environment, metrics, artifacts, logs, and checkpoint reference can be
recovered.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REQUIRED_FILES = (
    "config.yaml",
    "manifest.json",
    "environment.txt",
    "metrics.json",
    "summary.md",
    "checkpoint-or-reference.txt",
)


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(prefix: str = "run") -> str:
    """Return a deterministic-in-order run id: ``<prefix>-<UTC timestamp>``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    """Hash a file's bytes with the given algorithm (default sha256)."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON file with sorted keys and a trailing newline."""
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write a YAML file with sorted keys."""
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")


def environment_text() -> str:
    """Capture python, platform, and core package versions."""
    lines = [
        f"python: {sys.version.split()[0]} ({platform.python_implementation()})",
        f"platform: {platform.platform()}",
        f"machine: {platform.machine()}",
        "packages:",
    ]
    for name in ("numpy", "yaml"):
        try:
            module = __import__(name)
            version = getattr(module, "__version__", "unknown")
        except Exception:
            version = "MISSING"
        lines.append(f"  {name}: {version}")
    return "\n".join(lines) + "\n"


def create_run_bundle(
    runs_dir: Path,
    run_id: str,
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    summary: str,
    reference: str = "no-checkpoint",
    environment: str | None = None,
) -> Path:
    """Create a run bundle directory and write all required files.

    Returns the run directory. ``runs_dir`` and ``run_id`` must be
    ``runs/`` and a valid id per the run/artifact contract.
    """
    run_dir = runs_dir / run_id
    artifacts_dir = run_dir / "artifacts"
    logs_dir = run_dir / "logs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "metrics.json", metrics)
    (run_dir / "environment.txt").write_text(
        environment if environment is not None else environment_text(),
        encoding="utf-8",
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    (run_dir / "checkpoint-or-reference.txt").write_text(reference + "\n", encoding="utf-8")
    (logs_dir / "run.log").write_text(f"[{utc_now()}] run {run_id} created\n", encoding="utf-8")
    return run_dir
