"""Unit tests for run-bundle creation."""

import json
from pathlib import Path

import yaml

from evidence_net.reporting.run_bundle import (
    REQUIRED_FILES,
    create_run_bundle,
    hash_file,
    new_run_id,
)


def test_create_run_bundle_writes_required_files(tmp_path: Path) -> None:
    run_dir = create_run_bundle(
        tmp_path,
        "smoke-test-run",
        config={"a": 1},
        manifest={"b": 2},
        metrics={"c": 3.0},
        summary="# summary",
    )
    for name in REQUIRED_FILES:
        assert (run_dir / name).exists(), f"missing {name}"
    assert (run_dir / "artifacts").is_dir()
    assert (run_dir / "logs").is_dir()


def test_run_bundle_contents_are_valid(tmp_path: Path) -> None:
    run_dir = create_run_bundle(
        tmp_path,
        "smoke-test-run-2",
        config={"phase": 0},
        manifest={"sample": {"id": "x"}},
        metrics={"mean": 0.5},
        summary="# ok",
    )
    assert yaml.safe_load((run_dir / "config.yaml").read_text()) == {"phase": 0}
    assert json.loads((run_dir / "manifest.json").read_text()) == {"sample": {"id": "x"}}
    assert json.loads((run_dir / "metrics.json").read_text()) == {"mean": 0.5}
    env = (run_dir / "environment.txt").read_text()
    assert "python:" in env and "packages:" in env
    assert (run_dir / "checkpoint-or-reference.txt").read_text().strip() == "no-checkpoint"


def test_hash_file_matches_sha256(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "blob.bin"
    payload = b"evidence-net"
    path.write_bytes(payload)
    assert hash_file(path) == hashlib.sha256(payload).hexdigest()


def test_new_run_id_has_prefix_and_timestamp() -> None:
    run_id = new_run_id("smoke")
    prefix, stamp = run_id.split("-", 1)
    assert prefix == "smoke"
    assert len(stamp) == 15  # YYYYMMDD-HHMMSS
    assert stamp[8] == "-"
