"""Tests for the dataset manifest contract implementation."""

import json
from pathlib import Path

import pytest

from evidence_net.data.manifests import (
    MANIFEST_SCHEMA_VERSION,
    ManifestValidationError,
    SourceManifest,
    validate_source_manifest,
    verify_dataset_hash,
    write_manifest,
)


def _manifest_dict() -> dict:
    return {
        "manifest_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": "test-dataset-v1",
        "created_at": "2026-08-15T00:00:00Z",
        "source_root": "/data/root",
        "hash_algorithm": "sha256",
        "dataset_hash": "",
        "provenance": {"source": "test"},
        "grouping": {"source_group_field": "sample-id"},
        "files": [
            {
                "relative_path": "NoisyLR/000000.npy",
                "extension": ".npy",
                "byte_size": 100,
                "sha256": "a" * 64,
                "readable": True,
                "dimensions": [128, 128],
                "channels": 1,
                "dtype": "float32",
                "range": [0.0, 1.0],
            }
        ],
    }


def test_manifest_hash_is_stable_and_verifiable() -> None:
    data = _manifest_dict()
    manifest = SourceManifest.from_dict(data)
    digest = manifest.compute_dataset_hash()
    assert len(digest) == 64
    data["dataset_hash"] = digest
    assert verify_dataset_hash(data)
    data["files"][0]["byte_size"] += 1
    assert not verify_dataset_hash(data)


def test_validation_accepts_valid_manifest() -> None:
    data = _manifest_dict()
    validate_source_manifest(data, allow_development_labels=True)


def test_validation_rejects_bad_sha() -> None:
    data = _manifest_dict()
    data["files"][0]["sha256"] = "not-a-hash"
    with pytest.raises(ManifestValidationError, match="sha256"):
        validate_source_manifest(data, allow_development_labels=True)


def test_test_manifest_rejects_development_labels() -> None:
    data = _manifest_dict()
    data["files"][0]["split_label"] = "train"
    with pytest.raises(ManifestValidationError, match="split_label"):
        validate_source_manifest(data, allow_development_labels=False)
    data["files"][0].pop("split_label")
    data["files"][0]["role"] = "input"
    with pytest.raises(ManifestValidationError, match="role"):
        validate_source_manifest(data, allow_development_labels=False)


def test_validation_rejects_absolute_path() -> None:
    data = _manifest_dict()
    data["files"][0]["relative_path"] = "/abs/NoisyLR/000000.npy"
    with pytest.raises(ManifestValidationError, match="relative"):
        validate_source_manifest(data, allow_development_labels=True)


def test_write_manifest_round_trip(tmp_path: Path) -> None:
    manifest = SourceManifest.from_dict(_manifest_dict())
    manifest.dataset_hash = manifest.compute_dataset_hash()
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert verify_dataset_hash(loaded)
    assert loaded["files"][0]["relative_path"] == "NoisyLR/000000.npy"


def test_index_samples_is_deterministic_and_sorted() -> None:
    from evidence_net.data.manifests import FileEntry, index_samples

    def entry(path: str, sha: str) -> FileEntry:
        return FileEntry(
            relative_path=path,
            extension=".npy",
            byte_size=1,
            sha256=sha,
            readable=True,
        )

    entries = [entry("NoisyLR/000002.npy", "a" * 64), entry("NoisyLR/000001.npy", "b" * 64)]
    indexed = index_samples(entries)
    assert list(indexed) == ["NoisyLR/000001.npy", "NoisyLR/000002.npy"]
    assert index_samples(entries) == indexed


def test_index_samples_reports_duplicate_keys() -> None:
    from evidence_net.data.manifests import (
        FileEntry,
        ManifestValidationError,
        index_samples,
    )

    def entry(path: str, sha: str) -> FileEntry:
        return FileEntry(
            relative_path=path,
            extension=".npy",
            byte_size=1,
            sha256=sha,
            readable=True,
        )

    entries = [entry("NoisyLR/000000.npy", "a" * 64), entry("NoisyLR/000000.npy", "b" * 64)]
    with pytest.raises(ManifestValidationError, match="duplicate sample key"):
        index_samples(entries)
