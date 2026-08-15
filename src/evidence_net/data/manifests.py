"""Dataset manifest models, schema validation, and hashing.

Implements the dataset manifest contract v1
(``docs/dataset-manifest-contract.md``). Frozen source manifests are committed
under ``data/manifests/`` and are immutable once created.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "1"
HASH_ALGORITHM = "sha256"

SPLIT_LABELS = (
    "train",
    "validation",
    "calibration",
    "heldout-source",
    "heldout-degradation",
    "test-final",
)

ROLE_LABELS = ("input", "target", "input_and_target", "unknown")


class ManifestValidationError(ValueError):
    """Raised when a manifest does not conform to the v1 contract."""


@dataclass
class FileEntry:
    """One file entry in a source manifest."""

    relative_path: str
    extension: str
    byte_size: int
    sha256: str
    readable: bool
    dimensions: list[int] | None = None
    channels: int | None = None
    dtype: str | None = None
    range: list[float] | None = None
    source_group: str | None = None
    split_label: str | None = None
    role: str | None = None
    target_uncertainty: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "relative_path": self.relative_path,
            "extension": self.extension,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "readable": self.readable,
        }
        for key in (
            "dimensions",
            "channels",
            "dtype",
            "range",
            "source_group",
            "split_label",
            "role",
            "target_uncertainty",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileEntry:
        return cls(
            relative_path=data["relative_path"],
            extension=data["extension"],
            byte_size=data["byte_size"],
            sha256=data["sha256"],
            readable=data["readable"],
            dimensions=data.get("dimensions"),
            channels=data.get("channels"),
            dtype=data.get("dtype"),
            range=data.get("range"),
            source_group=data.get("source_group"),
            split_label=data.get("split_label"),
            role=data.get("role"),
            target_uncertainty=data.get("target_uncertainty"),
        )


@dataclass
class SourceManifest:
    """A frozen, versioned dataset source manifest."""

    manifest_version: str
    dataset_id: str
    created_at: str
    source_root: str
    hash_algorithm: str
    dataset_hash: str
    provenance: dict[str, Any]
    grouping: dict[str, Any]
    structure: dict[str, Any] | None = None
    files: list[FileEntry] = field(default_factory=list)

    def to_dict(self, *, include_files: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "manifest_version": self.manifest_version,
            "dataset_id": self.dataset_id,
            "created_at": self.created_at,
            "source_root": self.source_root,
            "hash_algorithm": self.hash_algorithm,
            "dataset_hash": self.dataset_hash,
            "provenance": self.provenance,
            "grouping": self.grouping,
        }
        if self.structure is not None:
            data["structure"] = self.structure
        if include_files:
            data["files"] = [entry.to_dict() for entry in self.files]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceManifest:
        return cls(
            manifest_version=data["manifest_version"],
            dataset_id=data["dataset_id"],
            created_at=data["created_at"],
            source_root=data["source_root"],
            hash_algorithm=data["hash_algorithm"],
            dataset_hash=data["dataset_hash"],
            provenance=data["provenance"],
            grouping=data["grouping"],
            structure=data.get("structure"),
            files=[FileEntry.from_dict(item) for item in data.get("files", [])],
        )

    def compute_dataset_hash(self) -> str:
        """Hash the canonical JSON of this manifest (without the stored hash)."""
        data = self.to_dict(include_files=True)
        data["dataset_hash"] = ""
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def _require(data: dict[str, Any], key: str, expected: type) -> Any:
    if key not in data:
        raise ManifestValidationError(f"missing required field: {key}")
    value = data[key]
    if not isinstance(value, expected):
        raise ManifestValidationError(f"field {key} must be {expected.__name__}")
    return value


def _validate_file_entry(item: dict[str, Any], index: int) -> None:
    _require(item, "relative_path", str)
    _require(item, "extension", str)
    _require(item, "byte_size", int)
    sha = _require(item, "sha256", str)
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ManifestValidationError(f"files[{index}].sha256 must be a 64-char hex digest")
    _require(item, "readable", bool)
    relative_path = item["relative_path"]
    if relative_path.startswith("/") or ".." in relative_path.split("/"):
        raise ManifestValidationError(
            f"files[{index}].relative_path must be relative: {relative_path}"
        )
    if not item["extension"].startswith("."):
        raise ManifestValidationError(
            f"files[{index}].extension must start with '.': {item['extension']}"
        )
    if "split_label" in item and item["split_label"] not in SPLIT_LABELS:
        raise ManifestValidationError(f"files[{index}].split_label invalid: {item['split_label']}")
    if "role" in item and item["role"] not in ROLE_LABELS:
        raise ManifestValidationError(f"files[{index}].role invalid: {item['role']}")
    if "dimensions" in item:
        dimensions = item["dimensions"]
        if not isinstance(dimensions, list) or not all(
            isinstance(d, int) and d > 0 for d in dimensions
        ):
            raise ManifestValidationError(
                f"files[{index}].dimensions must be a list of positive ints"
            )
    if "range" in item:
        rng = item["range"]
        if (
            not isinstance(rng, list)
            or len(rng) != 2
            or not all(isinstance(v, (int, float)) for v in rng)
        ):
            raise ManifestValidationError(f"files[{index}].range must be a [min, max] pair")


def validate_source_manifest(data: dict[str, Any], *, allow_development_labels: bool) -> None:
    """Validate a manifest dict against the v1 contract.

    ``allow_development_labels=False`` (used for the isolated test manifest)
    rejects any development split or role label.
    """
    _require(data, "manifest_version", str)
    if data["manifest_version"] != MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError(f"unsupported manifest_version: {data['manifest_version']}")
    _require(data, "dataset_id", str)
    _require(data, "created_at", str)
    _require(data, "source_root", str)
    _require(data, "hash_algorithm", str)
    _require(data, "dataset_hash", str)
    _require(data, "provenance", dict)
    _require(data, "grouping", dict)
    files = _require(data, "files", list)
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            raise ManifestValidationError(f"files[{index}] must be an object")
        _validate_file_entry(item, index)
        if not allow_development_labels:
            if item.get("split_label") is not None:
                raise ManifestValidationError(
                    f"files[{index}] carries a development split_label in a test-isolated manifest"
                )
            if item.get("role") is not None:
                raise ManifestValidationError(
                    f"files[{index}] carries a development role in a test-isolated manifest"
                )


def verify_dataset_hash(data: dict[str, Any]) -> bool:
    """Check the stored dataset_hash matches the canonical content hash."""
    manifest = SourceManifest.from_dict(data)
    return manifest.compute_dataset_hash() == data["dataset_hash"]


def index_samples(entries: list[FileEntry], key: str = "relative_path") -> dict[str, FileEntry]:
    """Deterministic sample index: sorted unique key -> entry.

    Duplicate keys raise so ambiguous samples are reported rather than
    silently overwritten.
    """
    indexed: dict[str, FileEntry] = {}
    for entry in sorted(entries, key=lambda e: getattr(e, key)):
        value = getattr(entry, key)
        if value in indexed:
            raise ManifestValidationError(f"duplicate sample key in index: {value}")
        indexed[value] = entry
    return indexed


def write_manifest(path: Path, manifest: SourceManifest) -> None:
    """Write a manifest atomically and verify its hash afterwards."""
    payload = json.dumps(manifest.to_dict(include_files=True), indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not verify_dataset_hash(loaded):
        raise ManifestValidationError(f"dataset_hash mismatch after writing {path}")
