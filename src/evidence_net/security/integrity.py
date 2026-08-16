"""SHA256 Integrity Validator for EVIDENCE-Net (Phase 16).

Verifies checksum integrity of model checkpoints, policies, contract manifests,
and artifact outputs against registered sha256 digests.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_sha256(path: Path | str) -> str:
    """Compute SHA256 hex digest for a file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for checksum calculation: {path}")

    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_integrity(path: Path | str, expected_sha256: str) -> bool:
    """Verify that a file's SHA256 digest matches the expected hash."""
    clean_expected = expected_sha256.removeprefix("sha256:").strip().lower()
    actual = compute_sha256(path).lower()
    return actual == clean_expected
