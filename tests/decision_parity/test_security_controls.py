"""Security controls and integrity test suite (Phase 16)."""

from __future__ import annotations

import pytest

from evidence_net.security.hardening import (
    redact_tensor_repr,
    sanitize_filename,
    validate_file_path,
    validate_upload_meta,
)
from evidence_net.security.integrity import compute_sha256, verify_file_integrity


def test_path_traversal_prevention(tmp_path):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    valid_file = base_dir / "safe.txt"
    valid_file.write_text("safe content", encoding="utf-8")

    # Valid relative access
    res = validate_file_path(base_dir, valid_file)
    assert res == valid_file.resolve()

    # Invalid path traversal attempt
    with pytest.raises(ValueError, match="Path traversal attempt detected"):
        validate_file_path(base_dir, base_dir / "../outside.txt")


def test_filename_sanitization():
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("safe_image.npy") == "safe_image.npy"


def test_upload_meta_validation():
    valid, _ = validate_upload_meta("sample.npy", 1000)
    assert valid is True

    valid_bad_ext, msg = validate_upload_meta("malicious.sh", 1000)
    assert valid_bad_ext is False
    assert "not allowed" in msg

    valid_huge, msg_huge = validate_upload_meta("huge.npy", 20 * 1024 * 1024)
    assert valid_huge is False
    assert "exceeds maximum limit" in msg_huge


def test_tensor_log_redaction():
    log_line = "Processed array([[0.1, 0.2], [0.3, 0.4]]) successfully"
    redacted = redact_tensor_repr(log_line)
    assert "array([<REDACTED_TENSOR_VALUES>])" in redacted
    assert "0.1" not in redacted


def test_file_integrity_verification(tmp_path):
    test_file = tmp_path / "data.txt"
    test_file.write_text("evidence_net_integrity_test\n", encoding="utf-8")

    actual_hash = compute_sha256(test_file)
    assert verify_file_integrity(test_file, actual_hash) is True
    assert verify_file_integrity(test_file, f"sha256:{actual_hash}") is True
    assert verify_file_integrity(test_file, "wrong_hash") is False
