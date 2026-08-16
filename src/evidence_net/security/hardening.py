"""Security hardening and path validation utilities (Phase 16).

Provides upload validation, path traversal prevention, filename sanitization,
and raw tensor log redaction.
"""

from __future__ import annotations

import re
from pathlib import Path

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".npy", ".npz", ".png", ".jpg", ".jpeg", ".json", ".yaml", ".pt"}


def sanitize_filename(filename: str) -> str:
    """Sanitize filename preventing path traversal characters or unsafe symbols."""
    clean = Path(filename).name
    clean = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", clean)
    return clean or "unnamed_file"


def validate_file_path(base_dir: Path | str, target_path: Path | str) -> Path:
    """Ensure target path resolves within base_dir (prevents directory traversal)."""
    base = Path(base_dir).resolve()
    target = Path(target_path).resolve()

    try:
        target.relative_to(base)
    except ValueError as exc:
        msg = f"Path traversal attempt detected: '{target_path}' outside '{base_dir}'"
        raise ValueError(msg) from exc

    return target


def validate_upload_meta(filename: str, size_bytes: int) -> tuple[bool, str]:
    """Validate uploaded file extension and size limits."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return (
            False,
            f"File extension '{ext}' is not allowed. Must be one of {sorted(ALLOWED_EXTENSIONS)}",
        )
    if size_bytes > MAX_FILE_SIZE_BYTES:
        return (
            False,
            f"File size ({size_bytes} bytes) exceeds maximum limit of {MAX_FILE_SIZE_BYTES} bytes",
        )
    return True, "valid"


def redact_tensor_repr(text: str) -> str:
    """Filter raw float arrays from log messages to prevent raw tensor exposure."""
    # Redact array contents in strings like array([...]) or tensor([...])
    redacted = re.sub(r"(array|tensor)\(\[[\s\S]*?\]\)", r"\1([<REDACTED_TENSOR_VALUES>])", text)
    return redacted
