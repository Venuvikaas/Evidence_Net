"""Raw-preserving dataset loaders.

Loaders preserve the raw tensor and its metadata exactly as read from disk.
They validate the tensor contract (``docs/tensor-contract.md``) and raise
``DatasetLoadError`` with a clear message on corrupted or unsupported inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_EXTENSIONS = (".npy",)
MAX_DIMENSIONS = 3


class DatasetLoadError(ValueError):
    """Raised when a sample cannot be loaded or violates the tensor contract."""


def load_npy(path: Path) -> np.ndarray:
    """Load a raw ``.npy`` tensor, preserving values and dtype.

    Raises ``DatasetLoadError`` for missing files, unsupported formats,
    object arrays, non-floating dtypes, unexpected dimensionality, or
    non-finite values.
    """
    if path.suffix not in SUPPORTED_EXTENSIONS:
        raise DatasetLoadError(
            f"unsupported format: {path.suffix} (supported: {SUPPORTED_EXTENSIONS})"
        )
    if not path.is_file():
        raise DatasetLoadError(f"file not found: {path}")
    try:
        array = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise DatasetLoadError(f"failed to load {path}: {exc}") from exc
    if not isinstance(array, np.ndarray):
        raise DatasetLoadError(f"{path} does not contain a numpy array")
    if array.dtype == object or array.dtype.kind == "O":
        raise DatasetLoadError(f"{path} is an object array (pickle disallowed)")
    if not np.issubdtype(array.dtype, np.floating):
        raise DatasetLoadError(f"{path} has dtype {array.dtype}; expected a floating dtype")
    if array.ndim < 2 or array.ndim > MAX_DIMENSIONS:
        raise DatasetLoadError(
            f"{path} has ndim={array.ndim}; expected 2D or 3D per tensor contract"
        )
    if not np.isfinite(array).all():
        raise DatasetLoadError(f"{path} contains non-finite values")
    return array


def tensor_metadata(array: np.ndarray) -> dict[str, Any]:
    """Metadata describing a loaded tensor: shape, channels, dtype, range."""
    channels = 1 if array.ndim == 2 else int(array.shape[0])
    return {
        "dimensions": list(array.shape),
        "channels": channels,
        "dtype": str(array.dtype),
        "range": [float(array.min()), float(array.max())],
        "finite": bool(np.isfinite(array).all()),
    }


def inspect_file(path: Path) -> dict[str, Any] | None:
    """Read-only metadata for a file; returns ``None`` when unreadable."""
    try:
        array = load_npy(path)
    except DatasetLoadError:
        return None
    return tensor_metadata(array)
