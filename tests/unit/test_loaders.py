"""Tensor-contract and loader robustness tests."""

from pathlib import Path

import numpy as np
import pytest

from evidence_net.data.loaders import (
    DatasetLoadError,
    inspect_file,
    load_npy,
)


def _write_npy(path: Path, array: np.ndarray) -> Path:
    np.save(path, array)
    return path


def test_load_preserves_raw_values_and_dtype(tmp_path: Path) -> None:
    array = np.random.default_rng(0).random((8, 8), dtype=np.float32)
    path = _write_npy(tmp_path / "sample.npy", array)
    loaded = load_npy(path)
    assert loaded.dtype == array.dtype
    assert loaded.shape == array.shape
    np.testing.assert_array_equal(loaded, array)


def test_tensor_metadata_contract(tmp_path: Path) -> None:
    array = np.arange(64, dtype=np.float32).reshape(8, 8) / 64.0
    path = _write_npy(tmp_path / "meta.npy", array)
    metadata = inspect_file(path)
    assert metadata is not None
    assert metadata["dimensions"] == [8, 8]
    assert metadata["channels"] == 1
    assert metadata["dtype"] == "float32"
    assert metadata["range"] == [0.0, pytest.approx(63.0 / 64.0)]


def test_3d_channel_first_contract(tmp_path: Path) -> None:
    array = np.zeros((3, 8, 8), dtype=np.float32)
    path = _write_npy(tmp_path / "channels.npy", array)
    metadata = inspect_file(path)
    assert metadata is not None
    assert metadata["channels"] == 3
    assert metadata["dimensions"] == [3, 8, 8]


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetLoadError, match="not found"):
        load_npy(tmp_path / "missing.npy")


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(DatasetLoadError, match="unsupported format"):
        load_npy(path)


def test_corrupted_npy_raises_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.npy"
    path.write_bytes(b"not a real npy file at all")
    with pytest.raises(DatasetLoadError, match="failed to load"):
        load_npy(path)


def test_non_floating_dtype_rejected(tmp_path: Path) -> None:
    array = np.zeros((8, 8), dtype=np.int32)
    path = _write_npy(tmp_path / "int.npy", array)
    with pytest.raises(DatasetLoadError, match="floating"):
        load_npy(path)


def test_non_finite_values_rejected(tmp_path: Path) -> None:
    array = np.full((8, 8), np.nan, dtype=np.float32)
    path = _write_npy(tmp_path / "nan.npy", array)
    with pytest.raises(DatasetLoadError, match="non-finite"):
        load_npy(path)


def test_inspect_file_returns_none_for_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "broken.npy"
    path.write_bytes(b"garbage")
    assert inspect_file(path) is None


def test_object_array_rejected(tmp_path: Path) -> None:
    path = tmp_path / "object.npy"
    np.save(path, np.array([["a", "b"], ["c", "d"]], dtype=object))
    # numpy refuses object arrays under allow_pickle=False; the loader must
    # surface this as a DatasetLoadError (pickle / object arrays disallowed).
    with pytest.raises(DatasetLoadError, match="failed to load|object"):
        load_npy(path)
