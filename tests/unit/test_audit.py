"""Tests for dataset audit computations."""

from pathlib import Path

import numpy as np
import pytest

from evidence_net.data.audit import (
    alignment_audit,
    compatibility_summary,
    degradation_summary,
    exact_duplicate_groups,
    export_alignment_examples,
    near_duplicate_signature,
    range_summary,
)
from evidence_net.data.manifests import FileEntry


def _entry(path: str, sha: str, *, readable: bool = True, dims=None, dtype="float32", rng=None):
    return FileEntry(
        relative_path=path,
        extension=".npy",
        byte_size=1,
        sha256=sha,
        readable=readable,
        dimensions=dims,
        channels=1 if dims and len(dims) == 2 else None,
        dtype=dtype,
        range=rng,
    )


def test_range_summary_counts_unreadable() -> None:
    entries = [
        _entry("a.npy", "a" * 64, dims=[8, 8], rng=[0.0, 1.0]),
        _entry("b.npy", "b" * 64, readable=False),
    ]
    summary = range_summary(entries)
    assert summary["n_readable"] == 1
    assert summary["n_unreadable"] == 1
    assert summary["shape_counts"] == {"(8, 8)": 1}


def test_compatibility_compares_sets_not_counts() -> None:
    train = [
        _entry(
            f"train/{i:06d}.npy", f"{i:02d}" * 32, dims=[128, 128], dtype="float32", rng=[0.0, 1.0]
        )
        for i in range(50)
    ]
    test = [
        _entry(
            f"test/{i:06d}.npy", f"{i:02d}" * 32, dims=[128, 128], dtype="float32", rng=[0.0, 1.0]
        )
        for i in range(5)
    ]
    summary = compatibility_summary(train, test)
    assert summary["compatible"] is True


def test_compatibility_detects_shape_mismatch() -> None:
    train = [_entry("a.npy", "a" * 64, dims=[128, 128], dtype="float32")]
    test = [_entry("b.npy", "b" * 64, dims=[256, 256], dtype="float32")]
    summary = compatibility_summary(train, test)
    assert summary["compatible"] is False


def test_exact_duplicate_groups(tmp_path: Path) -> None:
    arr = np.zeros((4, 4), dtype=np.float32)
    p1 = tmp_path / "one.npy"
    p2 = tmp_path / "two.npy"
    np.save(p1, arr)
    np.save(p2, arr)
    entries = [
        _entry("one.npy", "same" * 16),
        _entry("two.npy", "same" * 16),
        _entry("three.npy", "diff" * 16),
    ]
    groups = exact_duplicate_groups(entries)
    assert len(groups) == 1
    assert set(groups["same" * 16]) == {"one.npy", "two.npy"}


def test_near_duplicate_signature_is_content_based(tmp_path: Path) -> None:
    a = np.random.default_rng(0).random((64, 64), dtype=np.float32)
    b = a.copy()
    p1 = tmp_path / "a.npy"
    p2 = tmp_path / "b.npy"
    np.save(p1, a)
    np.save(p2, b)
    assert near_duplicate_signature(p1) == near_duplicate_signature(p2)


def test_alignment_audit_finds_dominant_offset(tmp_path: Path) -> None:
    gt = np.random.default_rng(0).random((32, 32), dtype=np.float32)
    # NoisyLR = GT subsampled at offset (1, 0) plus a little noise
    noisy = gt[1::2, 0::2] + 0.001 * np.random.default_rng(1).random((16, 16))
    gt_path = tmp_path / "gt.npy"
    noisy_path = tmp_path / "noisy.npy"
    np.save(gt_path, gt)
    np.save(noisy_path, noisy)
    result = alignment_audit([(noisy_path, gt_path)])
    assert result["n_pairs"] == 1
    assert result["best_offsets"] == {"1,0": 1}


def test_degradation_summary_reports_resolution_ratio(tmp_path: Path) -> None:
    gt = np.random.default_rng(0).random((32, 32), dtype=np.float32)
    noisy = gt.reshape(16, 2, 16, 2).mean(axis=(1, 3))
    gt_path = tmp_path / "gt.npy"
    noisy_path = tmp_path / "noisy.npy"
    np.save(gt_path, gt)
    np.save(noisy_path, noisy)
    result = degradation_summary([(noisy_path, gt_path)])
    assert result["resolution_ratio_gt_over_noisy"]["ratio_mean"] == pytest.approx(2.0)
    assert result["n_pairs"] == 1


def test_export_alignment_examples_writes_artifacts(tmp_path: Path) -> None:
    gt = np.random.default_rng(0).random((32, 32), dtype=np.float32)
    noisy = gt[::2, ::2]
    gt_path = tmp_path / "gt.npy"
    noisy_path = tmp_path / "noisy.npy"
    np.save(gt_path, gt)
    np.save(noisy_path, noisy)
    out = tmp_path / "artifacts"
    names = export_alignment_examples([(noisy_path, gt_path)], out, n=2)
    assert len(names) == 1
    assert (out / "alignment_example_0_input.npy").is_file()
    assert (out / "alignment_example_0_target.npy").is_file()
