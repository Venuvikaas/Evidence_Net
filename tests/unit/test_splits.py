"""Tests for deterministic grouped splits and Test_NoisyLR isolation."""

import pytest

from evidence_net.data.manifests import FileEntry
from evidence_net.data.splits import (
    DEFAULT_FRACTIONS,
    SplitError,
    assert_no_test_paths,
    assign_splits,
    bucket,
)


def test_splits_are_deterministic() -> None:
    ids = [f"{i:06d}" for i in range(200)]
    first = assign_splits(ids, seed=0)
    second = assign_splits(ids, seed=0)
    assert first == second
    assert set(first) == set(ids)


def test_splits_respect_fractions() -> None:
    from collections import Counter

    ids = [f"{i:06d}" for i in range(10000)]
    assignments = assign_splits(ids, seed=0)
    counts = Counter(assignments.values())
    total = len(ids)
    for label, fraction in DEFAULT_FRACTIONS.items():
        expected = fraction * total
        actual = counts.get(label, 0)
        assert abs(actual - expected) < 0.02 * total, (label, actual, expected)


def test_bucket_is_stable() -> None:
    assert bucket("000000", seed=0, buckets=1000) == bucket("000000", seed=0, buckets=1000)
    assert 0 <= bucket("000000", seed=0, buckets=1000) < 1000


def test_unknown_split_label_rejected() -> None:
    with pytest.raises(SplitError, match="unknown split"):
        assign_splits(["a"], fractions={"train": 1.0, "bogus": 0.0})


def test_fractions_must_sum_to_one() -> None:
    with pytest.raises(SplitError, match="sum to 1.0"):
        assign_splits(["a"], fractions={"train": 0.5})


def _entry(path: str) -> FileEntry:
    return FileEntry(
        relative_path=path,
        extension=".npy",
        byte_size=1,
        sha256="a" * 64,
        readable=True,
    )


def test_isolation_guard_rejects_test_path() -> None:
    test_paths = {"NoisyLR/000000.npy", "NoisyLR/000001.npy"}
    entries = [_entry("NoisyLR/000000.npy")]
    with pytest.raises(SplitError, match="isolated test path"):
        assert_no_test_paths(entries, test_paths)


def test_isolation_guard_accepts_train_paths() -> None:
    test_paths = {"NoisyLR/000000.npy"}
    entries = [_entry("train/GT/000000.npy"), _entry("train/NoisyLR/000000.npy")]
    assert_no_test_paths(entries, test_paths)  # no exception
