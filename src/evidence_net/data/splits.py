"""Deterministic grouped development splits.

Splits are assigned by source group using a documented, seeded hash of the
sample id, so they are reproducible and do not leak across repeated
structures. ``Test_NoisyLR/`` inputs can never enter a development split:
the split builder accepts only files from the train source manifest and the
isolation guard raises if any test-final path is present.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence

from evidence_net.data.manifests import FileEntry

DEVELOPMENT_SPLITS = ("train", "validation", "calibration", "heldout-source", "heldout-degradation")
TEST_FINAL_SPLIT = "test-final"

DEFAULT_FRACTIONS: dict[str, float] = {
    "train": 0.80,
    "validation": 0.10,
    "calibration": 0.05,
    "heldout-source": 0.05,
    "heldout-degradation": 0.0,
}


class SplitError(RuntimeError):
    """Raised when a development split violates the isolation or grouping rules."""


def bucket(sample_id: str, seed: int, buckets: int) -> int:
    """Deterministic bucket for a sample id under a documented seed."""
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()
    return int(digest[:8], 16) % buckets


def assign_splits(
    sample_ids: Sequence[str],
    *,
    fractions: Mapping[str, float] | None = None,
    seed: int = 0,
) -> dict[str, str]:
    """Assign each sample id to a development split deterministically."""
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    unknown = set(fractions) - set(DEVELOPMENT_SPLITS)
    if unknown:
        raise SplitError(f"unknown split labels in fractions: {sorted(unknown)}")
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise SplitError(f"fractions must sum to 1.0, got {total}")

    thresholds: dict[str, int] = {}
    cumulative = 0.0
    for label in DEVELOPMENT_SPLITS:
        cumulative += fractions[label]
        thresholds[label] = round(cumulative * 1000)

    assignments: dict[str, str] = {}
    for sample_id in sample_ids:
        value = bucket(sample_id, seed, 1000)
        for label in DEVELOPMENT_SPLITS:
            if value < thresholds[label]:
                assignments[sample_id] = label
                break
        else:  # pragma: no cover - thresholds cover [0, 1000)
            raise SplitError(f"sample {sample_id} fell outside split thresholds")
    return assignments


def assert_no_test_paths(entries: Sequence[FileEntry], test_relative_paths: set[str]) -> None:
    """Fail if any entry's relative path belongs to the isolated test source."""
    for entry in entries:
        if entry.relative_path in test_relative_paths:
            raise SplitError(
                f"isolated test path entered a development manifest: {entry.relative_path}"
            )


def build_split_manifest(
    train_entries: Sequence[FileEntry],
    *,
    sample_id_fn: Callable[[FileEntry], str],
    fractions: Mapping[str, float] | None = None,
    seed: int = 0,
) -> tuple[dict[str, str], dict[str, int]]:
    """Build ``{sample_id: split_label}`` and per-split counts from train files.

    ``sample_id_fn`` maps a ``FileEntry`` to its sample id.
    """
    ids = sorted({sample_id_fn(entry) for entry in train_entries})
    assignments = assign_splits(ids, fractions=fractions, seed=seed)
    counts = Counter(assignments.values())
    return assignments, dict(counts)
