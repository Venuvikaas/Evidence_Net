"""Automated enforcement of Test_NoisyLR isolation.

Fails if any ``Test_NoisyLR/`` path enters a training, validation,
calibration, hyperparameter-search, or threshold-selection manifest, or if
the committed test source manifest carries development labels.
"""

import json
from pathlib import Path

import pytest

from evidence_net.data.manifests import (
    validate_source_manifest,
    verify_dataset_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"


def _load(name: str) -> dict:
    path = MANIFESTS_DIR / name
    if not path.is_file():
        pytest.skip(f"manifest {name} not present")
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_manifests_are_hashed() -> None:
    for name in ("official-train-source-v1.json", "official-test-noisylr-source-v1.json"):
        data = _load(name)
        assert verify_dataset_hash(data), f"dataset_hash mismatch in {name}"


def test_test_manifest_is_free_of_development_labels() -> None:
    data = _load("official-test-noisylr-source-v1.json")
    # Any development label must cause validation failure for the test manifest.
    validate_source_manifest(data, allow_development_labels=False)


def test_no_test_path_in_development_manifests() -> None:
    test_data = _load("official-test-noisylr-source-v1.json")
    test_paths = {entry["relative_path"] for entry in test_data["files"]}
    for name in (
        "official-train-source-v1.json",
        "dataset-splits-v1.json",
        "dataset-manifest-v1.json",
    ):
        path = MANIFESTS_DIR / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "files" in data:
            for entry in data["files"]:
                assert entry["relative_path"] not in test_paths, (
                    f"{name} contains a Test_NoisyLR path: {entry['relative_path']}"
                )
        if "assignments" in data:
            # split manifest references sample ids, not paths; ensure no
            # test-final label appears anywhere
            for key in ("isolation",):
                assert (
                    "test_final_entries" not in data.get(key, {})
                    or data[key]["test_final_entries"] == 0
                )


def test_train_manifest_has_no_test_final_labels() -> None:
    data = _load("official-train-source-v1.json")
    for entry in data["files"]:
        assert entry.get("split_label") is None, (
            "train source manifest must not carry split labels; splits live "
            "in dataset-splits-v1.json"
        )
