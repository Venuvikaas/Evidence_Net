"""Build deterministic grouped development splits from the official train
manifest and freeze them.

Usage (from the repository root)::

    python scripts/build_splits.py [--seed 0]

Writes ``data/manifests/dataset-splits-v1.json`` containing the split
assignment per sample id plus per-split counts. Only files from the official
train source manifest are eligible; the isolated test manifest is never
consulted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from evidence_net.data.manifests import FileEntry, SourceManifest
from evidence_net.data.pairing import discover_train_structure
from evidence_net.data.paths import DatasetPathError, resolve_dataset_paths
from evidence_net.data.splits import (
    DEFAULT_FRACTIONS,
    assert_no_test_paths,
    assign_splits,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
TRAIN_MANIFEST = MANIFESTS_DIR / "official-train-source-v1.json"
TEST_MANIFEST = MANIFESTS_DIR / "official-test-noisylr-source-v1.json"
SPLITS_MANIFEST = MANIFESTS_DIR / "dataset-splits-v1.json"
DATASET_MANIFEST_V1 = MANIFESTS_DIR / "dataset-manifest-v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="documented split seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        datasets = resolve_dataset_paths()
    except DatasetPathError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if not TRAIN_MANIFEST.is_file():
        print(
            f"FAIL: {TRAIN_MANIFEST} missing; run scripts/inventory_dataset.py --dataset train",
            file=sys.stderr,
        )
        return 1

    train = SourceManifest.from_dict(json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8")))
    if not TEST_MANIFEST.is_file():
        print(
            f"FAIL: {TEST_MANIFEST} missing; run scripts/inventory_dataset.py --dataset test",
            file=sys.stderr,
        )
        return 1
    test_manifest = SourceManifest.from_dict(json.loads(TEST_MANIFEST.read_text(encoding="utf-8")))
    assert_no_test_paths(train.files, {e.relative_path for e in test_manifest.files})

    _, noisy_dir = discover_train_structure(datasets.train_dir)
    noisy_base = noisy_dir.relative_to(datasets.train_dir).as_posix()

    noisy_entries = [e for e in train.files if e.relative_path.startswith(noisy_base)]

    def sample_id(entry: FileEntry) -> str:
        return entry.relative_path.rsplit("/", 1)[-1].split(".")[0]

    ids = sorted({sample_id(e) for e in noisy_entries})
    assignments = assign_splits(ids, fractions=DEFAULT_FRACTIONS, seed=args.seed)

    counts = dict(Counter(assignments.values()))
    split_doc = {
        "manifest_version": "1",
        "dataset_id": "dataset-splits-v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_manifest": "official-train-source-v1",
        "seed": args.seed,
        "fractions": DEFAULT_FRACTIONS,
        "grouping_note": (
            "splits assigned per sample id by seeded sha256 bucket; no "
            "acquisition/session metadata exists in the official train "
            "directory, so each sample is its own source unit"
        ),
        "isolation": {
            "test_source": "official-test-noisylr-source-v1",
            "test_final_entries": 0,
            "rule": "Test_NoisyLR must never enter training, validation, "
            "calibration, hyperparameter-search, or threshold-selection "
            "manifests",
        },
        "n_samples": len(ids),
        "counts": counts,
        "assignments": assignments,
    }
    payload = json.dumps(split_doc, indent=2, sort_keys=True) + "\n"
    SPLITS_MANIFEST.write_text(payload, encoding="utf-8")

    dataset_manifest = {
        "manifest_version": "1",
        "dataset_id": "dataset-manifest-v1",
        "created_at": split_doc["created_at"],
        "source_manifests": {
            "official-train-source-v1": {
                "file": TRAIN_MANIFEST.name,
                "sha256": _sha256(TRAIN_MANIFEST),
            },
            "official-test-noisylr-source-v1": {
                "file": TEST_MANIFEST.name,
                "sha256": _sha256(TEST_MANIFEST),
            },
        },
        "splits_manifest": {
            "file": SPLITS_MANIFEST.name,
            "sha256": _sha256(SPLITS_MANIFEST),
        },
        "n_samples": len(ids),
        "split_counts": counts,
        "split_seed": args.seed,
        "isolation": split_doc["isolation"],
    }
    DATASET_MANIFEST_V1.write_text(
        json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Wrote {SPLITS_MANIFEST}")
    print(f"Wrote {DATASET_MANIFEST_V1}")
    print("counts:", counts)
    print(f"n_samples: {len(ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
