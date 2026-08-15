"""Generate a read-only inventory and freeze a source manifest.

Usage (from the repository root)::

    python scripts/inventory_dataset.py --dataset train
    python scripts/inventory_dataset.py --dataset test

Writes ``data/manifests/official-train-source-v1.json`` or
``data/manifests/official-test-noisylr-source-v1.json``. The test manifest is
kept free of development labels (split labels and roles).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from evidence_net.data.inventory import inventory_directory
from evidence_net.data.manifests import (
    HASH_ALGORITHM,
    MANIFEST_SCHEMA_VERSION,
    SourceManifest,
    index_samples,
    write_manifest,
)
from evidence_net.data.pairing import discover_train_structure
from evidence_net.data.paths import DatasetPathError, resolve_dataset_paths

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("train", "test"), required=True, help="which dataset to inventory"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        datasets = resolve_dataset_paths()
    except DatasetPathError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    structure: dict[str, str] | None = None
    if args.dataset == "train":
        root = datasets.train_dir
        dataset_id = "official-train-source-v1"
        try:
            gt_dir, noisy_dir = discover_train_structure(root)
            structure = {
                "gt_dir": gt_dir.relative_to(root).as_posix(),
                "noisy_lr_dir": noisy_dir.relative_to(root).as_posix(),
            }
        except Exception as exc:
            print(f"FAIL: train structure discovery: {exc}", file=sys.stderr)
            return 1
        provenance = {
            "source": "official local directory (KLA problem statement)",
            "permitted_uses": [
                "development",
                "training",
                "validation",
                "calibration",
                "source-held-out robustness",
            ],
            "restrictions": [
                "never commit dataset files to git",
                "Test_NoisyLR is isolated from all development decisions",
            ],
        }
    else:
        root = datasets.test_noisylr_dir
        dataset_id = "official-test-noisylr-source-v1"
        structure = None
        provenance = {
            "source": "official local directory (KLA problem statement)",
            "permitted_uses": ["final evaluation inference only"],
            "restrictions": [
                "never commit dataset files to git",
                "must not influence training, validation, calibration, "
                "hyperparameter search, or threshold selection",
                "kept free of development labels and metrics",
            ],
        }

    print(f"Inventorizing {dataset_id} at {root} ...")
    entries = inventory_directory(root)
    index_samples(entries)  # deterministic discovery; raises on duplicate keys
    manifest = SourceManifest(
        manifest_version=MANIFEST_SCHEMA_VERSION,
        dataset_id=dataset_id,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        source_root=str(root.resolve()),
        hash_algorithm=HASH_ALGORITHM,
        dataset_hash="",  # computed below
        provenance=provenance,
        grouping={
            "source_group_field": "sample-id",
            "hierarchy": ["source_group", "acquisition", "sample"],
            "note": (
                "no acquisition/session metadata is present in the official "
                "directory; each sample is its own source unit (see "
                "docs/grouping-and-splits.md)"
            ),
        },
        structure=structure,
        files=entries,
    )
    manifest.dataset_hash = manifest.compute_dataset_hash()

    out_path = MANIFESTS_DIR / f"{dataset_id}.json"
    write_manifest(out_path, manifest)
    print(f"Wrote {out_path} ({len(entries)} files, hash {manifest.dataset_hash[:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
