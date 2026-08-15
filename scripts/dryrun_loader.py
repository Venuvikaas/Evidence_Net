"""Dry-run loader for the isolated test inputs.

Reads every supported ``Test_NoisyLR/`` input through the raw-preserving
loader without running any model or final evaluation. Fails if any input
cannot be read. Run from the repository root::

    python scripts/dryrun_loader.py [--dataset test]
"""

from __future__ import annotations

import argparse
import sys

from evidence_net.data.inventory import inventory_directory
from evidence_net.data.loaders import DatasetLoadError, load_npy
from evidence_net.data.paths import DatasetPathError, resolve_dataset_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("train", "test"), default="test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        datasets = resolve_dataset_paths()
    except DatasetPathError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    root = datasets.test_noisylr_dir if args.dataset == "test" else datasets.train_dir
    entries = inventory_directory(root)
    failures = 0
    for entry in entries:
        try:
            load_npy(root / entry.relative_path)
        except DatasetLoadError as exc:
            failures += 1
            print(f"FAIL {entry.relative_path}: {exc}", file=sys.stderr)
    print(f"dry-run loader: {len(entries)} files read, {failures} failures ({args.dataset})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
