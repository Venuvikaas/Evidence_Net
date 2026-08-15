"""Verify frozen source manifests: schema conformance and dataset hashes.

Usage (from the repository root)::

    python scripts/verify_manifests.py [manifest.json ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evidence_net.data.manifests import (
    ManifestValidationError,
    validate_source_manifest,
    verify_dataset_hash,
)

MANIFESTS_DIR = Path(__file__).resolve().parent.parent / "data" / "manifests"

DEFAULT_MANIFESTS = (
    "official-train-source-v1.json",
    "official-test-noisylr-source-v1.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifests", nargs="*", help="manifest files (default: both source manifests)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = args.manifests or list(DEFAULT_MANIFESTS)
    ok = True
    for name in names:
        path = Path(name) if Path(name).is_file() else MANIFESTS_DIR / name
        if not path.is_file():
            print(f"FAIL: manifest not found: {path}", file=sys.stderr)
            ok = False
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            allow_dev = data.get("dataset_id") != "official-test-noisylr-source-v1"
            validate_source_manifest(data, allow_development_labels=allow_dev)
            hash_ok = verify_dataset_hash(data)
        except (ManifestValidationError, KeyError, TypeError) as exc:
            print(f"FAIL: {path.name}: {exc}", file=sys.stderr)
            ok = False
            continue
        print(
            f"[{'PASS' if hash_ok else 'FAIL'}] {path.name} "
            f"(hash {data.get('dataset_hash', '')[:16]}...)"
            if hash_ok
            else f"[FAIL] {path.name}: dataset_hash mismatch"
        )
        ok &= hash_ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
