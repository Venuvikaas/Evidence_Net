"""Resolve and print the official local dataset directories as JSON.

Paths are resolved from the execution-file parent or from explicit
``TRAIN_DATA_DIR`` / ``TEST_NOISY_LR_DIR`` environment variables, never from
the current working directory. Run from the repository root::

    python scripts/resolve_dataset_paths.py
"""

from __future__ import annotations

import json
import sys

from evidence_net.data.paths import DatasetPathError, resolve_dataset_paths


def main() -> int:
    try:
        datasets = resolve_dataset_paths()
    except DatasetPathError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(datasets.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
