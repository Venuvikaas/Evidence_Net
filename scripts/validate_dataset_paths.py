"""Validate the official local dataset paths.

Checks that the execution file, ``train/``, and ``Test_NoisyLR/`` share the
same parent directory and that both dataset directories exist. Fails with a
clear message otherwise. Run from the repository root::

    python scripts/validate_dataset_paths.py
"""

from __future__ import annotations

import sys

from evidence_net.data.paths import (
    EXECUTION_FILE_NAMES,
    REPO_ROOT,
    DatasetPathError,
    find_execution_parent,
    resolve_dataset_paths,
)


def main() -> int:
    print("EVIDENCE-Net official dataset path validation")
    try:
        datasets = resolve_dataset_paths()
    except DatasetPathError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    checks = [
        ("train directory exists", datasets.train_dir.is_dir()),
        ("Test_NoisyLR directory exists", datasets.test_noisylr_dir.is_dir()),
    ]
    execution_parent = find_execution_parent(REPO_ROOT)
    if execution_parent is not None:
        execution_name = next(
            (name for name in EXECUTION_FILE_NAMES if (execution_parent / name).is_file()),
            "unknown",
        )
        checks += [
            ("execution file present in parent", execution_parent is not None),
            (
                "train shares parent with execution file",
                datasets.train_dir.resolve().parent == execution_parent.resolve(),
            ),
            (
                "Test_NoisyLR shares parent with execution file",
                datasets.test_noisylr_dir.resolve().parent == execution_parent.resolve(),
            ),
        ]
        print(f"execution parent: {execution_parent} ({execution_name})")
    else:
        print(
            "[INFO] execution file not found above the repository root; "
            "standard-layout check skipped (explicit dataset env override in use)"
        )

    ok = True
    for label, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")
        ok &= passed
    print(f"train dir:        {datasets.train_dir}")
    print(f"test dir:         {datasets.test_noisylr_dir}")
    print(f"resolution:       {datasets.source}")
    if not ok:
        print("Dataset path validation FAILED.", file=sys.stderr)
        return 1
    print("Dataset path validation OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
