"""Resolution of the official local dataset directories.

Paths are resolved from the execution-file parent directory or from explicit
environment variables (``TRAIN_DATA_DIR`` / ``TEST_NOISY_LR_DIR``), never from
an assumed current working directory. See ``docs/data-card.md`` and
``docs/dataset-manifest-contract.md``.

Resolution order:

1. Explicit environment variables (highest priority).
2. ``.env`` file in the repository root (values do not override the real
   environment).
3. Discovery: walk up from the repository root to the directory containing
   ``EVIDENCE_NET_EXECUTION.md`` (or ``EVIDENCE_NET_EXECUTION_WITH_DATASET.md``)
   and use ``<parent>/train`` and ``<parent>/Test_NoisyLR``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

EXECUTION_FILE_NAMES = (
    "EVIDENCE_NET_EXECUTION_WITH_DATASET.md",
    "EVIDENCE_NET_EXECUTION.md",
)

TRAIN_DIR_ENV = "TRAIN_DATA_DIR"
TEST_NOISY_LR_DIR_ENV = "TEST_NOISY_LR_DIR"

REPO_ROOT = Path(__file__).resolve().parents[3]


class DatasetPathError(RuntimeError):
    """Raised when official dataset paths cannot be resolved or are invalid."""


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved official local dataset directories."""

    train_dir: Path
    test_noisylr_dir: Path
    execution_parent: Path
    source: str

    def as_dict(self) -> dict[str, str]:
        return {
            "train_dir": str(self.train_dir),
            "test_noisylr_dir": str(self.test_noisylr_dir),
            "execution_parent": str(self.execution_parent),
            "source": self.source,
        }


def load_dotenv(path: Path) -> dict[str, str]:
    """Parse a minimal ``KEY=VALUE`` env file. Real environment wins."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def find_execution_parent(start: Path) -> Path | None:
    """Return the ancestor of ``start`` containing an execution file, if any."""
    for candidate in (start, *start.parents):
        for name in EXECUTION_FILE_NAMES:
            if (candidate / name).is_file():
                return candidate
    return None


def resolve_dataset_paths(
    env: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> DatasetPaths:
    """Resolve the official dataset directories per the documented policy."""
    repo_root = repo_root or REPO_ROOT
    dotenv = load_dotenv(repo_root / ".env")
    environment = dict(os.environ)
    if env is not None:
        environment.update(env)
    merged = {**dotenv, **environment}

    both_from_env = bool(merged.get(TRAIN_DIR_ENV, "").strip()) and bool(
        merged.get(TEST_NOISY_LR_DIR_ENV, "").strip()
    )
    # The execution file is only required when at least one dataset directory
    # must be discovered from the execution-file parent; explicit env
    # overrides for both directories stand on their own.
    execution_parent = find_execution_parent(repo_root) if not both_from_env else None
    errors: list[str] = []

    def pick(var: str, default_name: str) -> tuple[Path | None, str]:
        raw = merged.get(var, "").strip()
        if raw:
            return Path(raw), "env"
        if execution_parent is not None:
            return execution_parent / default_name, "execution-parent"
        errors.append(f"{var} is not set and no execution file was found above {repo_root}")
        return None, "unresolved"

    train_dir, train_source = pick(TRAIN_DIR_ENV, "train")
    test_dir, test_source = pick(TEST_NOISY_LR_DIR_ENV, "Test_NoisyLR")
    source = train_source if train_source == test_source else f"{train_source}+{test_source}"

    if execution_parent is None and not both_from_env:
        errors.append(
            "no execution file "
            + " or ".join(EXECUTION_FILE_NAMES)
            + " found in any parent of the repository root"
        )

    if train_dir is not None and not train_dir.is_dir():
        errors.append(f"train directory not found: {train_dir}")
    if test_dir is not None and not test_dir.is_dir():
        errors.append(f"test directory not found: {test_dir}")
    if train_dir is not None and test_dir is not None and execution_parent is not None:
        same_parent = (
            train_dir.resolve().parent == execution_parent.resolve()
            and test_dir.resolve().parent == execution_parent.resolve()
        )
        # The standard layout check applies only to execution-parent discovery;
        # explicit environment overrides are trusted for the dataset location.
        if train_source == "execution-parent" and not same_parent:
            errors.append(
                "train/ and Test_NoisyLR/ must share the same parent directory as "
                "the execution file"
            )

    if errors:
        raise DatasetPathError("; ".join(errors))

    return DatasetPaths(
        train_dir=train_dir or Path(),
        test_noisylr_dir=test_dir or Path(),
        execution_parent=execution_parent or Path(),
        source=source,
    )
