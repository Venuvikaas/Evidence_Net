"""Validate the EVIDENCE-Net development environment.

Checks the Python version, required packages, the importable package, the
required repository directories, and optional compute devices. Exits non-zero
on failure. Run from the repository root::

    python scripts/check_env.py
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_PACKAGES = ("numpy", "yaml", "torch")

REQUIRED_DIRECTORIES = (
    "configs",
    "configs/data",
    "configs/modality",
    "configs/model",
    "configs/support_definition",
    "configs/calibration",
    "configs/decision_policy",
    "configs/experiments",
    "data/manifests",
    "data/fixtures",
    "docs",
    "runs",
    "artifacts",
    "src/evidence_net",
    "tests",
)

MIN_PYTHON = (3, 10)


def check_python_version() -> bool:
    """Check the running interpreter satisfies the minimum Python version."""
    ok = sys.version_info >= MIN_PYTHON
    minimum = ".".join(map(str, MIN_PYTHON))
    print(f"[{'PASS' if ok else 'FAIL'}] python {sys.version.split()[0]} (need >= {minimum})")
    return ok


def check_packages() -> bool:
    """Check required runtime packages are importable."""
    ok = True
    for name in REQUIRED_PACKAGES:
        present = importlib.util.find_spec(name) is not None
        ok &= present
        print(f"[{'PASS' if present else 'FAIL'}] package: {name}")
    return ok


def check_package_import() -> bool:
    """Check the evidence_net package imports and exposes a version."""
    try:
        import evidence_net

        print(f"[PASS] evidence_net importable (version {evidence_net.__version__})")
        return True
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"[FAIL] evidence_net import failed: {exc}")
        return False


def check_directories() -> bool:
    """Check every required repository directory exists relative to the repo root."""
    ok = True
    for relative in REQUIRED_DIRECTORIES:
        present = (REPO_ROOT / relative).is_dir()
        ok &= present
        print(f"[{'PASS' if present else 'FAIL'}] directory: {relative}")
    return ok


def check_devices() -> bool:
    """Report the available compute device (torch required from Phase 3)."""
    if importlib.util.find_spec("torch") is None:
        print("[FAIL] torch not installed (required from Phase 3)")
        return False
    try:
        torch_module: Any = importlib.import_module("torch")
        if torch_module.cuda.is_available():
            device = "cuda"
        elif getattr(torch_module.backends, "mps", None) is not None and (
            torch_module.backends.mps.is_available()
        ):
            device = "mps"
        else:
            device = "cpu"
        print(f"[INFO] torch {torch_module.__version__} available; device: {device}")
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"[INFO] torch present but device detection failed: {exc}")
    return True


def main() -> int:
    """Run all checks and return the process exit code."""
    print(f"EVIDENCE-Net environment check on {platform.platform()}")
    checks = (
        check_python_version(),
        check_packages(),
        check_package_import(),
        check_directories(),
        check_devices(),
    )
    if all(checks):
        print("Environment OK.")
        return 0
    print("Environment check FAILED. See messages above.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
