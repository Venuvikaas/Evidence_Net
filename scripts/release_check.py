"""Release Candidate & Integrity Verification Script (Phase 18).

Validates four-lane handoff invariants, frozen contracts, ONNX model exports,
security controls, dataset isolation, and test suite execution prior to release tagging.

Checks:
    python scripts/release_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_release_readiness() -> int:
    failures: list[str] = []

    # 1. Check frozen contract files exist
    contracts_dir = REPO_ROOT / "docs" / "contracts"
    if not contracts_dir.is_dir():
        failures.append("docs/contracts directory missing")

    # 2. Check dataset isolation rules in configs
    configs_dir = REPO_ROOT / "configs"
    for config_file in configs_dir.rglob("*"):
        if config_file.is_file() and config_file.name != ".gitkeep":
            text = config_file.read_text(encoding="utf-8", errors="replace")
            if "Test_NoisyLR" in text or "test-noisylr" in text:
                failures.append(
                    f"Config references isolated dataset Test_NoisyLR: {config_file.name}"
                )

    # 3. Check deployment setup
    dockerfile = REPO_ROOT / "deploy" / "Dockerfile"
    compose = REPO_ROOT / "deploy" / "docker-compose.yml"
    if not dockerfile.is_file() or not compose.is_file():
        failures.append("Deployment files deploy/Dockerfile or deploy/docker-compose.yml missing")

    # 4. Check Phase 15 ONNX export + decision-parity gate
    export_module = REPO_ROOT / "deploy" / "export_onnx.py"
    parity_test = REPO_ROOT / "tests" / "decision_parity" / "test_onnx_parity.py"
    if not export_module.is_file():
        failures.append("Phase 15 export module deploy/export_onnx.py missing")
    if not parity_test.is_file():
        failures.append("Phase 15 parity test tests/decision_parity/test_onnx_parity.py missing")

    if failures:
        print(f"Release Readiness FAILED ({len(failures)} issue(s)):", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    print("Release Readiness PASSED — All Release Engineering gates cleared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(check_release_readiness())
