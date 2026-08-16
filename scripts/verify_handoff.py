"""Verify the Phase 4 four-developer handoff invariants (kill switch).

Checks, from the repository root::

    python scripts/verify_handoff.py

- every frozen handoff contract in ``docs/contracts/`` exists, is versioned,
  and is marked frozen;
- ``CODEOWNERS`` covers all four lanes (a, b, c, d);
- the fixture registry (``data/fixtures/manifest-v1.json``) is valid and
  every fixture names its schema and producer versions;
- the synthetic error/optional-field example satisfies
  ``error-and-optional-fields-v1``;
- no ``configs/`` file references ``Test_NoisyLR`` (isolation kill switch);
- the checkpoint registry pins hashes for the promoted Base and Proposal.

Exits non-zero on any failure so CI and pre-merge checks act as the handoff
kill switch.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CONTRACTS_DIR = REPO_ROOT / "docs" / "contracts"
FROZEN_CONTRACTS = (
    "dataset-v1",
    "tensor-v1",
    "metrics-v1",
    "artifacts-v1",
    "base-output-v1",
    "proposal-output-v1",
    "structural-summary-v1",
    "oracle-report-v1",
    "error-and-optional-fields-v1",
)
CODEOWNERS_PATH = REPO_ROOT / "CODEOWNERS"
FIXTURE_MANIFEST = REPO_ROOT / "data" / "fixtures" / "manifest-v1.json"
OPTIONAL_EXAMPLE = REPO_ROOT / "data" / "fixtures" / "error-and-optional-fields-v1-example.json"
CHECKPOINT_REGISTRY = REPO_ROOT / "docs" / "handoff" / "checkpoint-registry.md"
PROMOTED_CHECKPOINTS = (
    "checkpoints/train-base-gate2/best.pt",
    "checkpoints/train-proposal-gate3v2/best.pt",
)
LANE_MARKERS = ("@lane-a", "@lane-b", "@lane-c", "@lane-d")

_FAILURES: list[str] = []


def fail(message: str) -> None:
    _FAILURES.append(message)
    print(f"FAIL: {message}", file=sys.stderr)


def check_contracts() -> None:
    if not CONTRACTS_DIR.is_dir():
        fail(f"contracts directory missing: {CONTRACTS_DIR}")
        return
    for name in FROZEN_CONTRACTS:
        path = CONTRACTS_DIR / f"{name}.md"
        if not path.is_file():
            fail(f"frozen contract missing: {path.name}")
            continue
        text = path.read_text(encoding="utf-8")
        if not re.search(r"^- \*\*Name:\*\* `" + re.escape(name) + r"`", text, re.MULTILINE):
            fail(f"contract {name}: missing 'Name: `{name}`' header")
        if "- **Version:** v1" not in text:
            fail(f"contract {name}: not versioned as v1")
        if "- **Status:** frozen" not in text:
            fail(f"contract {name}: not marked frozen")


def check_codeowners() -> None:
    if not CODEOWNERS_PATH.is_file():
        fail("CODEOWNERS missing")
        return
    text = CODEOWNERS_PATH.read_text(encoding="utf-8")
    for marker in LANE_MARKERS:
        if marker not in text:
            fail(f"CODEOWNERS missing lane marker: {marker}")


def check_fixture_registry() -> None:
    if not FIXTURE_MANIFEST.is_file():
        fail("fixture registry missing: data/fixtures/manifest-v1.json")
        return
    try:
        data = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"fixture registry is not valid JSON: {exc}")
        return
    if data.get("schema") != "fixtures-v1":
        fail("fixture registry: schema must be fixtures-v1")
    fixtures = data.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        fail("fixture registry: fixtures list missing or empty")
        return
    for fixture in fixtures:
        for field in ("name", "schema_version", "producer_version", "kind"):
            if not fixture.get(field):
                fail(f"fixture {fixture.get('name', '?')}: missing '{field}'")
        if fixture.get("kind") not in ("synthetic", "real"):
            fail(f"fixture {fixture.get('name', '?')}: kind must be synthetic or real")


def check_optional_example() -> None:
    if not OPTIONAL_EXAMPLE.is_file():
        fail("optional-field example fixture missing")
        return
    try:
        data = json.loads(OPTIONAL_EXAMPLE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"optional-field example is not valid JSON: {exc}")
        return
    if data.get("schema") != "error-and-optional-fields-v1":
        fail("optional-field example: schema must be error-and-optional-fields-v1")
    if "frozen_fields" not in data:
        fail("optional-field example: missing frozen_fields")
    if "optional_fields_absent" not in data:
        fail("optional-field example: missing optional_fields_absent")
    error = data.get("error_payload", {})
    if not error.get("error_code") or not error.get("message"):
        fail("optional-field example: error_payload needs error_code and message")


def check_config_isolation() -> None:
    configs_dir = REPO_ROOT / "configs"
    if not configs_dir.is_dir():
        fail("configs directory missing")
        return
    for path in sorted(configs_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == ".gitkeep":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "Test_NoisyLR" in text or "test-noisylr" in text:
            fail(f"config references Test_NoisyLR (isolation kill switch): {path}")


def check_checkpoint_registry() -> None:
    if not CHECKPOINT_REGISTRY.is_file():
        fail("checkpoint registry missing: docs/handoff/checkpoint-registry.md")
        return
    text = CHECKPOINT_REGISTRY.read_text(encoding="utf-8")
    for checkpoint in PROMOTED_CHECKPOINTS:
        if checkpoint not in text:
            fail(f"checkpoint registry missing entry: {checkpoint}")
    if not re.search(r"[0-9a-f]{64}", text):
        fail("checkpoint registry missing a sha256 hash")


def main() -> int:
    checks = (
        check_contracts,
        check_codeowners,
        check_fixture_registry,
        check_optional_example,
        check_config_isolation,
        check_checkpoint_registry,
    )
    for check in checks:
        check()
    if _FAILURES:
        print(f"Handoff verification FAILED ({len(_FAILURES)} issue(s)).", file=sys.stderr)
        return 1
    print(f"Handoff verification PASSED ({len(FROZEN_CONTRACTS)} contracts frozen).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
