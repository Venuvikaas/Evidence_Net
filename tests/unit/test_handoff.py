"""Phase 4 handoff invariants: frozen contracts, lane ownership, fixtures,
isolation, and checkpoint pinning. Mirrors scripts/verify_handoff.py so the
handoff kill switch is enforced both as a unit test and in CI.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

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


def test_all_handoff_contracts_are_frozen_and_versioned() -> None:
    contracts_dir = REPO_ROOT / "docs" / "contracts"
    assert contracts_dir.is_dir(), "docs/contracts/ directory missing"
    for name in FROZEN_CONTRACTS:
        path = contracts_dir / f"{name}.md"
        assert path.is_file(), f"frozen contract missing: {name}"
        text = path.read_text(encoding="utf-8")
        assert re.search(r"^- \*\*Name:\*\* `" + re.escape(name) + r"`", text, re.MULTILINE), (
            f"{name}: missing 'Name: `{name}`' header"
        )
        assert "- **Version:** v1" in text, f"{name}: not versioned as v1"
        assert "- **Status:** frozen" in text, f"{name}: not marked frozen"


def test_codeowners_covers_all_four_lanes() -> None:
    path = REPO_ROOT / "CODEOWNERS"
    assert path.is_file(), "CODEOWNERS missing"
    text = path.read_text(encoding="utf-8")
    for marker in ("@lane-a", "@lane-b", "@lane-c", "@lane-d"):
        assert marker in text, f"CODEOWNERS missing {marker}"


def test_fixture_registry_is_valid() -> None:
    path = REPO_ROOT / "data" / "fixtures" / "manifest-v1.json"
    assert path.is_file(), "fixture registry missing"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == "fixtures-v1"
    assert data["fixtures"], "no fixtures registered"
    for fixture in data["fixtures"]:
        for field in ("name", "schema_version", "producer_version", "kind"):
            assert fixture.get(field), f"fixture {fixture.get('name')}: missing {field}"
        assert fixture["kind"] in ("synthetic", "real")
        assert fixture["schema_version"].endswith("-v1"), (
            f"fixture {fixture['name']}: schema must be versioned v1"
        )


def test_optional_field_example_satisfies_contract() -> None:
    path = REPO_ROOT / "data" / "fixtures" / "error-and-optional-fields-v1-example.json"
    assert path.is_file(), "optional-field example fixture missing"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == "error-and-optional-fields-v1"
    assert "frozen_fields" in data
    absent = data["optional_fields_absent"]
    assert absent, "no absent optional fields demonstrated"
    for value in absent.values():
        assert value == "not-defined", "absent fields must read 'not-defined'"
    error = data["error_payload"]
    assert error["error_code"] and error["message"]


def test_configs_never_reference_test_noisylr() -> None:
    configs_dir = REPO_ROOT / "configs"
    assert configs_dir.is_dir()
    hits = []
    for path in sorted(configs_dir.rglob("*")):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "Test_NoisyLR" in text or "test-noisylr" in text:
            hits.append(str(path.relative_to(REPO_ROOT)))
    assert not hits, f"configs reference Test_NoisyLR (isolation kill switch): {hits}"


def test_checkpoint_registry_pins_promoted_checkpoints() -> None:
    path = REPO_ROOT / "docs" / "handoff" / "checkpoint-registry.md"
    assert path.is_file(), "checkpoint registry missing"
    text = path.read_text(encoding="utf-8")
    for checkpoint in (
        "checkpoints/train-base-gate2/best.pt",
        "checkpoints/train-proposal-gate3v2/best.pt",
    ):
        assert checkpoint in text, f"checkpoint registry missing {checkpoint}"
    assert re.search(r"[0-9a-f]{64}", text), "checkpoint registry missing a sha256 hash"
