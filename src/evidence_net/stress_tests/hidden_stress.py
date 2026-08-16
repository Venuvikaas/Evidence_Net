"""Frozen hidden stress definitions (Phase 10, structural-risk-v1 section 4).

Final stress perturbation and acquisition parameters live in
``data/stress/hidden-stress-v1.json`` with a content hash. Training code must
never read these definitions; ``training_isolation_clean`` backs the
automated isolation test. Changing the definitions requires
``hidden-stress-v2``, an ADR, and lane-D review of test integrity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
HIDDEN_STRESS_PATH = REPO_ROOT / "data" / "stress" / "hidden-stress-v1.json"
TRAINING_DIR = REPO_ROOT / "src" / "evidence_net" / "training"


class HiddenStressError(ValueError):
    """Raised when hidden stress definitions are missing or tampered with."""


def canonical_content(data: dict[str, object]) -> str:
    """Canonical JSON (sorted keys) of the definitions, excluding the hash."""
    payload = {key: value for key, value in data.items() if key != "hash"}
    return json.dumps(payload, indent=2, sort_keys=True)


def content_hash(data: dict[str, object]) -> str:
    """sha256 over the canonical content of the hidden definitions."""
    return hashlib.sha256(canonical_content(data).encode("utf-8")).hexdigest()


def load_hidden_stress(path: Path | None = None) -> dict[str, object]:
    """Load and verify the frozen hidden stress definitions.

    Raises ``HiddenStressError`` if the file is missing, malformed, or its
    content hash does not match the recorded hash.
    """
    target = path if path is not None else HIDDEN_STRESS_PATH
    if not target.is_file():
        raise HiddenStressError(f"hidden stress definitions not found: {target}")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HiddenStressError(f"hidden stress definitions are not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise HiddenStressError("hidden stress definitions must be a JSON object")
    if data.get("schema") != "hidden-stress-v1":
        raise HiddenStressError(f"unexpected schema: {data.get('schema')}")
    if data.get("frozen") is not True:
        raise HiddenStressError("hidden stress definitions are not marked frozen")
    recorded = data.get("hash")
    if not isinstance(recorded, str):
        raise HiddenStressError("hidden stress definitions missing content hash")
    if content_hash(data) != recorded:
        raise HiddenStressError(
            "hidden stress definitions content hash mismatch (definitions were edited)"
        )
    return data


def stress_params(path: Path | None = None) -> dict[str, object]:
    """Return the validated ``{perturbation, acquisition, seed}`` parameters."""
    data = load_hidden_stress(path)
    perturbation = data.get("perturbation")
    acquisition = data.get("acquisition")
    if not isinstance(perturbation, dict) or not isinstance(acquisition, dict):
        raise HiddenStressError("hidden stress definitions missing perturbation/acquisition")
    return {"perturbation": perturbation, "acquisition": acquisition, "seed": data["seed"]}


def training_isolation_clean(training_dir: Path | None = None) -> list[str]:
    """References to stress definitions inside the training package, if any.

    Training code must never read final stress definitions. Returns the list
    of offending source files (empty means the isolation holds).
    """
    target = training_dir if training_dir is not None else TRAINING_DIR
    if not target.is_dir():
        return [f"{target} missing"]
    forbidden = ("stress_tests", "hidden-stress", "hidden_stress")
    offenders: list[str] = []
    for path in sorted(target.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(token in text for token in forbidden):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    return offenders
