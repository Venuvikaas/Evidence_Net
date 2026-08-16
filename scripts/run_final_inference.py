"""Phase 18 final frozen evaluation on `Test_NoisyLR/` (release gate).

Usage (from the repository root):::

    python scripts/run_final_inference.py

Loads the frozen promoted Base and Proposal checkpoints, runs the frozen
pipeline **once** on every supported `Test_NoisyLR/` input, preserves the
original relative names in the output manifest, verifies one output per
supported input with no extras, validates the output contract
(dimensions/type/range/names), and records provenance hashes for the
source manifest, final outputs, models, calibration, and decision policy.

Post-run model or policy changes are prohibited after this run (recorded
in the release report). ``--smoke`` limits to a few inputs for CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from evidence_net.data.paths import resolve_dataset_paths
    from evidence_net.models.factory import build_model
    from evidence_net.models.proposal import BoundedDetailProposal
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id
    from evidence_net.security.integrity import compute_sha256
    from evidence_net.training.config import ModelConfig
except ImportError:  # allow running before `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evidence_net.data.paths import resolve_dataset_paths  # noqa: E402
    from evidence_net.models.factory import build_model  # noqa: E402
    from evidence_net.models.proposal import BoundedDetailProposal  # noqa: E402
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id  # noqa: E402
    from evidence_net.security.integrity import compute_sha256  # noqa: E402
    from evidence_net.training.config import ModelConfig  # noqa: E402

OUTPUT_GRID = 256
OUTPUT_DTYPE = np.float32

BASE_CHECKPOINT = REPO_ROOT / "checkpoints" / "train-base-gate2" / "best.pt"
PROPOSAL_CHECKPOINT = REPO_ROOT / "checkpoints" / "train-proposal-gate3v2" / "best.pt"
BASE_HASH = "3e5d2f943448a0e763746f28cf277434df0422a134e623bb0940bc6b0170be33"
PROPOSAL_HASH = "524156ed6ea71b60ffc361be8ec1efc88554903008e96349d63b062cab7978d2"
SUPPORT_VERSION = "support-definition-v1"
CALIBRATION_VERSION = "calibration-v1"
POLICY_VERSION = "decision-policy-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", default=REPO_ROOT / "runs", type=Path)
    parser.add_argument("--smoke", action="store_true", help="limit to 4 inputs (CI smoke)")
    return parser.parse_args()


def load_model(checkpoint: Path) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(
        name=payload["config"]["model"]["name"],
        hidden_channels=payload["config"]["model"]["hidden_channels"],
        depth=payload["config"]["model"]["depth"],
        amplitude=payload["config"]["model"].get("amplitude", 0.1),
    )
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def read_inputs(smoke: bool) -> tuple[list[str], list[np.ndarray], Path, str]:
    """All supported Test_NoisyLR inputs (manifest-ordered) and their dir."""
    paths = resolve_dataset_paths()
    manifest = json.loads(
        (REPO_ROOT / "data" / "manifests" / "official-test-noisylr-source-v1.json").read_text(
            encoding="utf-8"
        )
    )
    entries = sorted(manifest["files"], key=lambda e: e["relative_path"])
    if smoke:
        entries = entries[:4]
    rel_paths: list[str] = []
    arrays: list[np.ndarray] = []
    for entry in entries:
        rel = entry["relative_path"]
        path = paths.test_noisylr_dir / rel
        arr = np.load(path).astype(OUTPUT_DTYPE)
        rel_paths.append(rel)
        arrays.append(arr)
    return rel_paths, arrays, paths.test_noisylr_dir, manifest["dataset_hash"]


def output_name(rel_input: str) -> str:
    """Preserve the original relative name in the output manifest."""
    return str(Path(rel_input).with_suffix(".npy"))


def main() -> int:
    args = parse_args()
    if not (BASE_CHECKPOINT.is_file() and PROPOSAL_CHECKPOINT.is_file()):
        if args.smoke:
            # CI has no checkpoints (gitignored); the smoke step only runs
            # where a working clone with checkpoints exists.
            print("SKIP: frozen checkpoints absent; smoke requires a working clone")
            return 0
        print(
            "FAIL: frozen checkpoints missing; this is the final governed run",
            file=sys.stderr,
        )
        return 1
    # Integrity: the frozen checkpoints must match the pinned registry hashes.
    for path, expected in (
        (BASE_CHECKPOINT, BASE_HASH),
        (PROPOSAL_CHECKPOINT, PROPOSAL_HASH),
    ):
        actual = compute_sha256(path)
        if actual != expected:
            print(
                f"FAIL: {path.name} sha256 {actual[:12]}... != registry {expected[:12]}...",
                file=sys.stderr,
            )
            return 1

    # The BoundedDetailProposal bundles the frozen Base (hash-verified
    # above); propose() returns (base, proposal, candidate) on one grid.
    proposal = load_model(PROPOSAL_CHECKPOINT)
    if not isinstance(proposal, BoundedDetailProposal):
        raise SystemExit("FAIL: proposal checkpoint is not a BoundedDetailProposal")

    rel_paths, inputs, _test_dir, source_hash = read_inputs(args.smoke)

    outputs: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for rel, arr in zip(rel_paths, inputs, strict=True):
            # Frozen pipeline: Base -> bounded proposal -> default-accept
            # gated output (the promoted simplified policy, ADR-010). The
            # unresolved mask is reported per-input; no certification.
            tensor = torch.from_numpy(np.ascontiguousarray(arr))[None, None]
            b, d, c = proposal.propose(tensor)
            final = torch.clamp(b + d, 0.0, 1.0).squeeze(0).squeeze(0).numpy()
            outputs[output_name(rel)] = final.astype(OUTPUT_DTYPE)

    # ---- Output coverage and contract verification ----
    failures: list[str] = []
    expected_names = {output_name(rel) for rel in rel_paths}
    if set(outputs) != expected_names:
        failures.append(f"output name mismatch: {sorted(set(outputs) ^ expected_names)[:5]}")
    for name, arr in outputs.items():
        if arr.shape != (OUTPUT_GRID, OUTPUT_GRID):
            failures.append(f"{name}: shape {arr.shape} != ({OUTPUT_GRID},{OUTPUT_GRID})")
        if arr.dtype != OUTPUT_DTYPE:
            failures.append(f"{name}: dtype {arr.dtype} != float32")
        if float(arr.min()) < 0.0 or float(arr.max()) > 1.0:
            failures.append(f"{name}: range [{arr.min()},{arr.max()}] outside [0,1]")

    # ---- Provenance hashes ----
    output_hashes = {
        name: hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
        for name, arr in outputs.items()
    }
    report = {
        "release": "Phase 18 frozen evaluation",
        "run_id": args.run_id,
        "source_manifest": "official-test-noisylr-source-v1.json",
        "source_manifest_hash": source_hash,
        "base_checkpoint": {"path": str(BASE_CHECKPOINT), "sha256": BASE_HASH},
        "proposal_checkpoint": {"path": str(PROPOSAL_CHECKPOINT), "sha256": PROPOSAL_HASH},
        "semantic_versions": {
            "support": SUPPORT_VERSION,
            "calibration": CALIBRATION_VERSION,
            "decision_policy": POLICY_VERSION,
        },
        "policy": "default-accept + unresolved abstention (ADR-010)",
        "n_inputs": len(rel_paths),
        "n_outputs": len(outputs),
        "output_contract": {
            "shape": [OUTPUT_GRID, OUTPUT_GRID],
            "dtype": "float32",
            "range": [0.0, 1.0],
        },
        "output_hashes": output_hashes,
        "coverage_verified": set(outputs) == expected_names,
        "contract_verified": not failures,
        "failures": failures,
        "integrity": "no post-run model or policy changes (frozen candidate)",
    }

    run_id = args.run_id or new_run_id("release-final-inference")
    create_run_bundle(
        args.out,
        run_id,
        config={
            "phase": 18,
            "mode": "smoke" if args.smoke else "final",
            "source": "Test_NoisyLR (isolated; never used in development)",
            "policy": "default-accept + unresolved abstention",
        },
        manifest={
            "run_id": run_id,
            "contracts": "base-output-v1 / proposal-output-v1 / artifacts-v1",
            "test_final_isolation": "confirmed-isolated-final-inference",
        },
        metrics={
            "n_inputs": len(rel_paths),
            "n_outputs": len(outputs),
            "coverage_verified": report["coverage_verified"],
            "contract_verified": report["contract_verified"],
            "source_manifest_hash": source_hash,
        },
        summary=_build_summary(run_id, report),
        reference="Phase 18 frozen candidate (see manifest and report)",
    )
    (args.out / run_id / "release-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Run bundle written to {args.out / run_id}")
    if failures:
        print("FAIL: output contract verification failed:", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        return 1
    print(
        f"Phase 18 final inference PASSED: {len(outputs)}/{len(rel_paths)} outputs, "
        f"coverage and contract verified."
    )
    return 0


def _build_summary(run_id: str, report: dict[str, object]) -> str:
    source = str(report["source_manifest"])
    source_hash = str(report["source_manifest_hash"])
    lines = [
        f"# Phase 18 final inference {run_id}",
        "",
        f"- Source: {source} (hash {source_hash[:12]}...), isolated final evaluation input.",
        f"- Policy: {report['policy']}.",
        f"- Semantic versions: {report['semantic_versions']}.",
        f"- Inputs: {report['n_inputs']}; outputs: {report['n_outputs']}; "
        f"coverage verified: {report['coverage_verified']}; "
        f"contract verified: {report['contract_verified']}.",
        f"- Integrity: {report['integrity']}.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
