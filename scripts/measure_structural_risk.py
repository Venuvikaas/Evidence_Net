"""Structural-risk and downstream evaluation (Phase 10, structural-risk-v1).

Usage (from the repository root):::

    python scripts/measure_structural_risk.py                     # synthetic smoke
    python scripts/measure_structural_risk.py --n-samples 8
    python scripts/measure_structural_risk.py --real              # needs train/ + checkpoints

Reports the five separate Gate 9 evidence categories: candidate suite
(manipulations of restored outputs), ambiguity (clean-candidate pairs with
near-identical observations), acquisition (pre-inference artifacts),
natural failures (frozen bank), and downstream (measurement fidelity of
base vs candidate vs oracle-patch proxy). ``--real`` uses the promoted Base
and Proposal checkpoints on the frozen validation split. Never touches
``Test_NoisyLR/``. Writes a run bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id
    from evidence_net.stress_tests.acquisition import build_acquisition_suite
    from evidence_net.stress_tests.ambiguity import ambiguity_cases
    from evidence_net.stress_tests.downstream import evaluate_downstream
    from evidence_net.stress_tests.hidden_stress import (
        HIDDEN_STRESS_PATH,
        content_hash,
        load_hidden_stress,
    )
    from evidence_net.stress_tests.structural import build_candidate_suite
except ImportError:  # allow running before `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id  # noqa: E402
    from evidence_net.stress_tests.acquisition import build_acquisition_suite  # noqa: E402
    from evidence_net.stress_tests.ambiguity import ambiguity_cases  # noqa: E402
    from evidence_net.stress_tests.downstream import evaluate_downstream  # noqa: E402
    from evidence_net.stress_tests.hidden_stress import (  # noqa: E402
        HIDDEN_STRESS_PATH,
        content_hash,
        load_hidden_stress,
    )
    from evidence_net.stress_tests.structural import build_candidate_suite  # noqa: E402

NATURAL_BANK = REPO_ROOT / "data" / "failures" / "natural-failures-v1.json"
GRID = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic", action="store_true", help="synthetic smoke (default; CI-safe)"
    )
    parser.add_argument(
        "--real", action="store_true", help="real mode: train/ + promoted checkpoints"
    )
    parser.add_argument("--n-samples", default=8, type=int, help="samples per group")
    parser.add_argument("--seed", default=0, type=int, help="seeded rng")
    parser.add_argument("--run-id", default=None, help="explicit run id")
    parser.add_argument("--out", default=REPO_ROOT / "runs", type=Path, help="runs directory")
    return parser.parse_args()


def _structured_fixture(size: int, rng: np.random.Generator) -> np.ndarray:
    """Bright vertical lines on a dark background, with light noise variation."""
    image = np.full((size, size), 0.02)
    for column in (size // 4, size // 2, 3 * size // 4):
        image[:, column] = 0.9
    return np.clip(image + rng.normal(0.0, 0.005, size=image.shape), 0.0, 1.0)


def synthetic_case(
    n_samples: int, seed: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Seeded synthetic evidence for all five categories."""
    rng = np.random.default_rng(seed)
    fixtures = [_structured_fixture(GRID, rng) for _ in range(n_samples)]
    ids = [f"synthetic-{i:06d}" for i in range(n_samples)]

    # Candidate suite: apply every manipulation to the first fixture.
    suite = build_candidate_suite()
    candidate_effects: dict[str, dict[str, float]] = {}
    for manipulation in suite:
        modified = manipulation.apply(fixtures[0], np.random.default_rng(seed))
        from evidence_net.stress_tests.downstream import downstream_measurements

        base_measurements = downstream_measurements(fixtures[0], fixtures[0])
        modified_measurements = downstream_measurements(modified, fixtures[0])
        candidate_effects[manipulation.name] = {
            measurement: abs(modified_measurements[measurement] - base_measurements[measurement])
            for measurement in base_measurements
        }

    # Downstream: base vs candidate (false-line on every fixture).
    candidates = [
        build_candidate_suite(names=("false-line",))[0].apply(image, rng) for image in fixtures
    ]
    downstream = evaluate_downstream(
        {"base": fixtures, "candidate": candidates}, fixtures, ids, n_boot=200, seed=seed
    )

    # Ambiguity: non-identifiable pairs (synthetic).
    ambiguity = [
        case.as_dict() | {"case_id": case.case_id} for case in ambiguity_cases(size=64, sigma=1.5)
    ]

    # Acquisition: artifacts on a degraded input (synthetic).
    from evidence_net.stress_tests.forward import BlurDownsample

    input_grid = BlurDownsample(0.5).apply(fixtures[0])
    acquisition: dict[str, dict[str, float]] = {}
    for artifact in build_acquisition_suite():
        modified = artifact.apply(input_grid, np.random.default_rng(seed))
        acquisition[artifact.name] = {
            "input_delta": float(np.abs(modified - input_grid).mean()),
        }

    natural = _load_natural_bank()
    return (
        {"candidate_effects": candidate_effects, "downstream": downstream},
        {"ambiguity": ambiguity, "acquisition": acquisition, "natural_bank": natural},
        {
            "mode": "synthetic",
            "notes": "all probes synthetic software-only; never used in scientific reports",
        },
    )


def _load_natural_bank() -> dict[str, Any]:
    if not NATURAL_BANK.is_file():
        raise FileNotFoundError(f"natural failure bank not found: {NATURAL_BANK}")
    return json.loads(NATURAL_BANK.read_text(encoding="utf-8"))


def torch_model_fn(model: torch.nn.Module) -> Any:
    def fn(y: np.ndarray) -> np.ndarray:
        # ``y`` may be a non-contiguous view (negative strides from a
        # tensor ``.numpy()`` view); copy so ``torch.from_numpy`` accepts it.
        array = np.ascontiguousarray(np.asarray(y, dtype=np.float32))
        tensor = torch.from_numpy(array)[None, None]
        with torch.no_grad():
            output = model(tensor)
        return output[0, 0].numpy()

    return fn


def load_torch_model(checkpoint: Path) -> torch.nn.Module:
    from evidence_net.models.factory import build_model
    from evidence_net.training.config import ModelConfig

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = payload["config"]["model"]
    config = ModelConfig(
        name=model_config["name"],
        hidden_channels=model_config["hidden_channels"],
        depth=model_config["depth"],
        amplitude=model_config.get("amplitude", 0.1),
    )
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def real_case(n_samples: int, seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Real evidence: frozen Base/Proposal on the validation split."""
    from evidence_net.data.paths import resolve_dataset_paths
    from evidence_net.evaluation.oracle import oracle_output, patch_gate
    from evidence_net.training.dataset import RestorationDataset

    paths = resolve_dataset_paths()
    dataset = RestorationDataset(paths.train_dir, split="validation", n_samples=n_samples, seed=0)
    proposal_model = load_torch_model(
        REPO_ROOT / "checkpoints" / "train-proposal-gate3v2" / "best.pt"
    )
    from evidence_net.models.proposal import BoundedDetailProposal

    if not isinstance(proposal_model, BoundedDetailProposal):
        raise SystemExit("FAIL: proposal checkpoint is not a BoundedDetailProposal")

    bases: list[np.ndarray] = []
    proposals: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    oracle_patches: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    ids: list[str] = []
    for index in range(min(n_samples, len(dataset))):
        input_, target, sample_id = dataset[index]
        y = input_.squeeze(0).numpy()
        x = target.squeeze(0).numpy()
        # The proposal checkpoint is a full BoundedDetailProposal: calling it
        # directly returns the *candidate* (b + d). Use ``propose`` so ``d``
        # is the bounded detail residual (matching measure_oracle.py).
        tensor = torch.from_numpy(np.ascontiguousarray(np.asarray(y, dtype=np.float32)))[None, None]
        with torch.no_grad():
            b_t, d_t, c_t = proposal_model.propose(tensor)
        b = b_t.squeeze().numpy()
        d = d_t.squeeze().numpy()
        c = c_t.squeeze().numpy()
        gate = patch_gate(b, c, x)
        bases.append(b)
        proposals.append(d)
        candidates.append(c)
        oracle_patches.append(oracle_output(b, d, gate))
        targets.append(x)
        ids.append(f"{sample_id}")

    downstream = evaluate_downstream(
        {"base": bases, "candidate": candidates, "oracle-patch": oracle_patches},
        targets,
        ids,
        n_boot=200,
        seed=seed,
    )

    suite = build_candidate_suite()
    candidate_effects: dict[str, dict[str, float]] = {}
    for manipulation in suite:
        modified = manipulation.apply(bases[0], np.random.default_rng(seed))
        from evidence_net.stress_tests.downstream import downstream_measurements

        base_measurements = downstream_measurements(bases[0], targets[0])
        modified_measurements = downstream_measurements(modified, targets[0])
        candidate_effects[manipulation.name] = {
            measurement: abs(modified_measurements[measurement] - base_measurements[measurement])
            for measurement in base_measurements
        }

    ambiguity = [
        case.as_dict() | {"case_id": case.case_id} for case in ambiguity_cases(size=64, sigma=1.5)
    ]
    from evidence_net.stress_tests.forward import BlurDownsample

    input_grid = BlurDownsample(0.5).apply(bases[0])
    acquisition: dict[str, dict[str, float]] = {}
    for artifact in build_acquisition_suite():
        modified = artifact.apply(input_grid, np.random.default_rng(seed))
        acquisition[artifact.name] = {"input_delta": float(np.abs(modified - input_grid).mean())}

    natural = _load_natural_bank()
    return (
        {"candidate_effects": candidate_effects, "downstream": downstream},
        {"ambiguity": ambiguity, "acquisition": acquisition, "natural_bank": natural},
        {
            "mode": "real",
            "notes": (
                "downstream uses frozen Base/Proposal and the oracle-patch study proxy "
                "(never at inference); ambiguity/acquisition probes are synthetic (labeled)"
            ),
        },
    )


def build_summary(run_id: str, metrics: dict[str, Any], meta: dict[str, Any]) -> str:
    lines = [
        f"# Structural-risk run {run_id}",
        "",
        "- Contract: structural-risk-v1 (draft; freezes at Research Gate 9)",
        f"- Mode: {meta['mode']}",
        f"- Notes: {meta['notes']}",
        f"- Hidden stress hash: {meta['hidden_stress_hash'][:16]}...",
        f"- Natural failure bank: {meta['natural_count']} frozen cases",
        "",
        "## Candidate suite (local effect on downstream measurements)",
        "",
        "| manipulation | edge_displacement_px | edge_components | bright_components |",
        "| --- | --- | --- | --- |",
    ]
    for name, effects in sorted(metrics["candidate_effects"].items()):
        lines.append(
            f"| {name} | {effects['edge_displacement_px']:.4f} | "
            f"{effects['edge_components']:.2f} | {effects['bright_components']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Downstream error per output type (group bootstrap, mean)",
            "",
            "| output | edge_displacement_px | edge_components | bright_components |",
            "| --- | --- | --- | --- |",
        ]
    )
    for output_type, measurements in sorted(metrics["downstream"].items()):
        lines.append(
            f"| {output_type} | {measurements['edge_displacement_px']['aggregate']['mean']:.4f} | "
            f"{measurements['edge_components']['aggregate']['mean']:.2f} | "
            f"{measurements['bright_components']['aggregate']['mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Ambiguity cases (observation vs candidate MAE)",
            "",
            "| case | observation_mae | candidate_mae |",
            "| --- | --- | --- |",
        ]
    )
    for case in metrics["ambiguity"]:
        lines.append(
            f"| {case['case_id']} | {case['observation_mae']:.5f} | {case['candidate_mae']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Acquisition artifacts (mean input delta)",
            "",
            "| artifact | input_delta |",
            "| --- | --- |",
        ]
    )
    for name, values in sorted(metrics["acquisition"].items()):
        lines.append(f"| {name} | {values['input_delta']:.5f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Structural claims require separate candidate, ambiguity, "
            "acquisition, natural-failure, and downstream evidence (Gate 9). "
            "No hallucination-resistance claim follows from any single suite.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.real and args.synthetic:
        print("FAIL: --real and --synthetic are mutually exclusive", file=sys.stderr)
        return 1
    hidden = load_hidden_stress()
    hidden_hash = content_hash(hidden)
    try:
        if args.real:
            evidence, meta_probes, meta = real_case(args.n_samples, args.seed)
        else:
            evidence, meta_probes, meta = synthetic_case(args.n_samples, args.seed)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    metrics = {**evidence, **meta_probes}
    meta["hidden_stress_hash"] = hidden_hash
    meta["natural_count"] = len(meta_probes["natural_bank"].get("cases", []))

    run_id = args.run_id or new_run_id("structural-risk")
    manifest = {
        "run_id": run_id,
        "dataset_manifest": (
            "synthetic-software-only" if meta["mode"] == "synthetic" else "dataset-splits-v1"
        ),
        "mode": meta["mode"],
        "contract": "structural-risk-v1",
        "hidden_stress_hash": hidden_hash,
        "test_final_isolation": "confirmed-no-test-noisylr",
    }
    summary = build_summary(run_id, metrics, meta)
    run_dir = create_run_bundle(
        args.out,
        run_id,
        config={"mode": meta["mode"], "seed": args.seed, "n_samples": args.n_samples},
        manifest=manifest,
        metrics=metrics,
        summary=summary,
        reference=HIDDEN_STRESS_PATH.as_posix(),
    )
    print(f"Run bundle written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
