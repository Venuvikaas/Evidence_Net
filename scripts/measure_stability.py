"""Model stability diagnostic (Phase 8, stability-v1).

Usage (from the repository root):::

    python scripts/measure_stability.py                       # synthetic smoke
    python scripts/measure_stability.py --n-samples 8
    python scripts/measure_stability.py --config configs/modality/stability-v1.yaml
    python scripts/measure_stability.py --real                # needs train/ + checkpoints

``--synthetic`` (default) builds in-memory same-architecture models (labeled
synthetic checkpoints) over seeded synthetic observation/target pairs and
reports perturbation stability, checkpoint agreement, and measured error
diversity. ``--real`` uses the frozen validation split from ``train/`` and
the promoted Base checkpoints (``checkpoints/train-base-gate2/{best,last}.pt``,
plus the direct model when present). Never touches ``Test_NoisyLR/``. Writes
a run bundle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from evidence_net.models.base import BaseReconstruction
    from evidence_net.models.reference import deterministic_reconstruction
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id
    from evidence_net.stress_tests.forward import BlurDownsample
    from evidence_net.stress_tests.stability import (
        StabilityConfig,
        StabilityError,
        add_if_diverse,
        build_perturbations,
        checkpoint_agreement,
        error_diversity,
        load_stability_config,
        perturbation_stability,
    )
except ImportError:  # allow running before `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evidence_net.models.base import BaseReconstruction  # noqa: E402
    from evidence_net.models.reference import deterministic_reconstruction  # noqa: E402
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id  # noqa: E402
    from evidence_net.stress_tests.forward import BlurDownsample  # noqa: E402
    from evidence_net.stress_tests.stability import (  # noqa: E402
        StabilityConfig,
        StabilityError,
        add_if_diverse,
        build_perturbations,
        checkpoint_agreement,
        error_diversity,
        load_stability_config,
        perturbation_stability,
    )

DEFAULT_CONFIG = REPO_ROOT / "configs" / "modality" / "stability-v1.yaml"
BASE_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "train-base-gate2"
DIRECT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "train-direct-gate2"
SYNTHETIC_GRID = 64  # synthetic input grid (output grid = 2x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG, type=Path, help="stability-v1 config YAML"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="synthetic smoke (default; CI-safe)"
    )
    parser.add_argument(
        "--real", action="store_true", help="real mode: train/ + promoted checkpoints"
    )
    parser.add_argument("--n-samples", default=8, type=int, help="number of samples")
    parser.add_argument("--run-id", default=None, help="explicit run id")
    parser.add_argument("--out", default=REPO_ROOT / "runs", type=Path, help="runs directory")
    return parser.parse_args()


def torch_model_fn(model: torch.nn.Module) -> Any:
    """Wrap a torch model as a numpy ``(H, W) -> (2H, 2W)`` callable."""

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
    """Rebuild a model from a training-run checkpoint payload."""
    from evidence_net.models.factory import build_model
    from evidence_net.training.config import ModelConfig

    if not checkpoint.is_file():
        raise StabilityError(f"checkpoint not found: {checkpoint}")
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


def synthetic_case(
    n_samples: int, seed: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], dict[str, Any]]:
    """Seeded synthetic inputs/targets and in-memory models (labeled synthetic)."""
    rng = np.random.default_rng(seed)
    degraded = BlurDownsample(blur_sigma=0.5)
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    group_ids: list[str] = []
    for i in range(n_samples):
        clean = rng.random((SYNTHETIC_GRID, SYNTHETIC_GRID))
        inputs.append(degraded.apply(clean))
        targets.append(clean)
        group_ids.append(f"synthetic-{i:06d}")
    anchor = deterministic_reconstruction
    torch.manual_seed(seed)
    model_a = BaseReconstruction(hidden_channels=8, depth=2)
    torch.manual_seed(seed + 1)
    model_b = BaseReconstruction(hidden_channels=8, depth=2)
    # Same architecture, different (seeded) initialization: a synthetic
    # checkpoint pair, deterministic for a given config seed.
    models: dict[str, Any] = {
        "anchor": anchor,
        "synthetic-checkpoint-a": torch_model_fn(model_a),
        "synthetic-checkpoint-b": torch_model_fn(model_b),
    }
    notes = (
        "models 'synthetic-checkpoint-*' are same-architecture random "
        "initializations (synthetic, not a real training run)"
    )
    return inputs, targets, group_ids, {"models": models, "notes": notes}


def real_models() -> tuple[dict[str, Any], str]:
    """Promoted models: deterministic anchor, Base best/last, direct best (optional)."""
    base_best = BASE_CHECKPOINT_DIR / "best.pt"
    base_last = BASE_CHECKPOINT_DIR / "last.pt"
    direct_best = DIRECT_CHECKPOINT_DIR / "best.pt"
    models: dict[str, Any] = {"anchor": deterministic_reconstruction}
    if base_best.is_file() and base_last.is_file():
        models["base-best"] = torch_model_fn(load_torch_model(base_best))
        models["base-last"] = torch_model_fn(load_torch_model(base_last))
    else:
        raise StabilityError(
            f"real mode needs {base_best} and {base_last}; run --synthetic or train the Base first"
        )
    if direct_best.is_file():
        models["direct-best"] = torch_model_fn(load_torch_model(direct_best))
    reference = f"{base_best} + {base_last}" + (
        f" + {direct_best}" if direct_best.is_file() else ""
    )
    return models, reference


def real_case(
    n_samples: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], dict[str, Any]]:
    """Frozen validation observations/targets and promoted models."""
    from evidence_net.data.paths import resolve_dataset_paths
    from evidence_net.training.dataset import RestorationDataset

    paths = resolve_dataset_paths()
    dataset = RestorationDataset(paths.train_dir, split="validation", n_samples=n_samples, seed=0)
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    group_ids: list[str] = []
    for index in range(min(n_samples, len(dataset))):
        input_, target, sample_id = dataset[index]
        inputs.append(input_.squeeze(0).numpy())
        targets.append(target.squeeze(0).numpy())
        group_ids.append(f"{sample_id}")
    models, reference = real_models()
    return inputs, targets, group_ids, {"models": models, "notes": reference}


def build_summary(
    run_id: str,
    config: StabilityConfig,
    perturbation: Any,
    checkpoint: Any,
    diversity: dict[str, dict[str, float]],
    included: list[str],
    mode: str,
    notes: str,
) -> str:
    from evidence_net.stress_tests.stability import CheckpointStability, PerturbationStability

    assert isinstance(perturbation, PerturbationStability)
    assert isinstance(checkpoint, CheckpointStability)
    lines = [
        f"# Model-stability run {run_id}",
        "",
        f"- Mode: {mode}",
        f"- Contract: {config.version} (draft; freezes at Research Gate 7)",
        f"- Groups: {perturbation.n_groups}",
        f"- Notes: {notes}",
        "",
        "## Perturbation stability (invertible perturbations, MAE after inverse)",
        "",
        "| perturbation | mean | ci_lo | ci_hi |",
        "| --- | --- | --- | --- |",
    ]
    for result in perturbation.results:
        agg = result.aggregate
        lines.append(
            f"| {result.perturbation} | {agg.mean:.5f} | {agg.ci_lo:.5f} | {agg.ci_hi:.5f} |"
        )
    lines.extend(
        [
            "",
            f"- Max mean deviation: {perturbation.across['max_mean_deviation']:.5f} "
            f"({perturbation.across['argmax_perturbation']})",
            "",
            "## Checkpoint agreement (MAE between outputs)",
            "",
            "| pair | mean | ci_lo | ci_hi |",
            "| --- | --- | --- | --- |",
        ]
    )
    for name, aggregate in sorted(checkpoint.pairs.items()):
        lines.append(
            f"| {name} | {aggregate.mean:.5f} | {aggregate.ci_lo:.5f} | {aggregate.ci_hi:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Error diversity (measured; diversity is never accuracy)",
            "",
            "| pair | error_correlation | disagreement_rate | complementarity |",
            "| --- | --- | --- | --- |",
        ]
    )
    for name, metrics in sorted(diversity.items()):
        lines.append(
            f"| {name} | {metrics['error_correlation']:.4f} | "
            f"{metrics['disagreement_rate']:.4f} | {metrics['complementarity']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- Included after diversity guard: {', '.join(included) or 'none'}",
            "",
            "## Interpretation",
            "",
            "Agreement is stability, not correctness; it is never a probability "
            "of truth and never calibration (stability-v1, Gate 7).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.real and args.synthetic:
        print("FAIL: --real and --synthetic are mutually exclusive", file=sys.stderr)
        return 1
    try:
        config = load_stability_config(args.config)
    except StabilityError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    mode = "real" if args.real else "synthetic"
    if mode == "real":
        try:
            inputs, targets, group_ids, case = real_case(args.n_samples)
        except Exception as exc:
            print(f"FAIL: real mode needs train/ and promoted checkpoints: {exc}", file=sys.stderr)
            return 1
    else:
        inputs, targets, group_ids, case = synthetic_case(args.n_samples, config.seed)
    models: dict[str, Any] = case["models"]
    notes = str(case["notes"])

    perturbations = build_perturbations(config)
    perturbation = perturbation_stability(
        models["base-best" if "base-best" in models else "synthetic-checkpoint-a"],
        inputs,
        group_ids,
        perturbations,
        n_boot=config.n_boot,
        seed=config.seed,
    )
    checkpoint_models = [
        (name, fn) for name, fn in models.items() if name not in ("anchor", "direct-best")
    ]
    checkpoint = checkpoint_agreement(
        checkpoint_models, inputs, group_ids, n_boot=config.n_boot, seed=config.seed
    )

    # Diversity uses per-pixel signed errors vs the reference target,
    # averaged over the sample per group-consistent grid.
    full_errors: dict[str, np.ndarray] = {}
    for name, fn in models.items():
        full_errors[name] = np.stack(
            [fn(input_) - target for input_, target in zip(inputs, targets, strict=True)]
        ).mean(axis=0)
    diversity = error_diversity(full_errors)
    # Diversity guard accumulates the included set: a candidate joins only
    # when its measured disagreement vs the models already included meets the
    # threshold (redundant models are excluded).
    included: list[str] = []
    included_errors: dict[str, np.ndarray] = {}
    for name in models:
        if not included_errors:
            included.append(name)
            included_errors[name] = full_errors[name]
            continue
        combined = {**included_errors, name: full_errors[name]}
        if add_if_diverse(name, combined, config.min_diversity_threshold, tolerance=1e-6):
            included.append(name)
            included_errors[name] = full_errors[name]

    run_id = args.run_id or new_run_id("measure-stability")
    dataset_manifest = "synthetic-software-only" if mode == "synthetic" else "dataset-splits-v1"
    manifest = {
        "run_id": run_id,
        "dataset_manifest": dataset_manifest,
        "mode": mode,
        "contract": config.version,
        "test_final_isolation": "confirmed-no-test-noisylr",
    }
    summary = build_summary(
        run_id, config, perturbation, checkpoint, diversity, included, mode, notes
    )
    metrics = {
        "perturbation": perturbation.as_dict(),
        "checkpoint": checkpoint.as_dict(),
        "diversity": diversity,
        "included_models": included,
    }
    run_dir = create_run_bundle(
        args.out,
        run_id,
        config={**config.as_dict(), "mode": mode},
        manifest=manifest,
        metrics=metrics,
        summary=summary,
        reference=notes,
    )
    print(f"Run bundle written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
