"""Measurement-consistency diagnostic over the bounded forward family.

Usage (from the repository root):::

    python scripts/measure_consistency.py                      # synthetic smoke
    python scripts/measure_consistency.py --n-samples 8
    python scripts/measure_consistency.py --config configs/modality/forward-v1.yaml
    python scripts/measure_consistency.py --real              # needs train/ dataset

``--synthetic`` (default) generates seeded clean images on the output grid,
degrades them with the ``noisy-blur`` operator as the stand-in true
generator, restores with the deterministic bilinear anchor (no checkpoints
needed), and reports compatibility across the whole family. ``--real`` uses
the frozen validation split from ``train/`` with the deterministic anchor as
the restored output. Never touches ``Test_NoisyLR/``. Writes a run bundle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id
    from evidence_net.stress_tests.consistency import (
        build_consistency_report,
        measure_noise_variance,
    )
    from evidence_net.stress_tests.forward import (
        ForwardConfig,
        ForwardError,
        NoisyBlurDownsample,
        build_operator_family,
        load_forward_config,
    )
except ImportError:  # allow running before `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id  # noqa: E402
    from evidence_net.stress_tests.consistency import (  # noqa: E402
        build_consistency_report,
        measure_noise_variance,
    )
    from evidence_net.stress_tests.forward import (  # noqa: E402
        ForwardConfig,
        ForwardError,
        NoisyBlurDownsample,
        build_operator_family,
        load_forward_config,
    )

DEFAULT_CONFIG = REPO_ROOT / "configs" / "modality" / "forward-v1.yaml"
OUTPUT_GRID = 256  # official output grid (tensor-v1)
INPUT_GRID = 128  # official input grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        type=Path,
        help="forward-model-v1 config YAML",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use seeded synthetic samples (default; CI-safe, no dataset needed)",
    )
    parser.add_argument(
        "--real", action="store_true", help="use the frozen validation split from train/"
    )
    parser.add_argument("--n-samples", default=8, type=int, help="number of samples")
    parser.add_argument("--run-id", default=None, help="explicit run id")
    parser.add_argument("--out", default=REPO_ROOT / "runs", type=Path, help="runs directory")
    return parser.parse_args()


def synthetic_pairs(
    n_samples: int, seed: int
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Seeded clean images, noisy-blur observations, anchor restorations.

    The deterministic bilinear anchor is the stand-in for the frozen Base
    (2x up-sample of the observation), so the run needs no checkpoints.
    """
    rng = np.random.default_rng(seed)
    # True generator: noisy-blur with the family mid-range parameters.
    generator = NoisyBlurDownsample(blur_sigma=0.8, noise_sigma=0.02, seed=seed)
    restored: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    group_ids: list[str] = []
    for i in range(n_samples):
        clean = rng.random((OUTPUT_GRID, OUTPUT_GRID))
        observation = generator.apply(clean, rng)
        anchor = _bilinear_upsample(observation, 2)
        restored.append(anchor)
        observations.append(observation)
        group_ids.append(f"synthetic-{i:06d}")
    return restored, observations, group_ids


def _bilinear_upsample(image: np.ndarray, scale: int) -> np.ndarray:
    tensor = torch.from_numpy(image)[None, None]
    up = F.interpolate(tensor, scale_factor=scale, mode="bilinear", align_corners=False)
    return up[0, 0].numpy()


def real_pairs(n_samples: int) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Frozen validation observations and deterministic-anchor restorations."""
    from evidence_net.data.paths import resolve_dataset_paths
    from evidence_net.models.reference import deterministic_reconstruction
    from evidence_net.training.dataset import RestorationDataset

    paths = resolve_dataset_paths()
    dataset = RestorationDataset(paths.train_dir, split="validation", n_samples=n_samples, seed=0)
    restored: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    group_ids: list[str] = []
    for index in range(min(n_samples, len(dataset))):
        input_, _, sample_id = dataset[index]
        input_np = input_.squeeze(0).numpy()
        observations.append(input_np)
        restored.append(deterministic_reconstruction(input_np))
        group_ids.append(f"{sample_id}")
    return restored, observations, group_ids


def build_summary(
    run_id: str,
    config: ForwardConfig,
    report: object,
    noise_variances: dict[str, dict[str, float]],
    mode: str,
) -> str:
    from evidence_net.stress_tests.consistency import ConsistencyReport

    assert isinstance(report, ConsistencyReport)
    lines = [
        f"# Measurement-consistency run {run_id}",
        "",
        f"- Mode: {mode}",
        f"- Contract: {config.version} (draft; freezes at Research Gate 6)",
        f"- Operators: {', '.join(entry.operator for entry in report.operators)}",
        f"- Groups: {report.n_groups}",
        "",
        "## Per-operator residuals (group bootstrap, MAE on input grid)",
        "",
        "| operator | kind | mean MAE | ci_lo | ci_hi | bias |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in report.operators:
        agg = entry.aggregate
        lines.append(
            f"| {entry.operator} | {entry.kind} | {agg.mean:.5f} | "
            f"{agg.ci_lo:.5f} | {agg.ci_hi:.5f} | {entry.bias_mean:+.5f} |"
        )
    lines.extend(
        [
            "",
            "## Across the operator family (distribution, not minimum only)",
            "",
            f"- min MAE: {report.across_operators['min_mae']:.5f} "
            f"({report.across_operators['argmin_operator']})",
            f"- median MAE: {report.across_operators['median_mae']:.5f}",
            f"- max MAE: {report.across_operators['max_mae']:.5f}",
            "",
            "## Stochastic spread",
            "",
        ]
    )
    for operator, spread in sorted(noise_variances.items()):
        lines.append(
            f"- {operator}: std {spread['std']:.5f}, "
            f"min {spread['min']:.5f}, max {spread['max']:.5f} "
            f"over {spread['n_draws']} draws"
        )
    lines.extend(["", "## Interpretation", "", report.interpretation, ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.real and args.synthetic:
        print("FAIL: --real and --synthetic are mutually exclusive", file=sys.stderr)
        return 1
    try:
        config = load_forward_config(args.config)
    except ForwardError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    mode = "real" if args.real else "synthetic"
    if mode == "real":
        try:
            restored, observations, group_ids = real_pairs(args.n_samples)
        except Exception as exc:  # dataset missing or unreadable
            print(f"FAIL: real mode needs the train/ dataset: {exc}", file=sys.stderr)
            return 1
    else:
        restored, observations, group_ids = synthetic_pairs(args.n_samples, config.seed)

    operators = build_operator_family(config)
    report = build_consistency_report(
        restored, observations, group_ids, operators, n_boot=config.n_boot, seed=config.seed
    )
    noise_variances = {
        operator.name: measure_noise_variance(operator, restored[0], n_draws=32, seed=config.seed)
        for operator in operators
        if operator.is_stochastic
    }

    run_id = args.run_id or new_run_id("measure-consistency")
    dataset_manifest = "synthetic-software-only" if mode == "synthetic" else "dataset-splits-v1"
    manifest = {
        "run_id": run_id,
        "dataset_manifest": dataset_manifest,
        "mode": mode,
        "contract": config.version,
        "test_final_isolation": "confirmed-no-test-noisylr",
    }
    summary = build_summary(run_id, config, report, noise_variances, mode)
    run_dir = create_run_bundle(
        args.out,
        run_id,
        config={**config.as_dict(), "mode": mode},
        manifest=manifest,
        metrics=report.as_dict(),
        summary=summary,
        reference="no-checkpoint",
    )
    print(f"Run bundle written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
