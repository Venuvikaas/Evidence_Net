"""Distribution familiarity diagnostic (Phase 9, familiarity-v1).

Usage (from the repository root):::

    python scripts/measure_familiarity.py                        # synthetic smoke
    python scripts/measure_familiarity.py --n-samples 8
    python scripts/measure_familiarity.py --config configs/modality/familiarity-v1.yaml
    python scripts/measure_familiarity.py --real                 # needs train/ dataset

``--synthetic`` (default) builds a seeded synthetic population and declared
shift groups (source, severity, degradation, acquisition) plus rare-valid
structures, fits the reference-distance baseline, and reports shift
detection. ``--real`` fits the reference on the frozen calibration split of
``train/`` and probes validation, heldout-source, severity/acquisition shifts
of validation inputs, and synthetic rare-valid structures (labeled). Never
touches ``Test_NoisyLR/``. Writes a run bundle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id
    from evidence_net.stress_tests.familiarity import (
        RARE_VALID_GROUP,
        REFERENCE_GROUP,
        FamiliarityConfig,
        FamiliarityError,
        ReferenceFamiliarity,
        build_familiarity_report,
        build_shift_suite,
        load_familiarity_config,
    )
    from evidence_net.stress_tests.forward import NoisyBlurDownsample
except ImportError:  # allow running before `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id  # noqa: E402
    from evidence_net.stress_tests.familiarity import (  # noqa: E402
        RARE_VALID_GROUP,
        REFERENCE_GROUP,
        FamiliarityConfig,
        FamiliarityError,
        ReferenceFamiliarity,
        build_familiarity_report,
        build_shift_suite,
        load_familiarity_config,
    )
    from evidence_net.stress_tests.forward import NoisyBlurDownsample  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "modality" / "familiarity-v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG, type=Path, help="familiarity-v1 config YAML"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="synthetic smoke (default; CI-safe)"
    )
    parser.add_argument(
        "--real", action="store_true", help="real mode: train/ calibration reference"
    )
    parser.add_argument("--n-samples", default=32, type=int, help="samples per group")
    parser.add_argument("--run-id", default=None, help="explicit run id")
    parser.add_argument("--out", default=REPO_ROOT / "runs", type=Path, help="runs directory")
    return parser.parse_args()


def _group_ids(group_name: str, n: int) -> list[str]:
    return [f"{group_name}-{i:04d}" for i in range(n)]


def synthetic_case(
    config: FamiliarityConfig,
) -> tuple[ReferenceFamiliarity, dict[str, list[np.ndarray]], dict[str, list[str]], str]:
    suite = build_shift_suite(n_per_shift=config.n_per_shift, seed=config.seed)
    reference = ReferenceFamiliarity.fit(suite[REFERENCE_GROUP], threshold=config.threshold)
    probes = {name: images for name, images in suite.items() if name != REFERENCE_GROUP}
    ids = {name: _group_ids(name, len(images)) for name, images in probes.items()}
    notes = "synthetic software-only population and shift groups (never used in scientific reports)"
    return reference, probes, ids, notes


def real_case(
    config: FamiliarityConfig,
) -> tuple[ReferenceFamiliarity, dict[str, list[np.ndarray]], dict[str, list[str]], str]:
    """Reference on the calibration split; probes from validation and heldout-source."""
    from evidence_net.data.paths import resolve_dataset_paths
    from evidence_net.stress_tests.familiarity import _rare_valid_images
    from evidence_net.training.dataset import RestorationDataset

    paths = resolve_dataset_paths()
    rng = np.random.default_rng(config.seed)

    def load(split: str, n: int) -> tuple[list[np.ndarray], list[str]]:
        dataset = RestorationDataset(paths.train_dir, split=split, n_samples=n, seed=0)
        images: list[np.ndarray] = []
        ids: list[str] = []
        for index in range(min(n, len(dataset))):
            input_, _, sample_id = dataset[index]
            images.append(input_.squeeze(0).numpy())
            ids.append(f"{split}-{sample_id}")
        return images, ids

    reference_images, _ = load("calibration", config.n_reference)
    reference = ReferenceFamiliarity.fit(reference_images, threshold=config.threshold)

    validation, validation_ids = load("validation", config.n_per_shift)
    heldout_source, source_ids = load("heldout-source", config.n_per_shift)

    severe = NoisyBlurDownsample(blur_sigma=1.5, noise_sigma=0.03, seed=config.seed)
    severity = [severe.apply(image) for image in validation]
    acquisition = [np.clip(0.8 * image + 0.1, 0.0, 1.0) for image in validation]
    rare_valid = _rare_valid_images(config.n_per_shift, validation[0].shape[0], rng)

    probes: dict[str, list[np.ndarray]] = {
        "validation": validation,
        "source": heldout_source,
        "severity": severity,
        "acquisition": acquisition,
        RARE_VALID_GROUP: rare_valid,
    }
    ids: dict[str, list[str]] = {
        "validation": validation_ids,
        "source": source_ids,
        "severity": _group_ids("severity", len(severity)),
        "acquisition": _group_ids("acquisition", len(acquisition)),
        RARE_VALID_GROUP: _group_ids("rare-valid", len(rare_valid)),
    }
    notes = (
        "reference = train/ calibration split; validation/heldout-source from frozen "
        "splits; severity/acquisition are synthetic shifts of real validation inputs; "
        "rare-valid are synthetic structures (labeled)"
    )
    return reference, probes, ids, notes


def build_summary(
    run_id: str,
    config: FamiliarityConfig,
    report: Any,
    reference: ReferenceFamiliarity,
    mode: str,
    notes: str,
) -> str:
    from evidence_net.stress_tests.familiarity import FamiliarityReport

    assert isinstance(report, FamiliarityReport)
    lines = [
        f"# Distribution-familiarity run {run_id}",
        "",
        f"- Mode: {mode}",
        f"- Contract: {config.version} (draft; freezes at Research Gate 8)",
        f"- Reference population: {report.n_reference} inputs",
        f"- Threshold: {report.threshold}",
        f"- Notes: {notes}",
        "",
        "## Shift-group detection (fraction flagged unfamiliar)",
        "",
        "| group | detection_rate | mean_distance | n |",
        "| --- | --- | --- | --- |",
    ]
    for name in sorted(report.shift_groups):
        group = report.shift_groups[name]
        lines.append(
            f"| {name} | {group['detection_rate']:.3f} | "
            f"{group['mean_distance']:.3f} | {group['n']} |"
        )
    rare = report.rare_valid
    lines.extend(
        [
            "",
            "## Rare valid structures (evaluated separately)",
            "",
            f"- False-warning rate: {rare['false_warning_rate']:.3f} "
            f"({rare['n']} structures; cap {rare['max_allowed']:.2f})",
            f"- Cap exceeded: {rare['exceeds_cap']}",
            "",
            "## Applicability",
            "",
            report.applicability,
            "",
        ]
    )
    if rare["exceeds_cap"]:
        lines.append(
            "WARNING: the rare-valid false-warning cap is exceeded — Gate 8 "
            "requires the diagnostic to avoid systematically suppressing rare "
            "valid structures; redesign or remove before promotion.\n"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.real and args.synthetic:
        print("FAIL: --real and --synthetic are mutually exclusive", file=sys.stderr)
        return 1
    try:
        config = load_familiarity_config(args.config)
    except FamiliarityError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    mode = "real" if args.real else "synthetic"
    if mode == "real":
        try:
            reference, probes, ids, notes = real_case(config)
        except Exception as exc:
            print(f"FAIL: real mode needs the train/ dataset: {exc}", file=sys.stderr)
            return 1
    else:
        reference, probes, ids, notes = synthetic_case(config)

    report = build_familiarity_report(
        reference,
        probes,
        ids,
        rare_valid_max_false_warning_rate=config.rare_valid_max_false_warning_rate,
    )

    run_id = args.run_id or new_run_id("measure-familiarity")
    dataset_manifest = "synthetic-software-only" if mode == "synthetic" else "dataset-splits-v1"
    manifest = {
        "run_id": run_id,
        "dataset_manifest": dataset_manifest,
        "mode": mode,
        "contract": config.version,
        "test_final_isolation": "confirmed-no-test-noisylr",
    }
    summary = build_summary(run_id, config, report, reference, mode, notes)
    metrics = {**report.as_dict(), "reference": reference.as_dict()}
    run_dir = create_run_bundle(
        args.out,
        run_id,
        config={**config.as_dict(), "mode": mode},
        manifest=manifest,
        metrics=metrics,
        summary=summary,
        reference="no-checkpoint",
    )
    print(f"Run bundle written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
