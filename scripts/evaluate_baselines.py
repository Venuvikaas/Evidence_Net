"""Evaluate the classical baselines on a seeded sample of the validation split.

Usage (from the repository root)::

    python scripts/evaluate_baselines.py --n-samples 8 --seed 0 --split validation

Loads the frozen train manifest and split assignments, samples ``n_samples``
pairs from the requested development split, runs the deterministic reference
and the classical restoration baseline over identical inputs, computes the
contract metrics per group, and writes a full run bundle with comparison
sheets and a markdown report. Never touches ``Test_NoisyLR/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from evidence_net.data.loaders import load_npy
    from evidence_net.data.manifests import SourceManifest
    from evidence_net.data.paths import resolve_dataset_paths
    from evidence_net.inference.baseline import evaluate_restorers, run_restorer
    from evidence_net.models.reference import classical_restoration, deterministic_reconstruction
    from evidence_net.reporting.comparison_report import (
        write_comparison_report,
        write_comparison_sheet,
    )
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id
except ImportError:  # allow running before `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evidence_net.data.loaders import load_npy  # noqa: E402
    from evidence_net.data.manifests import SourceManifest  # noqa: E402
    from evidence_net.data.paths import resolve_dataset_paths  # noqa: E402
    from evidence_net.inference.baseline import evaluate_restorers, run_restorer  # noqa: E402
    from evidence_net.models.reference import (  # noqa: E402
        classical_restoration,
        deterministic_reconstruction,
    )
    from evidence_net.reporting.comparison_report import (  # noqa: E402
        write_comparison_report,
        write_comparison_sheet,
    )
    from evidence_net.reporting.run_bundle import (  # noqa: E402
        create_run_bundle,
        new_run_id,
    )

MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
TRAIN_MANIFEST = MANIFESTS_DIR / "official-train-source-v1.json"
SPLITS_MANIFEST = MANIFESTS_DIR / "dataset-splits-v1.json"

RESTORERS = {
    "deterministic-bilinear": deterministic_reconstruction,
    "classical-median5-bilinear": classical_restoration,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=8, help="paired groups to evaluate")
    parser.add_argument("--seed", type=int, default=0, help="seeded sample selection")
    parser.add_argument("--split", default="validation", help="development split to sample from")
    parser.add_argument("--run-id", default=None, help="explicit run id")
    parser.add_argument("--out", default=REPO_ROOT / "runs", type=Path, help="runs directory")
    return parser.parse_args()


def load_splits() -> tuple[dict[str, str], dict[str, int]]:
    """Load frozen split assignments and counts."""
    data = json.loads(SPLITS_MANIFEST.read_text(encoding="utf-8"))
    return data["assignments"], data["counts"]


def select_sample_ids(
    assignments: dict[str, str], split: str, n_samples: int, seed: int
) -> list[str]:
    """Seeded, deterministic selection from one development split."""
    ids = sorted(sample_id for sample_id, label in assignments.items() if label == split)
    if not ids:
        print(f"FAIL: split '{split}' has no samples", file=sys.stderr)
        raise SystemExit(1)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ids), size=min(n_samples, len(ids)), replace=False)
    return [ids[int(index)] for index in np.sort(indices)]


def build_pair_index(manifest: SourceManifest) -> dict[str, dict[str, Path]]:
    """Map sample id -> {input, target} relative paths from the train manifest."""
    pairs: dict[str, dict[str, Path]] = {}
    for entry in manifest.files:
        rel = Path(entry.relative_path)
        sample_id = rel.stem
        if "NoisyLR" in rel.parts:
            pairs.setdefault(sample_id, {})["input"] = rel
        elif "GT" in rel.parts:
            pairs.setdefault(sample_id, {})["target"] = rel
    return pairs


def main() -> int:
    args = parse_args()
    if args.n_samples < 1:
        print("FAIL: --n-samples must be >= 1", file=sys.stderr)
        return 1
    if not TRAIN_MANIFEST.is_file() or not SPLITS_MANIFEST.is_file():
        print("FAIL: frozen manifests missing; run Phase 1 first", file=sys.stderr)
        return 1

    manifest = SourceManifest.from_dict(json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8")))
    assignments, _ = load_splits()
    sample_ids = select_sample_ids(assignments, args.split, args.n_samples, args.seed)
    pair_index = build_pair_index(manifest)
    missing = [sample_id for sample_id in sample_ids if sample_id not in pair_index]
    if missing:
        print(f"FAIL: samples missing from train manifest: {missing}", file=sys.stderr)
        return 1

    paths = resolve_dataset_paths()
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    group_ids: list[str] = []
    for sample_id in sample_ids:
        rel = pair_index[sample_id]
        if "input" not in rel or "target" not in rel:
            print(f"FAIL: incomplete pair for {sample_id}", file=sys.stderr)
            return 1
        inputs.append(load_npy(paths.train_dir / rel["input"]))
        targets.append(load_npy(paths.train_dir / rel["target"]))
        group_ids.append(sample_id)

    results = evaluate_restorers(inputs, targets, group_ids, RESTORERS, n_boot=1000, seed=args.seed)
    run_id = args.run_id or new_run_id("baseline-eval")
    run_dir = create_run_bundle(
        args.out,
        run_id,
        config={
            "phase": 2,
            "pipeline": "sample -> infer -> evaluate -> save artifacts -> report",
            "n_samples": args.n_samples,
            "seed": args.seed,
            "split": args.split,
            "restorers": list(RESTORERS),
            "dataset_paths": paths.as_dict(),
        },
        manifest={
            "run_id": run_id,
            "dataset_manifest": TRAIN_MANIFEST.name,
            "splits_manifest": SPLITS_MANIFEST.name,
            "samples": [
                {"sample_id": sample_id, "source_group": sample_id} for sample_id in group_ids
            ],
        },
        metrics={name: result.aggregates for name, result in sorted(results.items())},
        summary=_build_summary(run_id, results, sample_ids, args.split),
        reference=(
            "classical baselines (no checkpoint): "
            "deterministic-bilinear, classical-median5-bilinear"
        ),
    )

    artifacts_dir = run_dir / "artifacts"
    deterministic = RESTORERS["deterministic-bilinear"]
    predictions = run_restorer(inputs, deterministic)
    for index, (sample_id, input_, prediction, target) in enumerate(
        zip(sample_ids, inputs, predictions, targets, strict=True)
    ):
        per_group = results["deterministic-bilinear"].per_group_metrics[sample_id]
        write_comparison_sheet(artifacts_dir, index, input_, prediction, target, per_group)
    write_comparison_report(
        run_dir,
        results,
        sample_ids=sample_ids,
        n_samples=len(sample_ids),
        split_label=args.split,
    )
    print(f"Run bundle written to {run_dir}")
    return 0


def _build_summary(run_id: str, results: dict, sample_ids: list[str], split: str) -> str:
    lines = [f"# Baseline evaluation {run_id}", ""]
    lines.append(f"- Split: `{split}` — {len(sample_ids)} paired groups (seeded).")
    lines.append("- Statistical unit: source group; CIs are group bootstraps.\n")
    lines.append("| restorer | PSNR (dB) | SSIM | MAE |")
    lines.append("| --- | --- | --- | --- |")
    for name, result in sorted(results.items()):
        agg = result.aggregates
        lines.append(
            f"| {name} | {agg['psnr']['mean']:.4f} "
            f"[{agg['psnr']['ci_lo']:.4f}, {agg['psnr']['ci_hi']:.4f}] | "
            f"{agg['ssim']['mean']:.4f} | {agg['mae']['mean']:.4f} |"
        )
    lines.append("")
    lines.append("See `comparison-report.md` for full metrics and sheets.")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
