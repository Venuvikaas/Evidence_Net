"""Governed comparison: classical, direct, and Base models on the same sample.

Usage (from the repository root)::

    python scripts/compare_restoration.py --n-samples 8 --seed 0
    python scripts/compare_restoration.py --checkpoints checkpoints/base-dev-1

Loads the frozen train manifest and splits, samples paired validation
groups, and evaluates the classical baselines plus every trained model whose
``best.pt`` checkpoint is registered under ``checkpoints/<run-id>/``. All
models see identical inputs (paired comparison) and results are aggregated
with group bootstraps, then written to a run bundle with comparison sheets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from evidence_net.data.loaders import load_npy
    from evidence_net.data.manifests import SourceManifest
    from evidence_net.data.paths import resolve_dataset_paths
    from evidence_net.inference.baseline import evaluate_restorers
    from evidence_net.models.factory import build_model
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
    from evidence_net.inference.baseline import evaluate_restorers  # noqa: E402
    from evidence_net.models.factory import build_model  # noqa: E402
    from evidence_net.reporting.comparison_report import (  # noqa: E402
        write_comparison_report,
        write_comparison_sheet,
    )
    from evidence_net.reporting.run_bundle import (  # noqa: E402
        create_run_bundle,
        new_run_id,
    )

from evidence_net.models.reference import (  # noqa: E402
    Restorer,
    classical_restoration,
    deterministic_reconstruction,
)

MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
TRAIN_MANIFEST = MANIFESTS_DIR / "official-train-source-v1.json"
SPLITS_MANIFEST = MANIFESTS_DIR / "dataset-splits-v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=8, help="paired groups to evaluate")
    parser.add_argument("--seed", type=int, default=0, help="seeded sample selection")
    parser.add_argument("--split", default="validation", help="development split")
    parser.add_argument("--run-id", default=None, help="explicit run id")
    parser.add_argument("--out", default=REPO_ROOT / "runs", type=Path, help="runs directory")
    parser.add_argument(
        "--checkpoints",
        default=REPO_ROOT / "checkpoints",
        type=Path,
        help="directory of <run-id>/best.pt checkpoints to include",
    )
    return parser.parse_args()


def _load_splits() -> dict[str, dict[str, str]]:
    data = json.loads(SPLITS_MANIFEST.read_text(encoding="utf-8"))
    return data["assignments"]


def _select_ids(split: str, n: int, seed: int) -> list[str]:
    assignments = _load_splits()
    ids = sorted(sample_id for sample_id, label in assignments.items() if label == split)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ids), size=min(n, len(ids)), replace=False)
    return [ids[int(index)] for index in np.sort(indices)]


def _pair_index(manifest: SourceManifest) -> dict[str, dict[str, Path]]:
    pairs: dict[str, dict[str, Path]] = {}
    for entry in manifest.files:
        rel = Path(entry.relative_path)
        sample_id = rel.stem
        if "NoisyLR" in rel.parts:
            pairs.setdefault(sample_id, {})["input"] = rel
        elif "GT" in rel.parts:
            pairs.setdefault(sample_id, {})["target"] = rel
    return pairs


def _load_checkpoint_models(checkpoints_dir: Path) -> dict[str, Restorer]:
    """Load every <run-id>/best.pt as a torch model wrapper."""
    restorers: dict[str, Restorer] = {}
    if not checkpoints_dir.is_dir():
        return restorers
    for run_dir in sorted(checkpoints_dir.iterdir()):
        best = run_dir / "best.pt"
        if not best.is_file():
            continue
        payload = torch.load(best, map_location="cpu", weights_only=False)
        model = _rebuild_from_config(payload["config"])
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        restorers[run_dir.name] = _torch_restorer(model)
    return restorers


def _torch_restorer(model: torch.nn.Module) -> Restorer:
    """Wrap a torch model as a numpy Restorer with clip-to-[0,1] outputs."""

    def restore(array: np.ndarray) -> np.ndarray:
        # 2D (H, W) -> (1, 1, H, W); 3D (C, H, W) -> (1, C, H, W).
        source = np.asarray(array, dtype=np.float32)
        if source.ndim == 2:
            tensor = torch.from_numpy(source)[None, None]
        else:
            tensor = torch.from_numpy(source)[None]
        with torch.no_grad():
            output = model(tensor)
        # Match the 2D convention of the numpy restorers: drop batch and
        # singleton channel dims.
        return output.squeeze().numpy()

    return restore


def _rebuild_from_config(config: dict) -> torch.nn.Module:
    from evidence_net.training.config import ModelConfig

    model_config = ModelConfig(
        name=config["model"]["name"],
        hidden_channels=config["model"]["hidden_channels"],
        depth=config["model"]["depth"],
    )
    return build_model(model_config)


def main() -> int:
    args = parse_args()
    if not TRAIN_MANIFEST.is_file() or not SPLITS_MANIFEST.is_file():
        print("FAIL: frozen manifests missing; run Phase 1 first", file=sys.stderr)
        return 1
    manifest = SourceManifest.from_dict(json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8")))
    sample_ids = _select_ids(args.split, args.n_samples, args.seed)
    pair_index = _pair_index(manifest)
    missing = [sample_id for sample_id in sample_ids if sample_id not in pair_index]
    if missing:
        print(f"FAIL: samples missing from train manifest: {missing}", file=sys.stderr)
        return 1
    paths = resolve_dataset_paths()
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for sample_id in sample_ids:
        rel = pair_index[sample_id]
        inputs.append(load_npy(paths.train_dir / rel["input"]))
        targets.append(load_npy(paths.train_dir / rel["target"]))

    restorers: dict[str, Restorer] = {
        "classical-median5-bilinear": classical_restoration,
        "deterministic-bilinear": deterministic_reconstruction,
    }
    learned = _load_checkpoint_models(args.checkpoints)
    restorers.update(learned)

    results = evaluate_restorers(
        inputs, targets, sample_ids, restorers, n_boot=1000, seed=args.seed
    )
    run_id = args.run_id or new_run_id("restoration-compare")
    run_dir = create_run_bundle(
        args.out,
        run_id,
        config={
            "phase": 3,
            "pipeline": "sample -> infer -> evaluate -> report",
            "n_samples": args.n_samples,
            "seed": args.seed,
            "split": args.split,
            "restorers": sorted(restorers),
            "dataset_paths": paths.as_dict(),
        },
        manifest={
            "run_id": run_id,
            "dataset_manifest": TRAIN_MANIFEST.name,
            "splits_manifest": SPLITS_MANIFEST.name,
            "samples": [{"sample_id": sid, "source_group": sid} for sid in sample_ids],
        },
        metrics={name: result.aggregates for name, result in sorted(results.items())},
        summary=_build_summary(run_id, results, sample_ids, args.split),
        reference="classical + learned models (see comparison-report.md)",
    )
    artifacts_dir = run_dir / "artifacts"
    for index, (sample_id, input_, target) in enumerate(
        zip(sample_ids, inputs, targets, strict=True)
    ):
        per_group = results["deterministic-bilinear"].per_group_metrics[sample_id]
        prediction = np.clip(deterministic_reconstruction(input_), 0.0, 1.0)
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
    lines = [f"# Restoration comparison {run_id}", ""]
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
