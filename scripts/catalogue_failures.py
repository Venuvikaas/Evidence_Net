"""Characterize natural Base Reconstruction errors by structural region type.

Usage (from the repository root)::

    python scripts/catalogue_failures.py --checkpoint checkpoints/train-base-gate2/best.pt

Runs the trained model over a seeded validation sample and decomposes the
per-pixel error by structural region: edge-adjacent bands, flat regions,
periodic (high edge-density) regions, and low-frequency background. Reports
worst samples and writes a run bundle. This is the Phase 3 failure catalogue
(input for the Research Gate 2 structural-group assessment).
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
    from evidence_net.evaluation.metrics import binary_edges, edge_magnitude
    from evidence_net.models.factory import build_model
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id
except ImportError:  # allow running before `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evidence_net.data.loaders import load_npy  # noqa: E402
    from evidence_net.data.manifests import SourceManifest  # noqa: E402
    from evidence_net.data.paths import resolve_dataset_paths  # noqa: E402
    from evidence_net.evaluation.metrics import (  # noqa: E402
        binary_edges,
        edge_magnitude,
    )
    from evidence_net.models.factory import build_model  # noqa: E402
    from evidence_net.reporting.run_bundle import (  # noqa: E402
        create_run_bundle,
        new_run_id,
    )

MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
TRAIN_MANIFEST = MANIFESTS_DIR / "official-train-source-v1.json"
SPLITS_MANIFEST = MANIFESTS_DIR / "dataset-splits-v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=REPO_ROOT / "checkpoints" / "train-base-gate2" / "best.pt",
        type=Path,
    )
    parser.add_argument("--n-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", default=REPO_ROOT / "runs", type=Path)
    return parser.parse_args()


def _load_checkpoint(path: Path) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_config = payload["config"]["model"]
    from evidence_net.training.config import ModelConfig

    model = build_model(
        ModelConfig(
            name=model_config["name"],
            hidden_channels=model_config["hidden_channels"],
            depth=model_config["depth"],
        )
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _select_ids(split: str, n: int, seed: int) -> list[str]:
    data = json.loads(SPLITS_MANIFEST.read_text(encoding="utf-8"))
    ids = sorted(sid for sid, label in data["assignments"].items() if label == split)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ids), size=min(n, len(ids)), replace=False)
    return [ids[int(i)] for i in np.sort(indices)]


def _region_masks(target: np.ndarray) -> dict[str, np.ndarray]:
    """Structural region masks on the 256x256 target grid."""
    edges = binary_edges(target)
    magnitude = edge_magnitude(target)
    # Edge band: dilation of the edge map by 2 px (4-neighborhood).
    band = edges.copy()
    for _ in range(2):
        shifted = np.zeros_like(band)
        shifted[1:, :] |= band[:-1, :]
        shifted[:-1, :] |= band[1:, :]
        shifted[:, 1:] |= band[:, :-1]
        shifted[:, :-1] |= band[:, 1:]
        band = band | shifted
    flat = (magnitude < 0.1) & ~band
    periodic = magnitude >= 0.5  # high edge density
    return {
        "edge_band": band,
        "flat": flat,
        "periodic": periodic,
        "all": np.ones_like(edges, dtype=bool),
    }


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        print(f"FAIL checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    manifest = SourceManifest.from_dict(json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8")))
    pairs: dict[str, dict[str, Path]] = {}
    for entry in manifest.files:
        rel = Path(entry.relative_path)
        sid = rel.stem
        if "NoisyLR" in rel.parts:
            pairs.setdefault(sid, {})["input"] = rel
        elif "GT" in rel.parts:
            pairs.setdefault(sid, {})["target"] = rel
    sample_ids = _select_ids("validation", args.n_samples, args.seed)
    paths = resolve_dataset_paths()
    model = _load_checkpoint(args.checkpoint)

    region_names = ("edge_band", "flat", "periodic", "all")
    per_region: dict[str, list[float]] = {name: [] for name in region_names}
    worst: list[dict] = []
    with torch.no_grad():
        for sample_id in sample_ids:
            pair_rel = pairs[sample_id]
            input_array = load_npy(paths.train_dir / pair_rel["input"])
            target = load_npy(paths.train_dir / pair_rel["target"])
            tensor = torch.from_numpy(input_array)[None, None]
            prediction = model(tensor).squeeze().numpy()
            error = np.abs(prediction - target)
            masks = _region_masks(target)
            row: dict = {"sample_id": sample_id, "mae": float(error.mean())}
            for name, mask in masks.items():
                per_region[name].append(float(error[mask].mean()))
                row[f"mae_{name}"] = float(error[mask].mean())
            worst.append(row)

    worst_sorted = sorted(worst, key=lambda r: r["mae"], reverse=True)[:5]
    run_id = args.run_id or new_run_id("failure-catalogue")
    summary_lines = [
        f"# Base Reconstruction failure catalogue {run_id}",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Samples: {len(sample_ids)} validation groups (seed {args.seed})",
        "- Error decomposed by structural region on the target grid.\n",
        "| region | mean MAE |",
        "| --- | --- |",
    ]
    for name in ("edge_band", "periodic", "flat", "all"):
        summary_lines.append(f"| {name} | {np.mean(per_region[name]):.5f} |")
    summary_lines.extend(
        [
            "",
            "## Worst samples",
            "",
            "| sample | overall MAE | edge-band MAE | flat MAE |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in worst_sorted:
        summary_lines.append(
            f"| {row['sample_id']} | {row['mae']:.5f} | "
            f"{row['mae_edge_band']:.5f} | {row['mae_flat']:.5f} |"
        )
    run_dir = create_run_bundle(
        args.out,
        run_id,
        config={
            "phase": 3,
            "kind": "failure-catalogue",
            "checkpoint": str(args.checkpoint),
            "n_samples": args.n_samples,
            "seed": args.seed,
        },
        manifest={
            "run_id": run_id,
            "dataset_manifest": TRAIN_MANIFEST.name,
            "splits_manifest": SPLITS_MANIFEST.name,
            "samples": sample_ids,
        },
        metrics={
            "per_region_mae": {name: float(np.mean(v)) for name, v in per_region.items()},
            "worst_samples": worst_sorted,
        },
        summary="\n".join(summary_lines),
        reference=str(args.checkpoint),
    )
    print(f"Run bundle written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
