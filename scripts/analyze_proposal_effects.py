"""Analyze proposal benefit and harm by structural and degradation group.

Usage (from the repository root):::

    python scripts/analyze_proposal_effects.py --n-samples 12
    python scripts/analyze_proposal_effects.py \
        --proposal-checkpoint checkpoints/train-proposal-gate3v2/best.pt

For each paired validation group the script compares the ungated candidate
against the frozen Base and decomposes the proposal's effect (benefit where
the candidate error is lower, harm where it is higher) by structural region
(edge band / periodic / flat) and by group. It also measures the oracle
patch-gate acceptance rate per region, so benefit and harm are understood by
structure rather than only averaged. Writes a run bundle with a markdown
summary and per-group JSON.
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
    from evidence_net.evaluation.oracle import oracle_decisions
    from evidence_net.models.factory import build_model
    from evidence_net.models.proposal import BoundedDetailProposal
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id
except ImportError:  # allow running before `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evidence_net.data.loaders import load_npy  # noqa: E402
    from evidence_net.data.manifests import SourceManifest  # noqa: E402
    from evidence_net.data.paths import resolve_dataset_paths  # noqa: E402
    from evidence_net.evaluation.metrics import binary_edges, edge_magnitude  # noqa: E402
    from evidence_net.evaluation.oracle import oracle_decisions  # noqa: E402
    from evidence_net.models.factory import build_model  # noqa: E402
    from evidence_net.models.proposal import BoundedDetailProposal  # noqa: E402
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id  # noqa: E402

MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
TRAIN_MANIFEST = MANIFESTS_DIR / "official-train-source-v1.json"
SPLITS_MANIFEST = MANIFESTS_DIR / "dataset-splits-v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=12, help="paired groups to evaluate")
    parser.add_argument("--seed", type=int, default=0, help="seeded sample selection")
    parser.add_argument("--split", default="validation", help="development split")
    parser.add_argument("--run-id", default=None, help="explicit run id")
    parser.add_argument("--out", default=REPO_ROOT / "runs", type=Path, help="runs directory")
    parser.add_argument(
        "--proposal-checkpoint",
        default=REPO_ROOT / "checkpoints" / "train-proposal-gate3v2" / "best.pt",
        type=Path,
        help="trained proposal checkpoint (frozen Base inside)",
    )
    return parser.parse_args()


def _select_ids(split: str, n: int, seed: int) -> list[str]:
    data = json.loads(SPLITS_MANIFEST.read_text(encoding="utf-8"))
    ids = sorted(sample_id for sample_id, label in data["assignments"].items() if label == split)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ids), size=min(n, len(ids)), replace=False)
    return [ids[int(index)] for index in np.sort(indices)]


def _region_masks(target: np.ndarray) -> dict[str, np.ndarray]:
    """Structural region masks on the target grid (same as the catalogue)."""
    edges = binary_edges(target)
    magnitude = edge_magnitude(target)
    band = edges.copy()
    for _ in range(2):
        shifted = np.zeros_like(band)
        shifted[1:, :] |= band[:-1, :]
        shifted[:-1, :] |= band[1:, :]
        shifted[:, 1:] |= band[:, :-1]
        shifted[:, :-1] |= band[:, 1:]
        band = band | shifted
    flat = (magnitude < 0.1) & ~band
    periodic = magnitude >= 0.5
    return {
        "edge_band": band,
        "flat": flat,
        "periodic": periodic,
        "all": np.ones_like(edges, dtype=bool),
    }


def _load_proposal_checkpoint(path: Path) -> BoundedDetailProposal:
    from evidence_net.training.config import ModelConfig

    if not path.is_file():
        raise SystemExit(f"FAIL: checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_config = ModelConfig(
        name=payload["config"]["model"]["name"],
        hidden_channels=payload["config"]["model"]["hidden_channels"],
        depth=payload["config"]["model"]["depth"],
        amplitude=payload["config"]["model"].get("amplitude", 0.1),
    )
    model = build_model(model_config)
    if not isinstance(model, BoundedDetailProposal):
        raise SystemExit("FAIL: checkpoint is not a BoundedDetailProposal")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    if not TRAIN_MANIFEST.is_file() or not SPLITS_MANIFEST.is_file():
        print("FAIL: frozen manifests missing; run Phase 1 first", file=sys.stderr)
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
    sample_ids = _select_ids(args.split, args.n_samples, args.seed)
    paths = resolve_dataset_paths()
    model = _load_proposal_checkpoint(args.proposal_checkpoint)

    region_names = ("edge_band", "flat", "periodic", "all")
    benefit: dict[str, list[float]] = {name: [] for name in region_names}
    harm: dict[str, list[float]] = {name: [] for name in region_names}
    accept_rate: dict[str, list[float]] = {name: [] for name in region_names}
    per_group: list[dict] = []
    with torch.no_grad():
        for sample_id in sample_ids:
            pair_rel = pairs[sample_id]
            input_array = load_npy(paths.train_dir / pair_rel["input"])
            target = load_npy(paths.train_dir / pair_rel["target"])
            tensor = torch.from_numpy(input_array)[None, None]
            b, d, c = model.propose(tensor)
            base_out = b.squeeze().numpy()
            candidate = c.squeeze().numpy()
            proposal = d.squeeze().numpy()
            # Oracle patch gate for the acceptance-rate breakdown.
            decisions = oracle_decisions(
                [sample_id], [base_out], [proposal], [candidate], [target]
            )[0]
            error_base = np.abs(base_out - target)
            error_candidate = np.abs(candidate - target)
            masks = _region_masks(target)
            row: dict = {"sample_id": sample_id, "benefit": 0.0, "harm": 0.0}
            for name, mask in masks.items():
                gain = (error_base[mask] - error_candidate[mask]).mean()
                if gain >= 0:
                    benefit[name].append(float(gain))
                    harm[name].append(0.0)
                else:
                    benefit[name].append(0.0)
                    harm[name].append(float(-gain))
                accept_rate[name].append(float(decisions.patch_gate[mask].mean()))
                row[f"gain_{name}"] = float(gain)
            # Group harm is the worst region where the candidate hurts; a
            # group can hurt in one region while helping on aggregate.
            row["benefit"] = max(float(row[f"gain_{name}"]) for name in region_names)
            row["harm"] = (
                max(-float(row[f"gain_{name}"]) for name in region_names if row[f"gain_{name}"] < 0)
                if any(row[f"gain_{name}"] < 0 for name in region_names)
                else 0.0
            )
            per_group.append(row)

    worst_harm = sorted(per_group, key=lambda r: r["harm"], reverse=True)[:5]
    best_benefit = sorted(per_group, key=lambda r: r["benefit"], reverse=True)[:5]

    run_id = args.run_id or new_run_id("proposal-effects")
    summary_lines = [
        f"# Proposal effect analysis {run_id}",
        "",
        f"- Checkpoint: `{args.proposal_checkpoint}`",
        f"- Samples: {len(sample_ids)} validation groups (seed {args.seed})",
        "- Benefit = mean error reduction of the candidate vs Base in a region;",
        "  harm = mean error increase. Oracle patch acceptance per region.\\n",
        "| region | mean benefit | mean harm | oracle accept |",
        "| --- | --- | --- | --- |",
    ]
    for name in region_names:
        summary_lines.append(
            f"| {name} | {np.mean(benefit[name]):.5f} | {np.mean(harm[name]):.5f} | "
            f"{np.mean(accept_rate[name]):.3f} |"
        )
    summary_lines.extend(
        [
            "",
            "## Most harmful groups (candidate worse than Base)",
            "",
            "| sample | harm | benefit |",
            "| --- | --- | --- |",
        ]
    )
    for row in worst_harm:
        summary_lines.append(f"| {row['sample_id']} | {row['harm']:.5f} | {row['benefit']:.5f} |")
    summary_lines.extend(
        [
            "",
            "## Most beneficial groups",
            "",
            "| sample | benefit | harm |",
            "| --- | --- | --- |",
        ]
    )
    for row in best_benefit:
        summary_lines.append(f"| {row['sample_id']} | {row['benefit']:.5f} | {row['harm']:.5f} |")

    run_dir = create_run_bundle(
        args.out,
        run_id,
        config={
            "phase": 4,
            "pipeline": "sample -> frozen base + proposal -> structural benefit/harm",
            "n_samples": args.n_samples,
            "seed": args.seed,
            "split": args.split,
            "proposal_checkpoint": str(args.proposal_checkpoint),
            "dataset_paths": paths.as_dict(),
        },
        manifest={
            "run_id": run_id,
            "dataset_manifest": TRAIN_MANIFEST.name,
            "splits_manifest": SPLITS_MANIFEST.name,
            "samples": [{"sample_id": sid, "source_group": sid} for sid in sample_ids],
        },
        metrics={
            "region_benefit": {name: float(np.mean(values)) for name, values in benefit.items()},
            "region_harm": {name: float(np.mean(values)) for name, values in harm.items()},
            "region_accept_rate": {
                name: float(np.mean(values)) for name, values in accept_rate.items()
            },
            "per_group": per_group,
        },
        summary="\n".join(summary_lines),
        reference="frozen base + trained proposal (see summary.md)",
    )
    print(f"Run bundle written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
