"""Measure oracle-gating headroom of the trained Detail Proposal (Phase 4).

Usage (from the repository root):::

    python scripts/measure_oracle.py --n-samples 12
    python scripts/measure_oracle.py --proposal-checkpoint checkpoints/train-proposal-gate3/best.pt

Loads the frozen Base and the trained proposal, samples paired validation
groups, and reports:

- Base, ungated candidate, oracle pixel-gated, and oracle patch-gated
  PSNR/SSIM/MAE with group bootstraps;
- pixel and patch coverage / risk;
- structural impact (edge displacement, structural error) of the oracle
  patch output;
- the same comparison against the equal-capacity direct model and the
  classical baseline, when their checkpoints are available.

The oracle sees ground truth; it is a study tool that never runs at
inference. Never touches ``Test_NoisyLR/``.
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
    from evidence_net.evaluation.metrics import all_metrics
    from evidence_net.evaluation.oracle import oracle_decisions
    from evidence_net.evaluation.oracle_report import (
        build_headroom_report,
        headroom_gain,
    )
    from evidence_net.evaluation.statistics import grouped_bootstrap_ci
    from evidence_net.models.factory import build_model
    from evidence_net.models.proposal import BoundedDetailProposal
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id
except ImportError:  # allow running before `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evidence_net.data.loaders import load_npy  # noqa: E402
    from evidence_net.data.manifests import SourceManifest  # noqa: E402
    from evidence_net.data.paths import resolve_dataset_paths  # noqa: E402
    from evidence_net.evaluation.metrics import all_metrics  # noqa: E402
    from evidence_net.evaluation.oracle import oracle_decisions  # noqa: E402
    from evidence_net.evaluation.oracle_report import (  # noqa: E402
        build_headroom_report,
        headroom_gain,
    )
    from evidence_net.evaluation.statistics import grouped_bootstrap_ci  # noqa: E402
    from evidence_net.models.factory import build_model  # noqa: E402
    from evidence_net.models.proposal import BoundedDetailProposal  # noqa: E402
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id  # noqa: E402

from evidence_net.models.reference import (  # noqa: E402
    classical_restoration,
    deterministic_reconstruction,
)

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
        "--base-checkpoint",
        default=REPO_ROOT / "checkpoints" / "train-base-gate2" / "best.pt",
        type=Path,
        help="frozen Base checkpoint",
    )
    parser.add_argument(
        "--proposal-checkpoint",
        default=REPO_ROOT / "checkpoints" / "train-proposal-gate3" / "best.pt",
        type=Path,
        help="trained proposal checkpoint",
    )
    parser.add_argument(
        "--direct-checkpoint",
        default=REPO_ROOT / "checkpoints" / "train-direct-gate2" / "best.pt",
        type=Path,
        help="equal-capacity direct model checkpoint (optional)",
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


def _torch_restorer(model: torch.nn.Module):
    """Wrap a torch model as a numpy restorer (input -> output in [0, 1])."""

    def restore(array: np.ndarray) -> np.ndarray:
        source = np.asarray(array, dtype=np.float32)
        if source.ndim == 2:
            tensor = torch.from_numpy(source)[None, None]
        else:
            tensor = torch.from_numpy(source)[None]
        with torch.no_grad():
            output = model(tensor)
        return output.squeeze().numpy()

    return restore


def _load_trained_checkpoint(path: Path, name: str) -> torch.nn.Module | None:
    from evidence_net.training.config import ModelConfig

    if not path.is_file():
        print(f"[warn] {name} checkpoint not found: {path}", file=sys.stderr)
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_config = ModelConfig(
        name=payload["config"]["model"]["name"],
        hidden_channels=payload["config"]["model"]["hidden_channels"],
        depth=payload["config"]["model"]["depth"],
        amplitude=payload["config"]["model"].get("amplitude", 0.1),
    )
    model = build_model(model_config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _aggregate(values: dict[str, float], *, n_boot: int, seed: int) -> dict[str, float]:
    return grouped_bootstrap_ci(values, n_boot=n_boot, seed=seed).as_dict()


def _metric_table(decisions, *, n_boot: int, seed: int) -> dict[str, dict[str, float]]:
    """Per-group primary metrics for each output family."""
    fields = {
        "base": "base_metrics",
        "candidate": "candidate_metrics",
        "oracle_pixel": "oracle_pixel_metrics",
        "oracle_patch": "oracle_patch_metrics",
    }
    table: dict[str, dict[str, float]] = {}
    for label, field in fields.items():
        for metric in ("psnr", "ssim", "mae"):
            table[f"{label}.{metric}"] = _aggregate(
                {
                    decision.sample_id: float(getattr(decision, field)[metric])
                    for decision in decisions
                },
                n_boot=n_boot,
                seed=seed,
            )
    return table


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

    base = _load_trained_checkpoint(args.base_checkpoint, "base")
    if base is None:
        print("FAIL: frozen base checkpoint required", file=sys.stderr)
        return 1
    proposal_ckpt = _load_trained_checkpoint(args.proposal_checkpoint, "proposal")
    if proposal_ckpt is None:
        print("FAIL: trained proposal checkpoint required", file=sys.stderr)
        return 1
    if not isinstance(proposal_ckpt, BoundedDetailProposal):
        print("FAIL: proposal checkpoint does not contain a BoundedDetailProposal", file=sys.stderr)
        return 1

    # Compute base / proposal / candidate per group.
    bases: list[np.ndarray] = []
    proposals: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    with torch.no_grad():
        for array in inputs:
            tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))[None, None]
            b, d, c = proposal_ckpt.propose(tensor)
            bases.append(b.squeeze().numpy())
            proposals.append(d.squeeze().numpy())
            candidates.append(c.squeeze().numpy())

    decisions = oracle_decisions(sample_ids, bases, proposals, candidates, targets)
    report = build_headroom_report(decisions, bases=bases, proposals=proposals, targets=targets)

    # Extra restorers for context (classical + direct, when available).
    extra: dict[str, dict[str, float]] = {}
    restorers = {
        "classical-median5-bilinear": classical_restoration,
        "deterministic-bilinear": deterministic_reconstruction,
    }
    if args.direct_checkpoint.is_file():
        direct = _load_trained_checkpoint(args.direct_checkpoint, "direct")
        if direct is not None:
            restorers["direct"] = _torch_restorer(direct)
    for name, restorer in restorers.items():
        for metric in ("psnr", "ssim", "mae"):
            values: dict[str, float] = {}
            for index, sample_id in enumerate(sample_ids):
                restored = np.clip(np.asarray(restorer(inputs[index]), dtype=np.float64), 0.0, 1.0)
                metric_value = all_metrics(targets[index], restored)[metric]
                values[sample_id] = float(metric_value)  # type: ignore[arg-type]
            extra[f"{name}.{metric}"] = _aggregate(values, n_boot=1000, seed=args.seed)

    metrics = {
        **report.as_dict(),
        "headroom_gain": headroom_gain(report),
        "comparison": {**_metric_table(decisions, n_boot=1000, seed=args.seed), **extra},
    }
    run_id = args.run_id or new_run_id("oracle-gate3")
    run_dir = create_run_bundle(
        args.out,
        run_id,
        config={
            "phase": 4,
            "pipeline": "sample -> frozen base + proposal -> oracle gating -> report",
            "n_samples": args.n_samples,
            "seed": args.seed,
            "split": args.split,
            "base_checkpoint": str(args.base_checkpoint),
            "proposal_checkpoint": str(args.proposal_checkpoint),
            "direct_checkpoint": str(args.direct_checkpoint),
            "dataset_paths": paths.as_dict(),
        },
        manifest={
            "run_id": run_id,
            "dataset_manifest": TRAIN_MANIFEST.name,
            "splits_manifest": SPLITS_MANIFEST.name,
            "samples": [{"sample_id": sid, "source_group": sid} for sid in sample_ids],
        },
        metrics=metrics,
        summary=_build_summary(run_id, report, metrics, sample_ids),
        reference="frozen base + trained proposal (see summary.md)",
    )
    print(f"Run bundle written to {run_dir}")
    return 0


def _fmt_metric(metrics: dict, key: str) -> str:
    entry = metrics["comparison"][key]
    return f"{entry['mean']:.4f} [{entry['ci_lo']:.4f}, {entry['ci_hi']:.4f}]"


def _build_summary(run_id: str, report, metrics: dict, sample_ids: list[str]) -> str:
    lines = [f"# Oracle gating headroom {run_id}", ""]
    lines.append(f"- {len(sample_ids)} paired validation groups (seeded).")
    lines.append("- Statistical unit: source group; CIs are group bootstraps.\n")
    lines.append("| output | PSNR (dB) | SSIM | MAE |")
    lines.append("| --- | --- | --- | --- |")
    for label in ("base", "candidate", "oracle_pixel", "oracle_patch"):
        lines.append(
            f"| {label} | {_fmt_metric(metrics, label + '.psnr')} | "
            f"{_fmt_metric(metrics, label + '.ssim')} | {_fmt_metric(metrics, label + '.mae')} |"
        )
    for label in ("classical-median5-bilinear", "deterministic-bilinear", "direct"):
        if label + ".psnr" in metrics["comparison"]:
            lines.append(
                f"| {label} | {_fmt_metric(metrics, label + '.psnr')} | "
                f"{_fmt_metric(metrics, label + '.ssim')} | "
                f"{_fmt_metric(metrics, label + '.mae')} |"
            )
    lines.append("")
    lines.append("## Coverage / risk")
    lines.append(
        f"- pixel coverage {report.coverage['pixel']['mean']:.3f}, "
        f"risk {report.risk['pixel']['mean']:.3f}"
    )
    lines.append(
        f"- patch coverage {report.coverage['patch']['mean']:.3f}, "
        f"risk {report.risk['patch']['mean']:.3f}"
    )
    lines.append("")
    lines.append("## Headroom gains (oracle patch vs Base)")
    gain = headroom_gain(report)
    lines.append(
        f"- PSNR {gain['oracle_vs_base_psnr']:+.4f} dB, "
        f"SSIM {gain['oracle_vs_base_ssim']:+.4f}, MAE {gain['oracle_vs_base_mae']:+.4f}"
    )
    if report.structural_impact:
        lines.append("")
        lines.append("## Structural impact (base vs oracle patch)")
        for label in ("edge_displacement_px", "structural_error"):
            base_value = report.structural_impact.get(f"{label}.base", {}).get("mean")
            oracle_value = report.structural_impact.get(f"{label}.oracle_patch", {}).get("mean")
            if base_value is not None and oracle_value is not None:
                lines.append(
                    f"- {label}: base {base_value:.4f} -> oracle patch "
                    f"{oracle_value:.4f} (delta {oracle_value - base_value:+.4f})"
                )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
