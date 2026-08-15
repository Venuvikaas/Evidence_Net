"""Experiment provenance for training runs.

Every governed training run writes a run bundle (per
``docs/run-and-artifact-contract.md``) containing the validated config, the
frozen dataset manifests it consumed, the environment (python, platform,
torch/numpy versions), the training history, and the checkpoint reference —
so the run can be reproduced and audited.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evidence_net.reporting.run_bundle import create_run_bundle, environment_text
from evidence_net.training.config import TrainConfig

MANIFESTS_DIR = Path(__file__).resolve().parents[3] / "data" / "manifests"


def capture_environment() -> str:
    """Extend the base environment text with the ML stack versions."""
    lines = [environment_text().rstrip("\n")]
    try:
        import torch

        lines.append(f"  torch: {torch.__version__}")
    except Exception:  # pragma: no cover - diagnostic path
        lines.append("  torch: MISSING")
    return "\n".join(lines) + "\n"


def manifest_hashes() -> dict[str, str]:
    """sha256 of every frozen manifest consumed by training (none is tracked here)."""
    import hashlib

    hashes: dict[str, str] = {}
    for path in sorted(MANIFESTS_DIR.glob("*.json")):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def create_experiment_bundle(
    runs_dir: Path,
    run_id: str,
    *,
    config: TrainConfig,
    model_summary: dict[str, Any],
    history: list[dict[str, float | int]],
    checkpoint_ref: str,
    extra_metrics: dict[str, Any] | None = None,
) -> Path:
    """Create a run bundle for one training run and return its directory."""
    metrics: dict[str, Any] = dict(extra_metrics or {})
    if history:
        last = history[-1]
        metrics["final_train_loss"] = last["train_loss"]
        if "val_loss" in last and last["val_loss"] == last["val_loss"]:
            metrics["final_val_loss"] = last["val_loss"]
    manifest = {
        "run_id": run_id,
        "kind": "training",
        "phase": 3,
        "config": config.as_dict(),
        "model": model_summary,
        "dataset_manifests": manifest_hashes(),
        "history": history,
    }
    summary = (
        f"# Training run {run_id}\n\n"
        f"- Model: `{model_summary.get('name', 'unknown')}` "
        f"({model_summary.get('n_params', '?')} parameters)\n"
        f"- Config: epochs={config.epochs}, batch={config.batch_size}, "
        f"lr={config.learning_rate}, seed={config.seed}\n"
        f"- Split: `{config.data.split}` ({config.data.n_samples} samples, "
        f"seed {config.data.seed})\n"
        f"- Final train loss: {metrics.get('final_train_loss', 'n/a')}\n"
        f"- Final val loss: {metrics.get('final_val_loss', 'n/a')}\n"
    )
    return create_run_bundle(
        runs_dir,
        run_id,
        config={"kind": "training", "phase": 3, **config.as_dict()},
        manifest=manifest,
        metrics=metrics,
        summary=summary,
        reference=checkpoint_ref,
        environment=capture_environment(),
    )


def write_checkpoint_ref(run_dir: Path, checkpoint_path: Path) -> None:
    """Record the checkpoint path in the bundle's checkpoint-or-reference file."""
    (run_dir / "checkpoint-or-reference.txt").write_text(
        str(checkpoint_path) + "\n", encoding="utf-8"
    )
