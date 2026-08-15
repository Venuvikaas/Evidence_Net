"""Train a Base Reconstruction or direct-restoration model and persist provenance.

Usage (from the repository root)::

    python scripts/train_base.py --config configs/model/base-smoke.yaml
    python scripts/train_base.py --config configs/model/base-dev.yaml --run-id base-dev-1

Loads a validated config, builds a torch dataset from the frozen train
manifest and splits, trains with the reproducible trainer (checkpointing +
resume), and writes a run bundle with environment capture, history, and
checkpoint reference. Never touches ``Test_NoisyLR/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from evidence_net.data.paths import resolve_dataset_paths
    from evidence_net.losses.base_losses import BaseLoss
    from evidence_net.models.factory import build_model, model_summary
    from evidence_net.reporting.run_bundle import new_run_id
    from evidence_net.training.config import load_config
    from evidence_net.training.dataset import RestorationDataset
    from evidence_net.training.provenance import create_experiment_bundle
    from evidence_net.training.trainer import Trainer, set_seed
except ImportError:  # allow running before `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evidence_net.data.paths import resolve_dataset_paths  # noqa: E402
    from evidence_net.losses.base_losses import BaseLoss  # noqa: E402
    from evidence_net.models.factory import build_model, model_summary  # noqa: E402
    from evidence_net.reporting.run_bundle import new_run_id  # noqa: E402
    from evidence_net.training.config import load_config  # noqa: E402
    from evidence_net.training.dataset import RestorationDataset  # noqa: E402
    from evidence_net.training.provenance import create_experiment_bundle  # noqa: E402
    from evidence_net.training.trainer import Trainer, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=REPO_ROOT / "configs" / "model" / "base-smoke.yaml",
        type=Path,
        help="training config YAML",
    )
    parser.add_argument("--run-id", default=None, help="explicit run id")
    parser.add_argument("--out", default=REPO_ROOT / "runs", type=Path, help="runs directory")
    parser.add_argument(
        "--resume", default=None, type=Path, help="checkpoint to resume from (last.pt)"
    )
    parser.add_argument("--device", default=None, help="torch device override (default: auto)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.is_file():
        print(f"FAIL config not found: {args.config}", file=sys.stderr)
        return 1
    config = load_config(args.config)
    set_seed(config.seed)
    paths = resolve_dataset_paths()
    run_id = args.run_id or new_run_id("train-base")
    # Checkpoints live in <repo>/checkpoints/<run-id>/ so the comparison
    # script can discover trained models by run id.
    checkpoint_dir = REPO_ROOT / "checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = RestorationDataset(
        paths.train_dir,
        split=config.data.split,
        n_samples=config.data.n_samples,
        seed=config.data.seed,
    )
    val_dataset = RestorationDataset(
        paths.train_dir,
        split="validation",
        n_samples=min(16, config.data.n_samples),
        seed=config.data.seed,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = build_model(config.model)
    loss_fn = BaseLoss(config.loss)
    device = torch.device(args.device) if args.device is not None else None
    trainer = Trainer(
        model,
        config,
        train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=checkpoint_dir,
        resume_from=args.resume,
    )
    if args.resume is not None:
        print(f"resumed from {args.resume} at epoch {trainer.start_epoch}")
    trainer.fit(log_every=1)

    run_dir = create_experiment_bundle(
        args.out,
        run_id,
        config=config,
        model_summary=model_summary(model),
        history=trainer.history.rows,
        checkpoint_ref=str(trainer.checkpoint_dir / "best.pt"),
    )
    trainer.save_history(run_dir / "logs" / "training-history.json")
    print(f"Run bundle written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
