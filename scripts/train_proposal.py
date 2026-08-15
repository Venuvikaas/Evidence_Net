"""Train the bounded Detail Proposal against a frozen Base Reconstruction.

Usage (from the repository root):::

    python scripts/train_proposal.py --config configs/model/proposal-smoke.yaml
    python scripts/train_proposal.py --config configs/model/proposal-gate3.yaml

Loads the frozen Base checkpoint (default ``checkpoints/train-base-gate2/
best.pt``), builds the ``BoundedDetailProposal`` wrapper (frozen Base +
amplitude-bounded proposer head), and trains the candidate ``c = b + d``
against the clean target with the composite base loss. The Base parameters
never receive gradients; only the proposer head is updated. Provenance is
persisted as a run bundle with a checkpoint reference. Never touches
``Test_NoisyLR/``.

``--synthetic`` runs the same script path on synthetic tensors (and a
freshly generated random frozen Base when ``--base-checkpoint`` is not
available) instead of the frozen dataset, so CI can exercise the proposal
training path without the data files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from evidence_net.data.paths import resolve_dataset_paths
    from evidence_net.losses.base_losses import ProposalLoss
    from evidence_net.models.factory import model_summary
    from evidence_net.models.proposal import BoundedDetailProposal, DetailProposer
    from evidence_net.reporting.run_bundle import new_run_id
    from evidence_net.training.config import TrainConfig, load_config
    from evidence_net.training.dataset import RestorationDataset
    from evidence_net.training.provenance import create_experiment_bundle
    from evidence_net.training.trainer import Trainer, set_seed
except ImportError:  # allow running before `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evidence_net.data.paths import resolve_dataset_paths  # noqa: E402
    from evidence_net.losses.base_losses import ProposalLoss  # noqa: E402
    from evidence_net.models.factory import model_summary  # noqa: E402
    from evidence_net.models.proposal import BoundedDetailProposal, DetailProposer  # noqa: E402
    from evidence_net.reporting.run_bundle import new_run_id  # noqa: E402
    from evidence_net.training.config import TrainConfig, load_config  # noqa: E402
    from evidence_net.training.dataset import RestorationDataset  # noqa: E402
    from evidence_net.training.provenance import create_experiment_bundle  # noqa: E402
    from evidence_net.training.trainer import Trainer, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=REPO_ROOT / "configs" / "model" / "proposal-smoke.yaml",
        type=Path,
        help="training config YAML (model.name must be 'proposal')",
    )
    parser.add_argument(
        "--base-checkpoint",
        default=REPO_ROOT / "checkpoints" / "train-base-gate2" / "best.pt",
        type=Path,
        help="frozen Base checkpoint to load",
    )
    parser.add_argument("--run-id", default=None, help="explicit run id")
    parser.add_argument("--out", default=REPO_ROOT / "runs", type=Path, help="runs directory")
    parser.add_argument(
        "--resume", default=None, type=Path, help="checkpoint to resume from (last.pt)"
    )
    parser.add_argument("--device", default=None, help="torch device override (default: auto)")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="train on synthetic tensors instead of the frozen dataset (CI smoke)",
    )
    return parser.parse_args()


def write_synthetic_base_checkpoint(config: TrainConfig, path: Path) -> None:
    """Write a random frozen Base checkpoint for ``--synthetic`` runs.

    Mirrors the trainer payload layout (``epoch``, ``model_state_dict``,
    ``config`` with a ``model`` section) so ``load_frozen_base`` rebuilds the
    architecture from the same fields a real training run would record.
    """
    from evidence_net.models.base import BaseReconstruction

    base = BaseReconstruction(
        hidden_channels=config.model.hidden_channels,
        depth=config.model.depth,
    )
    payload = {
        "epoch": 0,
        "model_state_dict": base.state_dict(),
        "optimizer_state_dict": {},
        "config": {
            "model": {
                "name": "base",
                "hidden_channels": config.model.hidden_channels,
                "depth": config.model.depth,
                "amplitude": config.model.amplitude,
            }
        },
        "history": [],
    }
    torch.save(payload, path)


def load_frozen_base(checkpoint: Path) -> torch.nn.Module:
    """Rebuild the frozen Base model from its checkpoint config and weights."""
    from evidence_net.models.factory import build_model
    from evidence_net.training.config import ModelConfig

    if not checkpoint.is_file():
        raise SystemExit(f"FAIL: frozen base checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base_config = ModelConfig(
        name=payload["config"]["model"]["name"],
        hidden_channels=payload["config"]["model"]["hidden_channels"],
        depth=payload["config"]["model"]["depth"],
        amplitude=payload["config"]["model"].get("amplitude", 0.1),
    )
    base = build_model(base_config)
    base.load_state_dict(payload["model_state_dict"])
    base.eval()
    return base


def main() -> int:
    args = parse_args()
    if not args.config.is_file():
        print(f"FAIL config not found: {args.config}", file=sys.stderr)
        return 1
    config = load_config(args.config)
    if config.model.name != "proposal":
        print("FAIL: config.model.name must be 'proposal'", file=sys.stderr)
        return 1
    set_seed(config.seed)
    run_id = args.run_id or new_run_id("train-proposal")
    checkpoint_dir = REPO_ROOT / "checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader: DataLoader[Any]
    val_loader: DataLoader[Any]
    if args.synthetic:
        # Same script path, synthetic tensors (128x128 input -> 256x256
        # target, matching the RestorationDataset contract).
        if args.base_checkpoint.is_file():
            base = load_frozen_base(args.base_checkpoint)
        else:
            synthetic_base = checkpoint_dir / "synthetic-base.pt"
            write_synthetic_base_checkpoint(config, synthetic_base)
            base = load_frozen_base(synthetic_base)
        n = config.data.n_samples
        rng = torch.Generator().manual_seed(config.data.seed)
        inputs = torch.rand(n, 1, 128, 128, generator=rng)
        targets = torch.rand(n, 1, 256, 256, generator=rng)
        ids = torch.arange(n).float()
        synthetic = TensorDataset(inputs, targets, ids)
        train_loader = DataLoader(synthetic, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(synthetic, batch_size=config.batch_size, shuffle=False)
    else:
        paths = resolve_dataset_paths()
        base = load_frozen_base(args.base_checkpoint)
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

    proposer = DetailProposer(
        hidden_channels=config.model.hidden_channels,
        depth=config.model.depth,
        amplitude=config.model.amplitude,
    )
    model = BoundedDetailProposal(base, proposer)

    loss_fn = ProposalLoss(config.loss, model)
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
