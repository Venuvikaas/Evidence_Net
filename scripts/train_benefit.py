"""Train the minimal Proposal-Benefit Predictor (Phase 5, two-stage).

Usage (from the repository root):::

    python scripts/train_benefit.py --synthetic
    python scripts/train_benefit.py --n-samples 128 --epochs 12

Loads the frozen Base and the trained proposal, generates the deterministic
benefit labels (support-definition-v1) on the **calibration split**, trains
the minimal MLP predictor on per-patch features, and optionally trains the
attention-gate CNN baseline. The predictor is trained **separately** from the
proposal and Base models (two-stage; their parameters never receive
gradients). Provenance is persisted as a run bundle with a checkpoint
reference. Never touches ``Test_NoisyLR/``.

``--synthetic`` runs the same path on synthetic tensors (freshly generated
Base and proposal behavior) so CI can exercise the predictor training path
without the data files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from evidence_net.benefit.labels import OUTPUT_GRID, patch_benefit_labels
    from evidence_net.benefit.predictors import (
        AttentionGateBaseline,
        MinimalBenefitPredictor,
        patch_features,
    )
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id
except ImportError:  # allow running before `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evidence_net.benefit.labels import (  # noqa: E402
        OUTPUT_GRID,
        patch_benefit_labels,
    )
    from evidence_net.benefit.predictors import (  # noqa: E402
        AttentionGateBaseline,
        MinimalBenefitPredictor,
        patch_features,
    )
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id  # noqa: E402

CALIBRATION_SPLIT = "calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint",
        default=REPO_ROOT / "checkpoints" / "train-base-gate2" / "best.pt",
        type=Path,
    )
    parser.add_argument(
        "--proposal-checkpoint",
        default=REPO_ROOT / "checkpoints" / "train-proposal-gate3v2" / "best.pt",
        type=Path,
    )
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", default=REPO_ROOT / "runs", type=Path)
    parser.add_argument(
        "--synthetic", action="store_true", help="train on synthetic tensors (CI smoke)"
    )
    return parser.parse_args()


def _synthetic_triple(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (input, base, proposal) grids with a structured benefit pattern.

    Flat low-signal regions get a large proposal (beneficial), noisy
    high-signal regions get a small proposal (rarely beneficial), so the
    labels are non-trivial and the features can learn structure.
    """
    rng = np.random.default_rng(seed)
    grid = OUTPUT_GRID
    y = np.full((grid, grid), 0.05)
    # Left half: flat, low signal.
    y[:, : grid // 2] = 0.05 + rng.normal(0.0, 0.005, size=(grid, grid // 2))
    # Right half: structured high signal (stripes).
    y[:, grid // 2 :] = 0.05
    for column in range(grid // 2, grid, 8):
        y[:, column] = 0.9
    base = np.clip(y, 0.0, 1.0)
    # Proposal is large and helpful on the flat half, small elsewhere.
    proposal = np.zeros((grid, grid))
    proposal[:, : grid // 2] = rng.normal(0.04, 0.01, size=(grid, grid // 2))
    proposal[:, grid // 2 :] = rng.normal(0.0, 0.002, size=(grid, grid // 2))
    return y, np.clip(base, 0.0, 1.0), proposal


def _build_dataset(seed: int, n_samples: int) -> tuple[list, list]:
    """Grid/label dataset on the calibration split (synthetic).

    Returns ``(grids, labels)`` where ``grids[i] = (y, b, d)`` triples and
    ``labels[i]`` is the deterministic benefit label map.
    """
    if n_samples < 1:
        raise SystemExit("FAIL: --n-samples must be >= 1")
    grids: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    labels: list[np.ndarray] = []
    for index in range(n_samples):
        y, b, d = _synthetic_triple(seed + index)
        # Clean target close to the candidate on the flat half: the proposal
        # is beneficial exactly where it is large (flat half) and harmful or
        # tied where it is tiny (striped half).
        target = np.clip(b + d, 0.0, 1.0)
        grids.append((y, b, d))
        labels.append(patch_benefit_labels(b, d, target).astype(np.float32))
    return grids, labels


def _real_dataset(
    base_checkpoint: Path, proposal_checkpoint: Path, n_samples: int, seed: int
) -> tuple[list, list]:
    """Grid/label dataset from the frozen Base/Proposal on calibration data."""
    from evidence_net.data.paths import resolve_dataset_paths
    from evidence_net.training.dataset import RestorationDataset
    from evidence_net.training.trainer import set_seed

    set_seed(seed)
    paths = resolve_dataset_paths()
    dataset = RestorationDataset(
        paths.train_dir, split=CALIBRATION_SPLIT, n_samples=n_samples, seed=seed
    )

    from evidence_net.models.factory import build_model
    from evidence_net.training.config import ModelConfig

    def load_model(checkpoint: Path) -> torch.nn.Module:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
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

    base = load_model(base_checkpoint)
    proposal = load_model(proposal_checkpoint)
    grids: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            inputs, target, _sample_id = dataset[index]
            y = inputs.squeeze(0).numpy()
            x = target.squeeze(0).numpy()
            b = base(inputs).squeeze(0).numpy()
            d = proposal(inputs).squeeze(0).numpy()
            grids.append((y, b, d))
            labels.append(patch_benefit_labels(b, d, x).astype(np.float32))
    return grids, labels


def _train(
    model: torch.nn.Module,
    inputs: list[np.ndarray],
    labels: list[np.ndarray],
    *,
    use_patch_features: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, float]:
    from evidence_net.training.trainer import set_seed

    set_seed(seed)
    label_tensor = torch.from_numpy(np.stack(labels))
    if use_patch_features:
        # MLP predictor: per-patch feature vectors (B, 16, 16, F).
        input_tensor = torch.from_numpy(np.stack(inputs))
    else:
        # Attention gate: 3-channel image grids (B, 3, H, W).
        stacked = np.stack(
            [np.stack([inp, base, prop], axis=0) for inp, base, prop in inputs]
        ).astype(np.float32)
        input_tensor = torch.from_numpy(stacked)
    loader = DataLoader(
        TensorDataset(input_tensor, label_tensor), batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    last_loss = float("nan")
    for _epoch in range(epochs):
        model.train()
        total = 0.0
        steps = 0
        for batch_inputs, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            # MLP logits: (B, 16, 16, 1); attention logits: (B, 1, 16, 16).
            if logits.shape[-1] == 1 and logits.dim() == 4:
                target = batch_labels.unsqueeze(-1)
            else:
                target = batch_labels.unsqueeze(1)
            loss = loss_fn(logits, target)
            if not torch.isfinite(loss):
                raise SystemExit("FAIL: non-finite predictor loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        if steps == 0:
            raise SystemExit("FAIL: empty predictor loader")
        last_loss = total / steps
    model.eval()
    return {"final_train_loss": last_loss, "epochs": epochs, "n_samples": len(inputs)}


def main() -> int:
    args = parse_args()
    if args.synthetic:
        grids, labels = _build_dataset(args.seed, args.n_samples)
    else:
        if not args.base_checkpoint.is_file() or not args.proposal_checkpoint.is_file():
            print(
                "FAIL: real mode needs base and proposal checkpoints "
                "(or use --synthetic for CI smoke)",
                file=sys.stderr,
            )
            return 1
        grids, labels = _real_dataset(
            args.base_checkpoint, args.proposal_checkpoint, args.n_samples, args.seed
        )

    predictor_features = [patch_features(y, b, d) for y, b, d in grids]
    predictor = MinimalBenefitPredictor(hidden_channels=args.hidden_channels, depth=args.depth)
    predictor_metrics = _train(
        predictor,
        predictor_features,
        labels,
        use_patch_features=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    attention = AttentionGateBaseline(hidden_channels=max(4, args.hidden_channels // 2))
    attention_metrics = _train(
        attention,
        grids,
        labels,
        use_patch_features=False,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed + 1,
    )

    run_id = args.run_id or new_run_id("benefit-predictor")
    checkpoint_dir = args.out / run_id / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    predictor_path = checkpoint_dir / "predictor-best.pt"
    attention_path = checkpoint_dir / "attention-gate-best.pt"
    torch.save(
        {
            "model_state_dict": predictor.state_dict(),
            "config": {
                "hidden_channels": args.hidden_channels,
                "depth": args.depth,
            },
            "metrics": predictor_metrics,
        },
        predictor_path,
    )
    torch.save(
        {
            "model_state_dict": attention.state_dict(),
            "config": {"hidden_channels": max(4, args.hidden_channels // 2)},
            "metrics": attention_metrics,
        },
        attention_path,
    )

    create_run_bundle(
        args.out,
        run_id,
        config={
            "phase": 5,
            "stage": "two-stage predictor training (calibration split only)",
            "mode": "synthetic" if args.synthetic else "real",
            "split": CALIBRATION_SPLIT,
            "n_samples": args.n_samples,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        manifest={
            "run_id": run_id,
            "contract": "support-definition-v1",
            "labels_version": "labels-v1",
            "split_isolation": "predictor trained on calibration split only; "
            "Test_NoisyLR never touched",
        },
        metrics={"predictor": predictor_metrics, "attention_gate": attention_metrics},
        summary=(
            f"# Benefit predictor training {run_id}\n\n"
            f"- Split: {CALIBRATION_SPLIT} only (two-stage; Base/proposal frozen).\n"
            f"- Predictor final train loss: {predictor_metrics['final_train_loss']:.6f}\n"
            f"- Attention gate final train loss: {attention_metrics['final_train_loss']:.6f}\n"
        ),
        reference=f"{predictor_path.as_posix()}\n{attention_path.as_posix()}",
    )
    print(f"Predictor checkpoint: {predictor_path}")
    print(f"Attention gate checkpoint: {attention_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
