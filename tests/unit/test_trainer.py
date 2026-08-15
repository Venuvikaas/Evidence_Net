"""Trainer sanity and failure-guard tests (Phase 3 boxes 4-5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from evidence_net.training.config import TrainConfig
from evidence_net.training.trainer import Trainer, TrainingFailure, set_seed


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.up(self.conv(x)), 0.0, 1.0)


def _synthetic_dataset(n: int = 8, size: int = 16) -> Dataset:
    rng = np.random.default_rng(0)
    inputs = torch.from_numpy(rng.random((n, 1, size, size))).float()
    targets = torch.from_numpy(rng.random((n, 1, size * 2, size * 2))).float()
    return TensorDataset(inputs, targets, torch.arange(n).float())


def _config(**overrides: object) -> TrainConfig:
    base = {
        "seed": 0,
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 1e-2,
        "checkpoint_dir": "checkpoints",
    }
    base.update(overrides)
    return TrainConfig(**base)


def test_set_seed_is_reproducible() -> None:
    set_seed(42)
    first = torch.rand(3)
    set_seed(42)
    assert torch.equal(first, torch.rand(3))


def test_single_step_changes_parameters(tmp_path: Path) -> None:
    model = _TinyModel()
    loader = DataLoader(_synthetic_dataset(4), batch_size=4)
    trainer = Trainer(model, _config(epochs=1), loader, checkpoint_dir=tmp_path)
    before = {name: param.clone() for name, param in model.named_parameters()}
    trainer.fit()
    changed = any(not torch.equal(param, before[name]) for name, param in model.named_parameters())
    assert changed


def test_tiny_batch_overfit_loss_decreases(tmp_path: Path) -> None:
    model = _TinyModel()
    loader = DataLoader(_synthetic_dataset(8, size=8), batch_size=4)
    trainer = Trainer(
        model, _config(epochs=25, learning_rate=1e-2), loader, checkpoint_dir=tmp_path
    )
    history = trainer.fit()
    first = history.rows[0]["train_loss"]
    last = history.rows[-1]["train_loss"]
    assert last < first


def test_checkpoint_and_resume(tmp_path: Path) -> None:
    model = _TinyModel()
    loader = DataLoader(_synthetic_dataset(8), batch_size=4)
    trainer = Trainer(model, _config(epochs=2), loader, checkpoint_dir=tmp_path)
    trainer.fit()
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "last.pt").is_file()

    resumed = Trainer(
        _TinyModel(),
        _config(epochs=4),
        loader,
        checkpoint_dir=tmp_path,
        resume_from=tmp_path / "last.pt",
    )
    assert resumed.start_epoch == 2
    assert len(resumed.history.rows) == 2


def test_empty_train_loader_raises(tmp_path: Path) -> None:
    model = _TinyModel()
    empty = DataLoader(_synthetic_dataset(0))
    trainer = Trainer(model, _config(), empty, checkpoint_dir=tmp_path)
    with pytest.raises(TrainingFailure, match="empty"):
        trainer.fit()


def test_nan_loss_raises(tmp_path: Path) -> None:
    class _ExplodingLoss(nn.Module):
        def forward(self, _pred: torch.Tensor, _target: torch.Tensor) -> torch.Tensor:
            return torch.tensor(float("nan"), requires_grad=True)

    model = _TinyModel()
    loader = DataLoader(_synthetic_dataset(4), batch_size=4)
    trainer = Trainer(model, _config(), loader, checkpoint_dir=tmp_path, loss_fn=_ExplodingLoss())
    with pytest.raises(TrainingFailure, match="non-finite loss"):
        trainer.fit()


def test_infinite_gradient_raises(tmp_path: Path) -> None:
    class _BombParameter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1000.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # exp(1000 * x) overflows to inf for x > 0 -> inf loss -> inf grads.
            return torch.exp(self.weight * x)

    model = _BombParameter()
    loader = DataLoader(
        TensorDataset(
            torch.ones(2, 1, 4, 4),
            torch.ones(2, 1, 4, 4),
            torch.arange(2).float(),
        ),
        batch_size=2,
    )
    trainer = Trainer(model, _config(), loader, checkpoint_dir=tmp_path)
    with pytest.raises(TrainingFailure, match="non-finite"):
        trainer.fit()


def test_history_export(tmp_path: Path) -> None:
    model = _TinyModel()
    loader = DataLoader(_synthetic_dataset(4), batch_size=4)
    trainer = Trainer(model, _config(epochs=2), loader, checkpoint_dir=tmp_path)
    trainer.fit()
    history_path = tmp_path / "history.json"
    trainer.save_history(history_path)
    assert history_path.is_file()
    assert "train_loss" in history_path.read_text()
