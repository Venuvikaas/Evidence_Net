"""Validate the shipped training configs and smoke-scale learning (box 11).

The Phase 3 experiment-series box requires broken configurations to be
rejected before training and smoke-scale learning to complete without
failure. This test loads every config under ``configs/model/``, validates it,
and runs a two-epoch smoke training on synthetic tensors for the smoke-scale
configs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from evidence_net.losses.base_losses import BaseLoss
from evidence_net.models.factory import build_model
from evidence_net.training.config import ConfigError, TrainConfig, load_config
from evidence_net.training.trainer import Trainer

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs" / "model"
SMOKE_CONFIGS = ("base-smoke.yaml",)


def _configs() -> list[Path]:
    return sorted(CONFIGS_DIR.glob("*.yaml"))


def test_all_shipped_configs_validate() -> None:
    paths = _configs()
    assert paths, "no configs under configs/model/"
    for path in paths:
        config = load_config(path)
        config.validate()
        assert isinstance(config, TrainConfig)


def test_smoke_configs_training_history_and_checkpoints(tmp_path: Path) -> None:
    for name in SMOKE_CONFIGS:
        config = load_config(CONFIGS_DIR / name)
        rng = torch.Generator().manual_seed(config.seed)
        inputs = torch.rand(4, 1, 16, 16, generator=rng)
        targets = torch.rand(4, 1, 32, 32, generator=rng)
        loader = DataLoader(
            TensorDataset(inputs, targets, torch.arange(4).float()),
            batch_size=config.batch_size,
        )
        model = build_model(config.model)
        trainer = Trainer(
            model,
            config,
            loader,
            loss_fn=BaseLoss(config.loss),
            checkpoint_dir=tmp_path / name,
        )
        history = trainer.fit()
        assert len(history.rows) == config.epochs
        assert (tmp_path / name / "best.pt").is_file()
        assert (tmp_path / name / "last.pt").is_file()


@pytest.mark.parametrize(
    "broken",
    [
        "model:\n  name: bogus\n",
        "epochs: 0\n",
        "data:\n  split: test-final\n",
        "loss:\n  pixel: -1\n",
        "batch_size: 0\n",
        "unknown_key: 1\n",
    ],
)
def test_broken_configs_rejected(tmp_path: Path, broken: str) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)
