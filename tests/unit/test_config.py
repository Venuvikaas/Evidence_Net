"""Tests for structured training configuration validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from evidence_net.training.config import ConfigError, TrainConfig, load_config, save_config


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_defaults(tmp_path: Path) -> None:
    config = load_config(_write(tmp_path, "{}\n"))
    assert config.seed == 0
    assert config.epochs == 10
    assert config.batch_size == 8
    assert config.model.name == "base"
    assert config.loss.pixel == 1.0


def test_load_full_config(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "seed: 3\nepochs: 5\nbatch_size: 4\nlearning_rate: 0.001\n"
        "data:\n  split: validation\n  n_samples: 16\n  seed: 1\n"
        "model:\n  name: direct\n  hidden_channels: 16\n  depth: 2\n"
        "loss:\n  pixel: 1.0\n  structural: 0.5\n  edge: 0.1\n  frequency: 0.0\n",
    )
    config = load_config(path)
    assert config.seed == 3
    assert config.epochs == 5
    assert config.data.split == "validation"
    assert config.model.name == "direct"
    assert config.loss.structural == 0.5


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown config keys"):
        load_config(_write(tmp_path, "epochs: 2\nnonsense: 1\n"))


def test_unknown_section_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown data config keys"):
        load_config(_write(tmp_path, "data:\n  split: train\n  magic: 1\n"))


def test_invalid_model_name_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="model.name"):
        load_config(_write(tmp_path, "model:\n  name: nonsense\n"))


def test_negative_epochs_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="epochs"):
        load_config(_write(tmp_path, "epochs: 0\n"))


def test_all_zero_loss_weights_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot all be zero"):
        load_config(
            _write(
                tmp_path,
                "loss:\n  pixel: 0\n  structural: 0\n  edge: 0\n  frequency: 0\n",
            )
        )


def test_wrong_type_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be an int"):
        load_config(_write(tmp_path, "epochs: many\n"))


def test_bad_yaml_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(_write(tmp_path, "epochs: [unclosed\n"))


def test_save_roundtrip(tmp_path: Path) -> None:
    original = TrainConfig(seed=7, epochs=3, batch_size=2)
    path = tmp_path / "roundtrip.yaml"
    save_config(path, original)
    restored = load_config(path)
    assert restored == original
