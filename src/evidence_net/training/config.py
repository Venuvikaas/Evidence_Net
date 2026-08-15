"""Structured training configuration.

Configurations are plain YAML dicts validated against a fixed schema so a
broken config fails loudly before training starts (Phase 3, training
infrastructure). Unknown keys, wrong types, and out-of-range values are
rejected; every accepted config is fully serializable for provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Raised when a training configuration is invalid."""


@dataclass(frozen=True)
class DataConfig:
    """Dataset selection for a training run."""

    split: str = "train"
    n_samples: int = 64
    seed: int = 0

    def validate(self) -> None:
        if self.split not in ("train", "validation", "calibration"):
            raise ConfigError(f"data.split must be train/validation/calibration, got {self.split}")
        if self.n_samples < 1:
            raise ConfigError(f"data.n_samples must be >= 1, got {self.n_samples}")


@dataclass(frozen=True)
class ModelConfig:
    """Model selection and architecture parameters."""

    name: str = "base"
    hidden_channels: int = 32
    depth: int = 3
    amplitude: float = 0.1

    def validate(self) -> None:
        if self.name not in ("base", "direct", "proposal"):
            raise ConfigError(f"model.name must be base, direct, or proposal, got {self.name}")
        if self.hidden_channels < 4:
            raise ConfigError(f"model.hidden_channels must be >= 4, got {self.hidden_channels}")
        if self.depth < 1:
            raise ConfigError(f"model.depth must be >= 1, got {self.depth}")
        if self.amplitude <= 0.0:
            raise ConfigError(f"model.amplitude must be > 0, got {self.amplitude}")


@dataclass(frozen=True)
class LossConfig:
    """Weights for the composite base loss (pixel, structural, edge, frequency)."""

    pixel: float = 1.0
    structural: float = 0.25
    edge: float = 0.25
    frequency: float = 0.1
    residual: float = 0.0

    def validate(self) -> None:
        for name in ("pixel", "structural", "edge", "frequency", "residual"):
            value = getattr(self, name)
            if value < 0.0:
                raise ConfigError(f"loss.{name} must be >= 0, got {value}")
        if all(
            getattr(self, name) == 0.0
            for name in ("pixel", "structural", "edge", "frequency", "residual")
        ):
            raise ConfigError("loss weights cannot all be zero")


@dataclass(frozen=True)
class TrainConfig:
    """Complete validated configuration for one training run."""

    seed: int = 0
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    mixed_precision: bool = False
    checkpoint_dir: str = "checkpoints"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def validate(self) -> None:
        if self.epochs < 1:
            raise ConfigError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0.0:
            raise ConfigError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0.0:
            raise ConfigError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.grad_clip is not None and self.grad_clip <= 0.0:
            raise ConfigError(f"grad_clip must be > 0, got {self.grad_clip}")
        if not self.checkpoint_dir:
            raise ConfigError("checkpoint_dir cannot be empty")
        self.data.validate()
        self.model.validate()
        self.loss.validate()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.as_dict(), sort_keys=True)


@dataclass(frozen=True)
class _RawConfig:
    """Internal: raw YAML dict before schema binding."""

    data: dict[str, object] = field(default_factory=dict)
    model: dict[str, object] = field(default_factory=dict)
    loss: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: object) -> _RawConfig:
        if not isinstance(raw, dict):
            raise ConfigError("config root must be a mapping")
        allowed = {
            "seed",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "grad_clip",
            "mixed_precision",
            "checkpoint_dir",
            "data",
            "model",
            "loss",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ConfigError(f"unknown config keys: {sorted(unknown)}")
        return cls(
            data=cls._section(raw, "data", {"split", "n_samples", "seed"}),
            model=cls._section(raw, "model", {"name", "hidden_channels", "depth", "amplitude"}),
            loss=cls._section(
                raw, "loss", {"pixel", "structural", "edge", "frequency", "residual"}
            ),
        )

    @staticmethod
    def _section(raw: dict[str, object], name: str, keys: set[str]) -> dict[str, object]:
        value = raw.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"{name} must be a mapping")
        unknown = set(value) - keys
        if unknown:
            raise ConfigError(f"unknown {name} config keys: {sorted(unknown)}")
        return value


def _as_bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be a bool, got {type(value).__name__}")
    return value


def _as_int(value: object, path: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{path} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ConfigError(f"{path} must be >= {minimum}, got {value}")
    return value


def _as_float(value: object, path: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"{path} must be a number, got {type(value).__name__}")
    return float(value)


def _as_str(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{path} must be a non-empty string")
    return value


def load_config(path: Path) -> TrainConfig:
    """Load and validate a training config from YAML."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    parsed = _RawConfig.from_dict(raw)

    def pick(raw_dict: dict[str, object], key: str, default: object) -> object:
        return raw_dict.get(key, default)

    seed = _as_int(pick(parsed.data, "seed", 0), "data.seed", 0)
    split = _as_str(pick(parsed.data, "split", "train"), "data.split")
    n_samples = _as_int(pick(parsed.data, "n_samples", 64), "data.n_samples", 1)

    model_name = _as_str(pick(parsed.model, "name", "base"), "model.name")
    hidden_channels = _as_int(pick(parsed.model, "hidden_channels", 32), "model.hidden_channels", 4)
    depth = _as_int(pick(parsed.model, "depth", 3), "model.depth", 1)
    amplitude = _as_float(pick(parsed.model, "amplitude", 0.1), "model.amplitude")

    pixel = _as_float(pick(parsed.loss, "pixel", 1.0), "loss.pixel")
    structural = _as_float(pick(parsed.loss, "structural", 0.25), "loss.structural")
    edge = _as_float(pick(parsed.loss, "edge", 0.25), "loss.edge")
    frequency = _as_float(pick(parsed.loss, "frequency", 0.1), "loss.frequency")
    residual = _as_float(pick(parsed.loss, "residual", 0.0), "loss.residual")

    config = TrainConfig(
        seed=_as_int(pick(raw, "seed", 0), "seed", 0),
        epochs=_as_int(pick(raw, "epochs", 10), "epochs", 1),
        batch_size=_as_int(pick(raw, "batch_size", 8), "batch_size", 1),
        learning_rate=_as_float(pick(raw, "learning_rate", 1e-3), "learning_rate"),
        weight_decay=_as_float(pick(raw, "weight_decay", 0.0), "weight_decay"),
        grad_clip=(
            None
            if raw.get("grad_clip") is None
            else _as_float(pick(raw, "grad_clip", 1.0), "grad_clip")
        ),
        mixed_precision=_as_bool(pick(raw, "mixed_precision", False), "mixed_precision"),
        checkpoint_dir=_as_str(pick(raw, "checkpoint_dir", "checkpoints"), "checkpoint_dir"),
        data=DataConfig(split=split, n_samples=n_samples, seed=seed),
        model=ModelConfig(
            name=model_name, hidden_channels=hidden_channels, depth=depth, amplitude=amplitude
        ),
        loss=LossConfig(
            pixel=pixel,
            structural=structural,
            edge=edge,
            frequency=frequency,
            residual=residual,
        ),
    )
    config.validate()
    return config


def save_config(path: Path, config: TrainConfig) -> None:
    """Write a validated config to YAML (round-trips through load_config)."""
    path.write_text(config.to_yaml(), encoding="utf-8")
