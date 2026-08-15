"""Trainer, checkpointing, and experiment provenance (Phase 3)."""

from evidence_net.training.config import TrainConfig, load_config, save_config
from evidence_net.training.dataset import RestorationDataset, select_sample_ids
from evidence_net.training.provenance import (
    capture_environment,
    create_experiment_bundle,
    manifest_hashes,
)
from evidence_net.training.trainer import Trainer, TrainingFailure, set_seed

__all__ = [
    "RestorationDataset",
    "TrainConfig",
    "Trainer",
    "TrainingFailure",
    "capture_environment",
    "create_experiment_bundle",
    "load_config",
    "manifest_hashes",
    "save_config",
    "select_sample_ids",
    "set_seed",
]
