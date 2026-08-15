"""Reproducible training loop.

Implements the Phase 3 trainer contract: controlled seeds, checkpointing
with resume, an optional mixed-precision path, and a device-agnostic loop
(CPU by default; CUDA/MPS when available). Failure guards are added in a
later checklist box.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from evidence_net.training.config import TrainConfig


class TrainingFailure(RuntimeError):
    """Raised when a training run must abort (numerical or batch failure)."""


class _History:
    """Append-only epoch history with JSON export."""

    def __init__(self) -> None:
        self.rows: list[dict[str, float | int]] = []

    def add(self, row: dict[str, float | int]) -> None:
        self.rows.append(dict(row))

    def best(self, metric: str, maximize: bool = True) -> tuple[float, int]:
        if not self.rows:
            raise TrainingFailure("cannot select best from an empty history")
        values = [float(row.get(metric, float("nan"))) for row in self.rows]
        valid = [(value, index) for index, value in enumerate(values) if value == value]
        if not valid:
            raise TrainingFailure(f"no finite {metric} values in history")
        key = max if maximize else min
        return key(valid, key=lambda pair: pair[0])

    def to_json(self) -> str:
        return json.dumps(self.rows, indent=2, sort_keys=True) + "\n"


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """Fit a model with checkpointable training."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        train_loader: DataLoader,
        *,
        val_loader: DataLoader | None = None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        device: torch.device | None = None,
        checkpoint_dir: Path | None = None,
        resume_from: Path | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn or (lambda pred, target: nn.functional.l1_loss(pred, target))
        self.device = device or _pick_device()
        self.checkpoint_dir = checkpoint_dir or Path(config.checkpoint_dir)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.history = _History()
        self.start_epoch = 0
        if resume_from is not None:
            self._load_checkpoint(resume_from)

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        steps = 0
        for _, (inputs, targets, _ids) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._forward_loss(inputs, targets)
            loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
        return {"epoch": epoch, "train_loss": total_loss / steps, "train_steps": steps}

    def _forward_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.config.mixed_precision and self.device.type in ("cuda", "cpu"):
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                predictions = self.model(inputs)
                return self.loss_fn(predictions, targets)
        predictions = self.model(inputs)
        return self.loss_fn(predictions, targets)

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        if self.val_loader is None:
            return {"epoch": epoch, "val_loss": float("nan"), "val_steps": 0}
        self.model.eval()
        total_loss = 0.0
        steps = 0
        for inputs, targets, _ids in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            loss = self.loss_fn(self.model(inputs), targets)
            total_loss += float(loss.detach())
            steps += 1
        return {"epoch": epoch, "val_loss": total_loss / steps, "val_steps": steps}

    def fit(self, *, log_every: int | None = None) -> _History:
        """Run the configured number of epochs, checkpointing best and last."""
        for epoch in range(self.start_epoch, self.config.epochs):
            train_row = self._train_epoch(epoch)
            val_row = self._validate(epoch)
            row = {**train_row, **{k: v for k, v in val_row.items() if k != "epoch"}}
            self.history.add(row)
            metric = "val_loss" if self.val_loader is not None else "train_loss"
            self._checkpoint(epoch, is_best=row[metric] <= self._best_metric(metric))
            if log_every is not None and (
                epoch % log_every == 0 or epoch == self.config.epochs - 1
            ):
                print(
                    f"epoch {epoch + 1}/{self.config.epochs} "
                    f"train {row['train_loss']:.6f} "
                    f"val {row['val_loss']:.6f}"
                )
        return self.history

    def _best_metric(self, metric: str) -> float:
        try:
            value, _ = self.history.best(metric, maximize=False)
            return value
        except TrainingFailure:
            return float("inf")

    def _checkpoint(self, epoch: int, *, is_best: bool) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.as_dict(),
            "history": self.history.rows,
        }
        torch.save(payload, self.checkpoint_dir / "last.pt")
        if is_best:
            torch.save(payload, self.checkpoint_dir / "best.pt")

    def _load_checkpoint(self, path: Path) -> None:
        if not path.is_file():
            raise TrainingFailure(f"resume checkpoint not found: {path}")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.start_epoch = int(payload["epoch"]) + 1
        self.history.rows = [dict(row) for row in payload.get("history", [])]

    def save_history(self, path: Path) -> None:
        path.write_text(self.history.to_json(), encoding="utf-8")

    @torch.no_grad()
    def predict(self, inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Deterministic batched inference in eval mode (inputs already on device)."""
        self.model.eval()
        outputs: list[torch.Tensor] = []
        for array in inputs:
            outputs.append(self.model(array.to(self.device)).cpu())
        return outputs
