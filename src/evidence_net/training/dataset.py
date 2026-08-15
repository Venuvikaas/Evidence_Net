"""PyTorch dataset over the frozen official train manifest.

Pairs are selected deterministically from a development split by seeded
sampling of sample ids; ``Test_NoisyLR/`` can never enter (Phase 1 isolation
still holds — the dataset only reads the train manifest). Inputs are the
128x128 ``NoisyLR`` tensors and targets the 256x256 ``GT`` tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from evidence_net.data.loaders import load_npy

MANIFESTS_DIR = Path(__file__).resolve().parents[3] / "data" / "manifests"
TRAIN_MANIFEST = MANIFESTS_DIR / "official-train-source-v1.json"
SPLITS_MANIFEST = MANIFESTS_DIR / "dataset-splits-v1.json"


class ManifestDatasetError(RuntimeError):
    """Raised when pairs cannot be built from the frozen manifests."""


def _load_splits() -> dict[str, str]:
    data = json.loads(SPLITS_MANIFEST.read_text(encoding="utf-8"))
    return data["assignments"]


def _load_train_entries() -> dict[str, dict[str, Path]]:
    """Map sample id -> {input, target} relative paths from the train manifest."""
    data = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
    pairs: dict[str, dict[str, Path]] = {}
    for entry in data["files"]:
        rel = Path(entry["relative_path"])
        sample_id = rel.stem
        if "NoisyLR" in rel.parts:
            pairs.setdefault(sample_id, {})["input"] = rel
        elif "GT" in rel.parts:
            pairs.setdefault(sample_id, {})["target"] = rel
    return pairs


def select_sample_ids(split: str, n_samples: int, seed: int) -> list[str]:
    """Seeded, deterministic sample-id selection from one development split."""
    assignments = _load_splits()
    ids = sorted(sample_id for sample_id, label in assignments.items() if label == split)
    if not ids:
        raise ManifestDatasetError(f"split '{split}' has no samples in dataset-splits-v1")
    rng = np.random.default_rng(seed)
    take = min(n_samples, len(ids))
    indices = rng.choice(len(ids), size=take, replace=False)
    return [ids[int(index)] for index in np.sort(indices)]


class RestorationDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """(input, target, sample_id) triples from the frozen train manifest."""

    def __init__(
        self,
        train_dir: Path,
        split: str = "train",
        n_samples: int = 64,
        seed: int = 0,
        *,
        sample_ids: list[str] | None = None,
    ) -> None:
        if sample_ids is None:
            sample_ids = select_sample_ids(split, n_samples, seed)
        pairs = _load_train_entries()
        missing = [sample_id for sample_id in sample_ids if sample_id not in pairs]
        if missing:
            raise ManifestDatasetError(f"sample ids missing from train manifest: {missing}")
        self.train_dir = train_dir
        self.sample_ids = sample_ids
        self.pairs = {sample_id: pairs[sample_id] for sample_id in sample_ids}

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample_id = self.sample_ids[index]
        rel = self.pairs[sample_id]
        if "input" not in rel or "target" not in rel:
            raise ManifestDatasetError(f"incomplete pair for {sample_id}")
        input_array = load_npy(self.train_dir / rel["input"])
        target_array = load_npy(self.train_dir / rel["target"])
        input_tensor = torch.from_numpy(input_array).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_array).float().unsqueeze(0)
        return input_tensor, target_tensor, sample_id
