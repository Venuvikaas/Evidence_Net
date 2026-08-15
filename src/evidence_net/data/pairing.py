"""Official ``train/`` pairing adapter.

Discovers the noisy-input / target relationship from the observed directory
structure (``train/GT`` and ``train/NoisyLR`` under the train source root),
pairs files by zero-padded base name, and reports unmatched, duplicated, or
ambiguous files instead of silently skipping them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evidence_net.data.manifests import FileEntry

JUNK_DIR_NAMES = ("__MACOSX",)
HIDDEN_PREFIX = "."
TARGET_DIR_NAMES = ("GT",)
INPUT_DIR_NAMES = ("NoisyLR",)


class PairingError(RuntimeError):
    """Raised when the official train structure cannot be resolved safely."""


def _is_junk_dir(name: str) -> bool:
    return name in JUNK_DIR_NAMES or name.startswith(HIDDEN_PREFIX)


def find_structure_dirs(root: Path, names: tuple[str, ...]) -> list[Path]:
    """Find directories with the given names under ``root``, excluding junk."""
    found: list[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _is_junk_dir(d)]
        for name in names:
            candidate = Path(dirpath) / name
            if candidate.is_dir():
                found.append(candidate)
    return found


def discover_train_structure(train_root: Path) -> tuple[Path, Path]:
    """Return ``(gt_dir, noisy_lr_dir)`` relative to ``train_root``.

    Requires exactly one ``GT`` and one ``NoisyLR`` directory under the train
    source root (excluding junk directories).
    """
    gt_dirs = find_structure_dirs(train_root, TARGET_DIR_NAMES)
    noisy_dirs = find_structure_dirs(train_root, INPUT_DIR_NAMES)
    if len(gt_dirs) != 1:
        raise PairingError(
            f"expected exactly one GT directory under {train_root}, found {len(gt_dirs)}"
        )
    if len(noisy_dirs) != 1:
        raise PairingError(
            f"expected exactly one NoisyLR directory under {train_root}, found {len(noisy_dirs)}"
        )
    return gt_dirs[0], noisy_dirs[0]


@dataclass
class PairReport:
    """Pairing audit results for the official train directory."""

    pairs: list[tuple[Path, Path]] = field(default_factory=list)
    unmatched_gt: list[str] = field(default_factory=list)
    unmatched_noisy: list[str] = field(default_factory=list)
    duplicate_gt: list[str] = field(default_factory=list)
    duplicate_noisy: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (
            self.unmatched_gt or self.unmatched_noisy or self.duplicate_gt or self.duplicate_noisy
        )

    def summary(self) -> dict[str, int]:
        return {
            "pairs": len(self.pairs),
            "unmatched_gt": len(self.unmatched_gt),
            "unmatched_noisy": len(self.unmatched_noisy),
            "duplicate_gt": len(self.duplicate_gt),
            "duplicate_noisy": len(self.duplicate_noisy),
            "is_clean": self.is_clean,
        }


def pair_integrity_report(entries: list[FileEntry]) -> dict[str, Any]:
    """Verify every NoisyLR input has exactly one GT partner in the manifest.

    Works on manifest ``FileEntry`` records (paths relative to the train
    source root) and reports missing, duplicate, or ambiguous partners.
    """
    noisy: dict[str, list[FileEntry]] = {}
    targets: dict[str, list[FileEntry]] = {}
    for entry in entries:
        if "NoisyLR" in entry.relative_path:
            noisy.setdefault(Path(entry.relative_path).stem, []).append(entry)
        elif "/GT/" in entry.relative_path:
            targets.setdefault(Path(entry.relative_path).stem, []).append(entry)
        else:
            continue
    missing_partner = sorted(set(noisy) - set(targets)) + sorted(set(targets) - set(noisy))
    duplicated = sorted({k for k, v in list(noisy.items()) + list(targets.items()) if len(v) != 1})
    return {
        "n_inputs": len(noisy),
        "n_targets": len(targets),
        "n_pairs": sum(1 for k in noisy if k in targets),
        "missing_partners": missing_partner,
        "duplicated_ids": duplicated,
        "is_clean": not missing_partner and not duplicated,
    }


def _base_name(path: Path) -> str:
    return path.stem


def audit_pairing(gt_dir: Path, noisy_dir: Path) -> PairReport:
    """Pair GT and NoisyLR files by base name; report anomalies."""
    gt_files = sorted(Path(gt_dir).glob("*.npy"))
    noisy_files = sorted(Path(noisy_dir).glob("*.npy"))
    gt_bases = [_base_name(p) for p in gt_files]
    noisy_bases = [_base_name(p) for p in noisy_files]

    gt_index: dict[str, list[Path]] = {}
    for path, base in zip(gt_files, gt_bases, strict=True):
        gt_index.setdefault(base, []).append(path)
    noisy_index: dict[str, list[Path]] = {}
    for path, base in zip(noisy_files, noisy_bases, strict=True):
        noisy_index.setdefault(base, []).append(path)

    report = PairReport()
    for base, paths in gt_index.items():
        if len(paths) > 1:
            report.duplicate_gt.append(base)
        if base not in noisy_index:
            report.unmatched_gt.append(base)
        elif len(paths) == 1 and len(noisy_index[base]) == 1:
            report.pairs.append((noisy_index[base][0], paths[0]))
    for base, paths in noisy_index.items():
        if len(paths) > 1:
            report.duplicate_noisy.append(base)
        if base not in gt_index:
            report.unmatched_noisy.append(base)
    return report
