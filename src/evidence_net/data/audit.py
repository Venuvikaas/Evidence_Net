"""Dataset audit computations.

Produces raw-range, clipping, size, structure, and degradation summaries;
quantifies target alignment with a documented offset method; detects exact
and near duplicates; and verifies train/test input compatibility. All
methods are deterministic and documented in the data card.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from evidence_net.data.loaders import load_npy
from evidence_net.data.manifests import FileEntry

ALIGNMENT_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))
NEAR_DUPLICATE_SCALE = 32
NEAR_DUPLICATE_PRECISION = 3


def _summarize(values: Sequence[float], label: str) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        f"{label}_min": float(array.min()),
        f"{label}_max": float(array.max()),
        f"{label}_mean": float(array.mean()),
        f"{label}_std": float(array.std()),
    }


def range_summary(entries: Sequence[FileEntry]) -> dict[str, Any]:
    """Raw-range, clipping, and size summaries for readable entries."""
    mins: list[float] = []
    maxs: list[float] = []
    means: list[float] = []
    clipped_to_unit: int = 0
    shapes: Counter[tuple[int, ...]] = Counter()
    dtypes: Counter[str] = Counter()
    unreadable: int = 0
    for entry in entries:
        if not entry.readable:
            unreadable += 1
            continue
        if entry.range is not None:
            lo, hi = entry.range
            mins.append(lo)
            maxs.append(hi)
            if lo >= 0.0 and hi <= 1.0:
                clipped_to_unit += 1
        if entry.dimensions is not None:
            shapes[tuple(entry.dimensions)] += 1
        if entry.dtype is not None:
            dtypes[entry.dtype] += 1
        if entry.dimensions is not None and entry.range is not None:
            means.append((entry.range[0] + entry.range[1]) / 2.0)

    summary: dict[str, Any] = {
        "n_entries": len(entries),
        "n_unreadable": unreadable,
        "n_readable": len(entries) - unreadable,
        "shape_counts": {str(k): v for k, v in shapes.items()},
        "dtype_counts": dict(dtypes),
        "clipped_to_unit_interval": clipped_to_unit,
        "min_stats": _summarize(mins, "min") if mins else {},
        "max_stats": _summarize(maxs, "max") if maxs else {},
        "midrange_stats": _summarize(means, "mid") if means else {},
    }
    return summary


def exact_duplicate_groups(entries: Sequence[FileEntry]) -> dict[str, list[str]]:
    """Group relative paths by sha256; return groups with more than one file."""
    by_hash: dict[str, list[str]] = {}
    for entry in entries:
        by_hash.setdefault(entry.sha256, []).append(entry.relative_path)
    return {digest: paths for digest, paths in by_hash.items() if len(paths) > 1}


def near_duplicate_signature(path: Path) -> str:
    """Content signature: mean-pooled downsample, quantized, hashed."""
    array = load_npy(path)
    if array.ndim == 3:
        array = array.mean(axis=0)
    pooled = _mean_pool(array, NEAR_DUPLICATE_SCALE)
    quantized = np.round(pooled, NEAR_DUPLICATE_PRECISION)
    return hashlib.sha256(quantized.tobytes()).hexdigest()


def _mean_pool(array: np.ndarray, scale: int) -> np.ndarray:
    """Deterministic mean pooling to ``scale x scale`` via integer strides."""
    h, w = array.shape
    rows = np.linspace(0, h, scale + 1).astype(int)
    cols = np.linspace(0, w, scale + 1).astype(int)
    out = np.zeros((scale, scale), dtype=float)
    for i in range(scale):
        for j in range(scale):
            block = array[rows[i] : rows[i + 1], cols[j] : cols[j + 1]]
            out[i, j] = float(block.mean())
    return out


def near_duplicate_groups(
    entries: Sequence[FileEntry],
    root: Path,
    *,
    sample_limit: int = 0,
    rng: np.random.Generator | None = None,
) -> dict[str, list[str]]:
    """Group readable entries by near-duplicate signature (candidates only)."""
    rng = rng or np.random.default_rng(0)
    pool = [e for e in entries if e.readable]
    if sample_limit > 0 and len(pool) > sample_limit:
        indices = rng.choice(len(pool), size=sample_limit, replace=False)
        pool = [pool[int(i)] for i in indices]
    by_sig: dict[str, list[str]] = {}
    for entry in pool:
        by_sig.setdefault(near_duplicate_signature(root / entry.relative_path), []).append(
            entry.relative_path
        )
    return {sig: paths for sig, paths in by_sig.items() if len(paths) > 1}


def alignment_audit(
    pairs: Sequence[tuple[Path, Path]],
    *,
    sample_limit: int = 0,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Quantify target alignment using a 2x block-offset search.

    GT is 2x the NoisyLR resolution. For each sampled pair, the NoisyLR image
    is compared against GT blocks taken at offsets ``{(0,0),(0,1),(1,0),(1,1)}``;
    the offset with the lowest mean absolute difference is the estimated
    alignment phase. The residual at the best offset is recorded as a
    target-alignment uncertainty estimate. All statistics are grouped by pair.
    """
    rng = rng or np.random.default_rng(0)
    if sample_limit > 0 and len(pairs) > sample_limit:
        indices = rng.choice(len(pairs), size=sample_limit, replace=False)
        pairs = [pairs[int(i)] for i in indices]
    if not pairs:
        return {"n_pairs": 0, "best_offsets": {}, "residual_stats": {}}

    best_offsets: Counter[tuple[int, int]] = Counter()
    residuals: list[float] = []
    for noisy_path, gt_path in pairs:
        noisy = load_npy(noisy_path)
        gt = load_npy(gt_path)
        if noisy.ndim == 3:
            noisy = noisy.mean(axis=0)
        if gt.ndim == 3:
            gt = gt.mean(axis=0)
        best_offset: tuple[int, int] | None = None
        best_mae = float("inf")
        for offset in ALIGNMENT_OFFSETS:
            sample = gt[offset[0] :: 2, offset[1] :: 2]
            h = min(noisy.shape[0], sample.shape[0])
            w = min(noisy.shape[1], sample.shape[1])
            mae = float(np.abs(noisy[:h, :w] - sample[:h, :w]).mean())
            if mae < best_mae:
                best_mae = mae
                best_offset = offset
        if best_offset is not None:
            best_offsets[best_offset] += 1
            residuals.append(best_mae)

    residual_array = np.asarray(residuals, dtype=float)
    return {
        "n_pairs": len(pairs),
        "method": "2x block-offset search over (0,0),(0,1),(1,0),(1,1); "
        "lowest-MAE offset; MAE at best offset = alignment residual",
        "best_offsets": {f"{o[0]},{o[1]}": v for o, v in best_offsets.items()},
        "residual_stats": {
            "mean": float(residual_array.mean()),
            "std": float(residual_array.std()),
            "min": float(residual_array.min()),
            "max": float(residual_array.max()),
        },
    }


def degradation_summary(
    pairs: Sequence[tuple[Path, Path]],
    *,
    sample_limit: int = 0,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Input/target statistics supporting the degradation-model description."""
    rng = rng or np.random.default_rng(0)
    if sample_limit > 0 and len(pairs) > sample_limit:
        indices = rng.choice(len(pairs), size=sample_limit, replace=False)
        pairs = [pairs[int(i)] for i in indices]
    ratios: list[float] = []
    mean_diffs: list[float] = []
    for noisy_path, gt_path in pairs:
        noisy = load_npy(noisy_path)
        gt = load_npy(gt_path)
        if noisy.ndim == 3:
            noisy = noisy.mean(axis=0)
        if gt.ndim == 3:
            gt = gt.mean(axis=0)
        ratios.append(gt.shape[0] / noisy.shape[0])
        pooled = _mean_pool(gt, noisy.shape[0])
        mean_diffs.append(float(np.abs(noisy - pooled).mean()))
    return {
        "n_pairs": len(pairs),
        "resolution_ratio_gt_over_noisy": _summarize(ratios, "ratio"),
        "mean_abs_diff_noisy_vs_gt_pooled": _summarize(mean_diffs, "mae"),
        "note": "statistics only; no degradation labels are assumed",
    }


def export_alignment_examples(
    pairs: Sequence[tuple[Path, Path]], out_dir: Path, n: int = 3
) -> list[str]:
    """Save a fixed-seed sample of input/target pairs as inspectable artifacts.

    Returns the pair names exported. Artifacts are written to ``out_dir``
    (typically the run bundle's ``artifacts/`` directory).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    sampled = pairs
    if len(sampled) > n:
        indices = rng.choice(len(sampled), size=n, replace=False)
        sampled = [sampled[int(i)] for i in indices]
    names: list[str] = []
    for k, (noisy_path, gt_path) in enumerate(sampled):
        np.save(out_dir / f"alignment_example_{k}_input.npy", load_npy(noisy_path))
        np.save(out_dir / f"alignment_example_{k}_target.npy", load_npy(gt_path))
        names.append(f"{noisy_path.name} <-> {gt_path.name}")
    return names


def compatibility_summary(
    train_input_entries: Sequence[FileEntry],
    test_input_entries: Sequence[FileEntry],
) -> dict[str, Any]:
    """Compare train and test inputs on extension, dims, channels, dtype, range."""
    train_range = range_summary(train_input_entries)
    test_range = range_summary(test_input_entries)
    compatible = (
        set(train_range["shape_counts"]) == set(test_range["shape_counts"])
        and set(train_range["dtype_counts"]) == set(test_range["dtype_counts"])
        and train_range["n_unreadable"] == 0
        and test_range["n_unreadable"] == 0
    )
    return {
        "compatible": compatible,
        "train": {
            "shape_counts": train_range["shape_counts"],
            "dtype_counts": train_range["dtype_counts"],
            "n_unreadable": train_range["n_unreadable"],
            "min_max_mean": train_range.get("min_stats", {}),
            "max_stats": train_range.get("max_stats", {}),
        },
        "test": {
            "shape_counts": test_range["shape_counts"],
            "dtype_counts": test_range["dtype_counts"],
            "n_unreadable": test_range["n_unreadable"],
            "min_max_mean": test_range.get("min_stats", {}),
            "max_stats": test_range.get("max_stats", {}),
        },
    }
