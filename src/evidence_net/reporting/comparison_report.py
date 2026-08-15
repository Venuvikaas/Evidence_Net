"""Restoration comparison reporting.

Generates per-sample comparison sheets (input / output / target / error /
edges) as PNG images with a dependency-free writer, plus a markdown report
that tabulates grouped aggregates for every restorer. See
``docs/evaluation-protocol.md`` section 5.
"""

from __future__ import annotations

import json
import struct
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from evidence_net.evaluation.metrics import binary_edges
from evidence_net.inference.baseline import RestorerResult

# Panel order in every comparison sheet (no text rendering in pure numpy,
# so the order is documented here and in the report).
PANEL_ORDER = ("input", "output", "target", "error", "edges")


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def write_png(path: Path, array: np.ndarray, *, vmin: float = 0.0, vmax: float = 1.0) -> None:
    """Write a grayscale 8-bit PNG from a 2D float array (no dependencies)."""
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"write_png expects a 2D array, got ndim={arr.ndim}")
    scale = 255.0 / (vmax - vmin) if vmax > vmin else 0.0
    pixels = np.clip(np.round((arr - vmin) * scale), 0, 255).astype(np.uint8)
    height, width = pixels.shape
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw, 6))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _montage(panels: Sequence[np.ndarray]) -> np.ndarray:
    if not panels:
        raise ValueError("montage requires at least one panel")
    target_shape = panels[0].shape
    for panel in panels[1:]:
        array = np.asarray(panel)
        if array.ndim != 2 or array.shape != target_shape:
            raise ValueError("all comparison panels must share the same 2D shape")
    return np.hstack([np.asarray(panel, dtype=np.float64) for panel in panels])


def _resize_to_grid(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor upscale to the target grid (display only).

    Super-resolution comparisons place a small input beside full-resolution
    panels; the input is repeated (never interpolated) so it stays visually
    faithful and shares the sheet grid.
    """
    if array.shape == target_shape:
        return array
    height, width = array.shape
    target_height, target_width = target_shape
    if target_height % height or target_width % width:
        raise ValueError("input panel must be an integer factor of the sheet grid")
    return np.repeat(
        np.repeat(array, target_height // height, axis=0), target_width // width, axis=1
    )


def _error_map(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(target, dtype=np.float64) - np.asarray(prediction, dtype=np.float64))


def write_comparison_sheet(
    artifacts_dir: Path,
    sample_index: int,
    input_: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    metrics: Mapping[str, Any],
) -> Path:
    """Write one sample's comparison sheet PNG plus its metrics JSON.

    Panels (left to right): input, output, target, error, edges of the
    target. Returns the sheet path.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    grid = target.shape
    panels = [
        _resize_to_grid(np.asarray(input_, dtype=np.float64), grid),
        prediction,
        target,
        _error_map(target, prediction),
        binary_edges(target).astype(np.float64),
    ]
    sheet_name = f"comparison-{sample_index:06d}"
    sheet_path = artifacts_dir / f"{sheet_name}.png"
    write_png(sheet_path, _montage(panels), vmin=0.0, vmax=1.0)
    (artifacts_dir / f"{sheet_name}.json").write_text(
        json.dumps(dict(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sheet_path


def _format_aggregate(aggregate: dict[str, float | int]) -> str:
    return f"{aggregate['mean']:.4f} [{aggregate['ci_lo']:.4f}, {aggregate['ci_hi']:.4f}]"


def write_comparison_report(
    run_dir: Path,
    results: Mapping[str, RestorerResult],
    *,
    sample_ids: Sequence[str],
    n_samples: int,
    split_label: str,
) -> Path:
    """Write ``comparison-report.md`` summarizing grouped aggregates."""
    rows: list[str] = []
    rows.append(f"# Baseline comparison report ({split_label} split)\n")
    rows.append(
        f"- Samples: {n_samples} paired groups (`{', '.join(map(str, sample_ids[:5]))}"
        + ("...`)" if n_samples > 5 else "`)")
    )
    rows.append(
        "- Statistical unit: source group (image); CIs are 95% group "
        "bootstraps, never pixel counts.\n"
    )
    rows.append("| restorer | PSNR (dB) | SSIM | MAE | edge displacement (px) | structural error |")
    rows.append("| --- | --- | --- | --- | --- | --- |")
    for name in sorted(results):
        aggregates = results[name].aggregates
        rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    _format_aggregate(aggregates["psnr"]),
                    _format_aggregate(aggregates["ssim"]),
                    _format_aggregate(aggregates["mae"]),
                    _format_aggregate(aggregates["edge_displacement_px"]),
                    _format_aggregate(aggregates["structural_error"]),
                ]
            )
            + " |"
        )
    rows.append("")
    rows.append("Frequency-band relative power differences (per restorer):\n")
    for name in sorted(results):
        bands = {
            key: f"{value['mean']:+.4f}"
            for key, value in results[name].aggregates.items()
            if key.startswith("frequency_bands.")
        }
        rows.append(f"- `{name}`: {bands}")
    rows.append("")
    rows.append("## Panel order in comparison sheets")
    rows.append(
        "Each `artifacts/comparison-<index>.png` shows five panels left to "
        f"right: {', '.join(PANEL_ORDER)}.\n"
    )
    report_path = run_dir / "comparison-report.md"
    report_path.write_text("\n".join(rows), encoding="utf-8")
    return report_path
