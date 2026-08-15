"""Deterministic reference reconstruction and classical restoration baselines.

Phase 2 baselines: a deterministic bilinear up-sampling reconstruction and a
classical denoising + up-sampling restoration. Both are seed-free,
deterministic, and suitable as comparison anchors for learned models. See
``docs/evaluation-protocol.md``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

UPSCALE_FACTOR = 2
MEDIAN_FILTER_SIZE = 5

Restorer = Callable[[np.ndarray], np.ndarray]


def _bilinear_coordinates(
    output_size: int, scale: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Source coordinates for bilinear resampling of one axis."""
    source = (np.arange(output_size, dtype=np.float64) + 0.5) / scale - 0.5
    lower = np.floor(source).astype(np.int64)
    upper = lower + 1
    weight = source - lower
    return lower, upper, 1.0 - weight, weight


def bilinear_upsample(array: np.ndarray, scale: int = UPSCALE_FACTOR) -> np.ndarray:
    """Deterministic bilinear up-sampling (2D; 3D arrays are applied per channel)."""
    if array.ndim not in (2, 3):
        raise ValueError(f"bilinear_upsample expects 2D or 3D input, got ndim={array.ndim}")
    work = np.asarray(array, dtype=np.float64)
    if work.ndim == 3:
        return np.stack([bilinear_upsample(work[c], scale) for c in range(work.shape[0])])

    height, width = work.shape
    out_height, out_width = height * scale, width * scale
    y_lo, y_hi, wy_lo, wy_hi = _bilinear_coordinates(out_height, scale)
    x_lo, x_hi, wx_lo, wx_hi = _bilinear_coordinates(out_width, scale)
    y_lo = np.clip(y_lo, 0, height - 1)
    y_hi = np.clip(y_hi, 0, height - 1)
    x_lo = np.clip(x_lo, 0, width - 1)
    x_hi = np.clip(x_hi, 0, width - 1)

    # Interpolate along x at both y rows, then along y.
    row_lo = work[y_lo][:, x_lo] * wx_lo[None, :] + work[y_lo][:, x_hi] * wx_hi[None, :]
    row_hi = work[y_hi][:, x_lo] * wx_lo[None, :] + work[y_hi][:, x_hi] * wx_hi[None, :]
    return row_lo * wy_lo[:, None] + row_hi * wy_hi[:, None]


def deterministic_reconstruction(array: np.ndarray) -> np.ndarray:
    """Deterministic reference: bilinear 2x up-sampling of the input."""
    return bilinear_upsample(array, scale=UPSCALE_FACTOR)


def median_filter(array: np.ndarray, size: int = MEDIAN_FILTER_SIZE) -> np.ndarray:
    """Square median filter with reflect padding (classical denoising)."""
    if size % 2 != 1:
        raise ValueError(f"median filter size must be odd, got {size}")
    if array.ndim == 3:
        return np.stack([median_filter(array[c], size) for c in range(array.shape[0])])
    pad = size // 2
    padded = np.pad(array, pad, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
    return np.median(windows, axis=(-2, -1))


def classical_restoration(array: np.ndarray) -> np.ndarray:
    """Classical baseline: 5x5 median denoising followed by bilinear 2x up-sampling."""
    return bilinear_upsample(median_filter(array, MEDIAN_FILTER_SIZE), scale=UPSCALE_FACTOR)
