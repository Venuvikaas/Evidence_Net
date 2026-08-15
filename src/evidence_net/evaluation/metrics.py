"""Restoration metrics.

Implements the metric contracts in ``docs/evaluation-protocol.md``:
PSNR, SSIM, MAE, edge displacement, structural error, and frequency-band
diagnostics. All metrics are deterministic, computed per image, and operate
on ``[0, 1]`` floating arrays on the same grid.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

DATA_RANGE = 1.0

# SSIM parameters (Wang et al. 2004, defaults used by the contract).
SSIM_WINDOW = 7
SSIM_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03

# Edge / structural parameters (documented in evaluation-protocol.md).
EDGE_THRESHOLD = 0.5
EDGE_DISPLACEMENT_RADIUS = 16
STRUCTURAL_SCALES = (1, 2)
FREQUENCY_BANDS = ((0.0, 1 / 8), (1 / 8, 1 / 2), (1 / 2, 1.0))


def _as_float64(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)


def mae(target: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error between target and predicted images."""
    return float(np.abs(_as_float64(target) - _as_float64(predicted)).mean())


def psnr(target: np.ndarray, predicted: np.ndarray, data_range: float = DATA_RANGE) -> float:
    """Peak signal-to-noise ratio in dB; ``inf`` for identical images."""
    mse = float(np.mean((_as_float64(target) - _as_float64(predicted)) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / mse))


def _gaussian_kernel_1d(window: int, sigma: float) -> np.ndarray:
    coords = np.arange(window) - window // 2
    kernel = np.exp(-(coords**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _filter_separable(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Separable 1D convolution with zero padding (mode='same')."""
    out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 0, array)
    out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, out)
    return out


def ssim(
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    data_range: float = DATA_RANGE,
    window: int = SSIM_WINDOW,
    sigma: float = SSIM_SIGMA,
    k1: float = SSIM_K1,
    k2: float = SSIM_K2,
) -> float:
    """Structural similarity (Wang et al. 2004). Identical images yield 1.0."""
    x = _as_float64(target)
    y = _as_float64(predicted)
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    kernel = _gaussian_kernel_1d(window, sigma)

    mu_x = _filter_separable(x, kernel)
    mu_y = _filter_separable(y, kernel)
    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_xx = _filter_separable(x * x, kernel) - mu_xx
    sigma_yy = _filter_separable(y * y, kernel) - mu_yy
    sigma_xy = _filter_separable(x * y, kernel) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)
    return float(np.mean(numerator / denominator))


_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def _convolve2d(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    padded = np.pad(array, 1, mode="edge")
    result = np.zeros_like(array)
    for i in range(3):
        for j in range(3):
            if kernel[i, j] != 0:
                result += kernel[i, j] * padded[i : i + array.shape[0], j : j + array.shape[1]]
    return result


def edge_magnitude(array: np.ndarray) -> np.ndarray:
    """Normalized Sobel gradient magnitude in [0, 1] (0 if flat)."""
    x = _as_float64(array)
    gx = _convolve2d(x, _SOBEL_X)
    gy = _convolve2d(x, _SOBEL_Y)
    magnitude = np.sqrt(gx**2 + gy**2)
    maximum = float(magnitude.max())
    if maximum == 0.0:
        return np.zeros_like(magnitude)
    return magnitude / maximum


def binary_edges(array: np.ndarray, threshold: float = EDGE_THRESHOLD) -> np.ndarray:
    """Binary edge map at the documented threshold."""
    return edge_magnitude(array) >= threshold


def _distance_transform(mask: np.ndarray, radius: int) -> np.ndarray:
    """4-neighborhood distance transform from the mask, capped at ``radius``."""
    dist = np.full(mask.shape, radius, dtype=np.float64)
    dist[mask] = 0.0
    frontier = mask.copy()
    step = 1
    while step <= radius:
        shifted = np.zeros_like(frontier, dtype=bool)
        shifted[1:, :] |= frontier[:-1, :]
        shifted[:-1, :] |= frontier[1:, :]
        shifted[:, 1:] |= frontier[:, :-1]
        shifted[:, :-1] |= frontier[:, 1:]
        new_frontier = shifted & (dist >= step) & ~mask
        if not new_frontier.any():
            break
        dist[new_frontier] = step
        frontier = new_frontier
        step += 1
    return dist


def edge_displacement(
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    radius: int = EDGE_DISPLACEMENT_RADIUS,
    threshold: float = EDGE_THRESHOLD,
) -> float:
    """Mean bounded Chamfer distance (px) from target edges to predicted edges."""
    target_edges = binary_edges(target, threshold)
    predicted_edges = binary_edges(predicted, threshold)
    if not target_edges.any():
        return 0.0
    distances = _distance_transform(predicted_edges, radius)
    return float(distances[target_edges].mean())


def _mean_pool(array: np.ndarray, scale: int) -> np.ndarray:
    h, w = array.shape
    rows = np.linspace(0, h, scale + 1).astype(int)
    cols = np.linspace(0, w, scale + 1).astype(int)
    out = np.zeros((scale, scale), dtype=np.float64)
    for i in range(scale):
        for j in range(scale):
            block = array[rows[i] : rows[i + 1], cols[j] : cols[j + 1]]
            out[i, j] = block.mean()
    return out


def structural_error(
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    scales: tuple[int, ...] = STRUCTURAL_SCALES,
) -> float:
    """Mean over scales of the mean |edge_magnitude(target) - edge_magnitude(pred)|."""
    values: list[float] = []
    for scale in scales:
        t = _as_float64(target)
        p = _as_float64(predicted)
        if scale > 1:
            t = _mean_pool(t, scale)
            p = _mean_pool(p, scale)
        values.append(float(np.abs(edge_magnitude(t) - edge_magnitude(p)).mean()))
    return float(np.mean(values))


def _radial_power_profile(power: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Normalized radial-frequency coordinate for each rfft2 power bin."""
    height, width = image_shape
    freq_y = np.fft.fftfreq(height)[:, None]
    freq_x = np.fft.rfftfreq(width)[None, :]
    radial = np.sqrt(freq_y**2 + freq_x**2)
    nyquist = 0.5
    return np.minimum(radial / nyquist, 1.0)


def frequency_band_diagnostics(
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    bands: tuple[tuple[float, float], ...] = FREQUENCY_BANDS,
) -> dict[str, float]:
    """Relative power difference per frequency band (relative to Nyquist).

    Returns ``{band_label: relative_diff}`` where relative_diff is
    ``(mean_power_pred - mean_power_target) / mean_power_target`` and 0 when
    the target band power is zero.
    """
    power_t = np.abs(np.fft.rfft2(_as_float64(target))) ** 2
    power_p = np.abs(np.fft.rfft2(_as_float64(predicted))) ** 2
    radial = _radial_power_profile(power_t, target.shape)
    result: dict[str, float] = {}
    for lo, hi in bands:
        mask = (radial >= lo) & (radial < hi)
        mean_t = float(power_t[mask].mean()) if mask.any() else 0.0
        mean_p = float(power_p[mask].mean()) if mask.any() else 0.0
        if mean_t == 0.0:
            result[f"[{lo:.3f},{hi:.3f})"] = 0.0
        else:
            result[f"[{lo:.3f},{hi:.3f})"] = (mean_p - mean_t) / mean_t
    return result


MetricFn = Callable[[np.ndarray, np.ndarray], float]


def all_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, float | dict[str, float]]:
    """All contract metrics for one pair (primary + structural + frequency)."""
    return {
        "psnr": psnr(target, predicted),
        "ssim": ssim(target, predicted),
        "mae": mae(target, predicted),
        "edge_displacement_px": edge_displacement(target, predicted),
        "structural_error": structural_error(target, predicted),
        "frequency_bands": frequency_band_diagnostics(target, predicted),
    }
