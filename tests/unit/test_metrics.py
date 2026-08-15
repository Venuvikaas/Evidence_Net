"""Metric behavior tests with analytically simple fixtures.

Each fixture is chosen so the expected value can be derived by hand:
constant images, uniform perturbations, single-edge steps, and
identical-image degeneracies (per docs/evaluation-protocol.md section 3).
"""

from __future__ import annotations

import numpy as np
import pytest

from evidence_net.evaluation.metrics import (
    all_metrics,
    binary_edges,
    edge_displacement,
    edge_magnitude,
    frequency_band_diagnostics,
    mae,
    psnr,
    ssim,
    structural_error,
)

# 32x32 images keep every fixture small while still exercising the window.
SIZE = 32


def constant(value: float) -> np.ndarray:
    return np.full((SIZE, SIZE), value, dtype=np.float64)


def random_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((SIZE, SIZE))


def test_identical_images_degenerate_values() -> None:
    image = random_image()
    assert psnr(image, image) == float("inf")
    assert mae(image, image) == 0.0
    assert ssim(image, image) == pytest.approx(1.0)
    assert edge_displacement(image, image) == 0.0
    assert structural_error(image, image) == 0.0
    bands = frequency_band_diagnostics(image, image)
    assert all(value == 0.0 for value in bands.values())


def test_psnr_matches_hand_computation() -> None:
    target = constant(0.5)
    predicted = constant(0.5 + 0.1)
    expected = float(10.0 * np.log10(1.0 / 0.1**2))
    assert psnr(target, predicted) == pytest.approx(expected)
    assert psnr(predicted, target) == pytest.approx(expected)


def test_psnr_zero_mse_is_infinite() -> None:
    assert psnr(constant(0.25), constant(0.25)) == float("inf")


def test_mae_matches_hand_computation() -> None:
    target = constant(0.0)
    predicted = constant(0.5)
    assert mae(target, predicted) == pytest.approx(0.5)
    assert mae(target, constant(0.25)) == pytest.approx(0.25)


def test_ssim_identical_with_known_constant() -> None:
    assert ssim(constant(0.5), constant(0.5)) == pytest.approx(1.0)


def test_ssim_symmetric_and_bounded() -> None:
    a = random_image(1)
    b = random_image(2)
    assert ssim(a, b) == pytest.approx(ssim(b, a))
    assert -1.0 <= ssim(a, b) <= 1.0


def test_edge_magnitude_flat_image_is_zero() -> None:
    assert edge_magnitude(constant(0.5)).max() == 0.0


def test_binary_edges_clean_step() -> None:
    # Single vertical step at the image center.
    step = np.zeros((SIZE, SIZE))
    step[:, SIZE // 2 :] = 1.0
    edges = binary_edges(step)
    # The Sobel response peaks along the step; a thin edge column exists.
    assert edges.any()
    column_counts = edges.sum(axis=0)
    assert column_counts.max() > 0


def test_edge_displacement_shifted_step() -> None:
    target = np.zeros((SIZE, SIZE))
    target[:, SIZE // 2 :] = 1.0
    predicted = np.zeros((SIZE, SIZE))
    predicted[:, SIZE // 2 + 1 :] = 1.0
    displacement = edge_displacement(target, predicted)
    # A 1px-shifted step gives a mean displacement near 1px.
    assert displacement == pytest.approx(1.0, abs=0.5)


def test_edge_displacement_empty_target_edges() -> None:
    flat = constant(0.5)
    assert edge_displacement(flat, random_image()) == 0.0


def test_structural_error_identical_is_zero() -> None:
    image = random_image()
    assert structural_error(image, image) == 0.0


def test_frequency_diagnostics_identical_are_zero() -> None:
    image = random_image()
    bands = frequency_band_diagnostics(image, image)
    assert set(bands) == {"[0.000,0.125)", "[0.125,0.500)", "[0.500,1.000)"}
    assert all(value == 0.0 for value in bands.values())


def test_frequency_diagnostics_zero_target_power_is_zero() -> None:
    # Flat target has no power in high bands; relative diff must not blow up.
    bands = frequency_band_diagnostics(constant(0.5), random_image())
    assert all(np.isfinite(value) for value in bands.values())


def test_all_metrics_shape_and_keys() -> None:
    image = random_image()
    result = all_metrics(constant(0.5), image)
    assert set(result) == {
        "psnr",
        "ssim",
        "mae",
        "edge_displacement_px",
        "structural_error",
        "frequency_bands",
    }
    expected_mse = float(np.mean((constant(0.5) - image) ** 2))
    assert result["psnr"] == pytest.approx(float(10.0 * np.log10(1.0 / expected_mse)))
