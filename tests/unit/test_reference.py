"""Tests for the deterministic reference reconstruction and the classical
restoration baseline (Phase 2, classical reference path)."""

from __future__ import annotations

import numpy as np
import pytest

from evidence_net.models.reference import (
    bilinear_upsample,
    classical_restoration,
    deterministic_reconstruction,
    median_filter,
)


def test_bilinear_upsample_doubles_shape() -> None:
    image = np.random.default_rng(0).random((8, 8))
    out = bilinear_upsample(image)
    assert out.shape == (16, 16)


def test_bilinear_upsample_preserves_constant_image() -> None:
    constant = np.full((8, 8), 0.5)
    out = bilinear_upsample(constant)
    assert np.allclose(out, 0.5)


def test_bilinear_upsample_scale_one_is_identity() -> None:
    image = np.random.default_rng(1).random((8, 8))
    assert np.allclose(bilinear_upsample(image, scale=1), image)


def test_bilinear_upsample_linear_image_exact() -> None:
    # For a linear ramp, bilinear interpolation is exact: the output at
    # output coordinate o samples the ramp at source (o + 0.5) / 2 - 0.5.
    ramp = np.tile(np.linspace(0.0, 1.0, 8), (8, 1))
    out = bilinear_upsample(ramp)
    source = (np.arange(16, dtype=np.float64) + 0.5) / 2.0 - 0.5
    # Ramp values are source/7 (step 1/7), clamped to the pixel grid [0, 7].
    expected_values = np.clip(source, 0.0, 7.0) / 7.0
    expected = np.tile(expected_values, (16, 1))
    assert np.allclose(out, expected, atol=1e-12)


def test_bilinear_upsample_rejects_bad_ndim() -> None:
    with pytest.raises(ValueError):
        bilinear_upsample(np.zeros((2, 2, 2, 2)))


def test_deterministic_reconstruction_is_bilinear() -> None:
    image = np.random.default_rng(2).random((16, 16))
    assert np.allclose(deterministic_reconstruction(image), bilinear_upsample(image))


def test_median_filter_preserves_constant() -> None:
    constant = np.full((8, 8), 0.5)
    assert np.allclose(median_filter(constant), 0.5)


def test_median_filter_removes_salt_and_pepper() -> None:
    # Constant background with sparse outliers: a 3x3 median restores the
    # background exactly wherever an outlier is corrupted.
    clean = np.full((16, 16), 0.5)
    noisy = clean.copy()
    noisy[::4, ::4] = 0.0
    noisy[2::4, 2::4] = 1.0
    filtered = median_filter(noisy, size=3)
    assert np.allclose(filtered[::4, ::4], 0.5, atol=1e-12)
    assert np.allclose(filtered[2::4, 2::4], 0.5, atol=1e-12)


def test_median_filter_rejects_even_size() -> None:
    with pytest.raises(ValueError):
        median_filter(np.zeros((4, 4)), size=4)


def test_classical_restoration_shape_and_range() -> None:
    image = np.random.default_rng(4).random((8, 8))
    out = classical_restoration(image)
    assert out.shape == (16, 16)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
