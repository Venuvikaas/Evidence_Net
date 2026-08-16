"""Forward operator family tests (Phase 7, forward-model-v1).

Analytical fixtures: exact mean-pooling, constant-preservation, parameter
bounds (misspecification detection), seeded stochastic reproducibility and
variance, operation-order sensitivity, and the canonical non-identifiability
cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from evidence_net.evaluation.metrics import mae
from evidence_net.stress_tests.forward import (
    AreaDownsample,
    BilinearDownsample,
    BlurDownsample,
    ForwardConfig,
    ForwardError,
    NoisyBlurDownsample,
    area_downsample,
    bilinear_downsample,
    build_operator_family,
    gaussian_blur,
    non_identifiable_line_pair,
    non_identifiable_stripe_pair,
)


def test_area_downsample_is_exact_mean_pooling() -> None:
    image = np.arange(16, dtype=np.float64).reshape(4, 4)
    down = area_downsample(image)
    assert down.shape == (2, 2)
    # 2x2 blocks: [[0,1],[4,5]] -> 2.5, [[2,3],[6,7]] -> 4.5, ...
    np.testing.assert_allclose(down, [[2.5, 4.5], [10.5, 12.5]])


def test_area_downsample_class_matches_function() -> None:
    image = np.random.default_rng(0).random((16, 16))
    np.testing.assert_allclose(AreaDownsample().apply(image), area_downsample(image))


def test_bilinear_downsample_preserves_constants() -> None:
    image = np.full((16, 16), 0.7)
    np.testing.assert_allclose(bilinear_downsample(image), np.full((8, 8), 0.7), atol=1e-6)


def test_bilinear_downsample_contract_shape() -> None:
    image = np.random.default_rng(1).random((256, 256))
    assert bilinear_downsample(image).shape == (128, 128)


def test_gaussian_blur_preserves_constants_and_identity_at_zero() -> None:
    constant = np.full((16, 16), 0.4)
    np.testing.assert_allclose(gaussian_blur(constant, 1.5), constant, atol=1e-12)
    rng = np.random.default_rng(0)
    image = rng.random((16, 16))
    np.testing.assert_allclose(gaussian_blur(image, 0.0), image)


def test_blur_downsample_constant_stays_constant() -> None:
    operator = BlurDownsample(blur_sigma=1.5)
    image = np.full((16, 16), 0.3)
    np.testing.assert_allclose(operator.apply(image), np.full((8, 8), 0.3), atol=1e-6)


def test_operator_bounds_reject_misspecification() -> None:
    with pytest.raises(ForwardError):
        BlurDownsample(blur_sigma=-0.1)
    with pytest.raises(ForwardError):
        BlurDownsample(blur_sigma=2.5)
    with pytest.raises(ForwardError):
        NoisyBlurDownsample(blur_sigma=0.5, noise_sigma=0.2)
    with pytest.raises(ForwardError):
        NoisyBlurDownsample(blur_sigma=0.5, noise_sigma=-0.01)
    with pytest.raises(ForwardError):
        BilinearDownsample(scale=3)
    with pytest.raises(ForwardError):
        gaussian_blur(np.zeros((8, 8)), -1.0)


def test_config_validation_rejects_bad_bounds() -> None:
    with pytest.raises(ForwardError):
        ForwardConfig(scale=3).validate()
    with pytest.raises(ForwardError):
        ForwardConfig(blur_sigma=3.0).validate()
    with pytest.raises(ForwardError):
        ForwardConfig(noise_sigma=0.5).validate()
    with pytest.raises(ForwardError):
        ForwardConfig(deterministic_operators=("bogus",)).validate()


def test_stochastic_operator_is_seeded_reproducible() -> None:
    operator = NoisyBlurDownsample(blur_sigma=0.5, noise_sigma=0.02, seed=0)
    image = np.random.default_rng(0).random((16, 16))
    first = operator.apply(image, np.random.default_rng(42))
    second = operator.apply(image, np.random.default_rng(42))
    np.testing.assert_allclose(first, second)
    # A different seed draws different noise (variance > 0).
    other = operator.apply(image, np.random.default_rng(43))
    assert float(np.abs(first - other).mean()) > 1e-9


def test_stochastic_operator_has_positive_variance() -> None:
    operator = NoisyBlurDownsample(blur_sigma=0.0, noise_sigma=0.05, seed=0)
    image = np.zeros((16, 16))
    draws = [operator.apply(image, np.random.default_rng(i)) for i in range(5)]
    stack = np.stack(draws)
    assert stack.std() > 0.0
    # Without noise the operator reduces to the deterministic blur path.
    clean_operator = NoisyBlurDownsample(blur_sigma=0.0, noise_sigma=0.0, seed=0)
    draws_clean = [clean_operator.apply(image, np.random.default_rng(i)) for i in range(3)]
    for draw in draws_clean:
        np.testing.assert_allclose(draw, np.zeros((8, 8)))


def test_operation_order_matters() -> None:
    # blur-then-downsample differs from downsample-then-blur on a structured
    # image: the family records its order; the true order is not claimed.
    rng = np.random.default_rng(0)
    image = rng.random((32, 32))
    blur_then_down = bilinear_downsample(gaussian_blur(image, 1.0))
    down_then_blur = gaussian_blur(bilinear_downsample(image), 1.0)
    assert float(np.abs(blur_then_down - down_then_blur).mean()) > 1e-6


def test_non_identifiability_stripes() -> None:
    clean_a, clean_b, obs_a, obs_b = non_identifiable_stripe_pair(size=64, sigma=1.5)
    # The clean images are maximally different ...
    assert mae(clean_a, clean_b) == pytest.approx(1.0)
    # ... but re-degrade to near-identical observations.
    assert mae(obs_a, obs_b) < 0.01


def test_non_identifiability_line_present_absent() -> None:
    clean_a, clean_b, obs_a, obs_b = non_identifiable_line_pair(size=64, sigma=1.5)
    # A full-width structure differs between the cleans ...
    assert mae(clean_a, clean_b) > 0.0
    # ... yet the observations differ only by a faint smear, below family
    # resolution: the family cannot certify whether the line existed.
    assert mae(obs_a, obs_b) < 0.06


def test_build_operator_family_matches_config() -> None:
    config = ForwardConfig()
    operators = build_operator_family(config)
    names = [operator.name for operator in operators]
    assert names == ["bilinear", "area", "blur", "noisy-blur"]
    assert all(not operator.is_stochastic for operator in operators[:-1])
    assert operators[-1].is_stochastic


def test_odd_dimensions_rejected() -> None:
    with pytest.raises(ForwardError):
        bilinear_downsample(np.zeros((15, 16)))
    with pytest.raises(ForwardError):
        area_downsample(np.zeros((16, 15)))
