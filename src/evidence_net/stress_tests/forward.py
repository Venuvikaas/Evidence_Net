"""Bounded modality-specific forward operator family (Phase 7).

Implements ``forward-model-v1`` (docs/contracts/forward-model-v1.md): a
declared set of plausible degradation operators mapping the output grid
(256x256 for the official dataset) to the input grid (128x128) with fixed
scale 2. Operators are deterministic or stochastic, their parameters are
bounded and validated at construction (out-of-bounds raises ``ForwardError``
— misspecification is detected, never silently clamped), and the operation
order is part of each operator definition. The family is *compatibility, not
truth*: it never claims to identify the true hidden degradation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F

# Frozen bounds from forward-model-v1.
SCALE = 2
BLUR_SIGMA_MIN = 0.0
BLUR_SIGMA_MAX = 2.0
NOISE_SIGMA_MIN = 0.0
NOISE_SIGMA_MAX = 0.1


class ForwardError(ValueError):
    """Raised for invalid operators or out-of-bounds parameters (misspecification)."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForwardConfig:
    """Validated operator-family configuration (forward-model-v1)."""

    version: str = "forward-model-v1"
    scale: int = SCALE
    blur_sigma: float = 0.5
    noise_sigma: float = 0.01
    seed: int = 0
    n_boot: int = 1000
    deterministic_operators: tuple[str, ...] = ("bilinear", "area", "blur")
    stochastic_operators: tuple[str, ...] = ("noisy-blur",)

    def validate(self) -> None:
        if self.version != "forward-model-v1":
            raise ForwardError(f"version must be forward-model-v1, got {self.version}")
        if self.scale != SCALE:
            raise ForwardError(f"scale must be {SCALE}, got {self.scale}")
        if not BLUR_SIGMA_MIN <= self.blur_sigma <= BLUR_SIGMA_MAX:
            raise ForwardError(
                f"blur_sigma must be in [{BLUR_SIGMA_MIN}, {BLUR_SIGMA_MAX}], got {self.blur_sigma}"
            )
        if not NOISE_SIGMA_MIN <= self.noise_sigma <= NOISE_SIGMA_MAX:
            raise ForwardError(
                f"noise_sigma must be in [{NOISE_SIGMA_MIN}, {NOISE_SIGMA_MAX}], "
                f"got {self.noise_sigma}"
            )
        if self.n_boot < 1:
            raise ForwardError(f"n_boot must be >= 1, got {self.n_boot}")
        known_deterministic = {"bilinear", "area", "blur"}
        known_stochastic = {"noisy-blur"}
        unknown = set(self.deterministic_operators) - known_deterministic
        if unknown:
            raise ForwardError(f"unknown deterministic operators: {sorted(unknown)}")
        unknown = set(self.stochastic_operators) - known_stochastic
        if unknown:
            raise ForwardError(f"unknown stochastic operators: {sorted(unknown)}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "scale": self.scale,
            "blur_sigma": self.blur_sigma,
            "noise_sigma": self.noise_sigma,
            "seed": self.seed,
            "n_boot": self.n_boot,
            "deterministic_operators": list(self.deterministic_operators),
            "stochastic_operators": list(self.stochastic_operators),
        }


def load_forward_config(path: Path) -> ForwardConfig:
    """Load and validate a forward-model config from YAML (forward-model-v1)."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ForwardError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ForwardError(f"config root must be a mapping: {path}")
    allowed = {
        "version",
        "scale",
        "blur_sigma",
        "noise_sigma",
        "seed",
        "n_boot",
        "deterministic_operators",
        "stochastic_operators",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ForwardError(f"unknown config keys: {sorted(unknown)}")

    def pick(key: str, default: Any) -> Any:
        return raw.get(key, default)

    def as_int(value: Any, key: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ForwardError(f"{key} must be an int, got {type(value).__name__}")
        return value

    def as_float(value: Any, key: str) -> float:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ForwardError(f"{key} must be a number, got {type(value).__name__}")
        return float(value)

    def as_str_tuple(value: Any, key: str) -> tuple[str, ...]:
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ForwardError(f"{key} must be a list of strings")
        return tuple(value)

    config = ForwardConfig(
        version=str(pick("version", "forward-model-v1")),
        scale=as_int(pick("scale", SCALE), "scale"),
        blur_sigma=as_float(pick("blur_sigma", 0.5), "blur_sigma"),
        noise_sigma=as_float(pick("noise_sigma", 0.01), "noise_sigma"),
        seed=as_int(pick("seed", 0), "seed"),
        n_boot=as_int(pick("n_boot", 1000), "n_boot"),
        deterministic_operators=as_str_tuple(
            pick("deterministic_operators", ["bilinear", "area", "blur"]),
            "deterministic_operators",
        ),
        stochastic_operators=as_str_tuple(
            pick("stochastic_operators", ["noisy-blur"]), "stochastic_operators"
        ),
    )
    config.validate()
    return config


# ---------------------------------------------------------------------------
# Image helpers (forward-model-v1 grids and bounds)
# ---------------------------------------------------------------------------


def _as_2d(clean: np.ndarray) -> np.ndarray:
    """Accept ``(H, W)`` or ``(1, H, W)`` and return a float64 ``(H, W)`` plane."""
    array = np.asarray(clean)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ForwardError(f"clean image must be (H, W) or (1, H, W), got shape {array.shape}")
    return array.astype(np.float64, copy=False)


def _require_even(shape: tuple[int, ...], name: str) -> None:
    if shape[0] % 2 != 0 or shape[1] % 2 != 0:
        raise ForwardError(f"{name} dimensions must be even for scale {SCALE}, got {shape}")


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with reflect padding.

    ``sigma`` must be within ``[BLUR_SIGMA_MIN, BLUR_SIGMA_MAX]``; a zero
    sigma returns the image unchanged. Constant images stay constant.
    """
    if not BLUR_SIGMA_MIN <= sigma <= BLUR_SIGMA_MAX:
        raise ForwardError(
            f"gaussian_blur sigma must be in [{BLUR_SIGMA_MIN}, {BLUR_SIGMA_MAX}], got {sigma}"
        )
    array = _as_2d(image)
    if sigma == 0.0:
        return array.copy()
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    padded = np.pad(array, radius, mode="reflect")
    blurred = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)
    blurred = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 0, blurred)
    return blurred


def bilinear_downsample(image: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """Bilinear 2x down-sample (``align_corners=False``), matching the anchor family."""
    if scale != SCALE:
        raise ForwardError(f"scale must be {SCALE}, got {scale}")
    array = _as_2d(image)
    _require_even(array.shape, "image")
    tensor = torch.from_numpy(array)[None, None]
    down = F.interpolate(tensor, scale_factor=1.0 / SCALE, mode="bilinear", align_corners=False)
    return down[0, 0].numpy()


def area_downsample(image: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """2x2 mean-pool down-sample (exact average over each 2x2 block)."""
    if scale != SCALE:
        raise ForwardError(f"scale must be {SCALE}, got {scale}")
    array = _as_2d(image)
    _require_even(array.shape, "image")
    h, w = array.shape
    return array.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


# ---------------------------------------------------------------------------
# Operator family
# ---------------------------------------------------------------------------


class ForwardOperator(ABC):
    """A bounded forward operator from the output grid to the input grid."""

    name: str
    kind: str

    def __init__(self, scale: int = SCALE) -> None:
        if scale != SCALE:
            raise ForwardError(f"scale must be {SCALE}, got {scale}")
        self.scale = scale

    @abstractmethod
    def apply(self, clean: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Map a clean image on the output grid to an observation on the input grid."""

    @property
    def is_stochastic(self) -> bool:
        return self.kind == "stochastic"

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "scale": self.scale}


class BilinearDownsample(ForwardOperator):
    """Deterministic bilinear 2x down-sample."""

    name = "bilinear"
    kind = "deterministic"

    def apply(self, clean: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        return bilinear_downsample(clean, self.scale)


class AreaDownsample(ForwardOperator):
    """Deterministic 2x2 mean-pool down-sample."""

    name = "area"
    kind = "deterministic"

    def apply(self, clean: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        return area_downsample(clean, self.scale)


class BlurDownsample(ForwardOperator):
    """Deterministic Gaussian blur then bilinear 2x down-sample."""

    name = "blur"
    kind = "deterministic"

    def __init__(self, blur_sigma: float, scale: int = SCALE) -> None:
        super().__init__(scale)
        if not BLUR_SIGMA_MIN <= blur_sigma <= BLUR_SIGMA_MAX:
            raise ForwardError(
                f"blur_sigma must be in [{BLUR_SIGMA_MIN}, {BLUR_SIGMA_MAX}], got {blur_sigma}"
            )
        self.blur_sigma = blur_sigma

    def apply(self, clean: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        return bilinear_downsample(gaussian_blur(clean, self.blur_sigma), self.scale)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "blur_sigma": self.blur_sigma}


class NoisyBlurDownsample(ForwardOperator):
    """Stochastic: Gaussian blur -> bilinear 2x down-sample -> additive Gaussian noise.

    Noise is sampled from ``N(0, noise_sigma^2)`` with the passed (seeded)
    generator, or with ``np.random.default_rng(0)`` when none is given, so
    stochastic evaluations are reproducible under an explicit seed.
    """

    name = "noisy-blur"
    kind = "stochastic"

    def __init__(
        self, blur_sigma: float, noise_sigma: float, scale: int = SCALE, seed: int = 0
    ) -> None:
        super().__init__(scale)
        if not BLUR_SIGMA_MIN <= blur_sigma <= BLUR_SIGMA_MAX:
            raise ForwardError(
                f"blur_sigma must be in [{BLUR_SIGMA_MIN}, {BLUR_SIGMA_MAX}], got {blur_sigma}"
            )
        if not NOISE_SIGMA_MIN <= noise_sigma <= NOISE_SIGMA_MAX:
            raise ForwardError(
                f"noise_sigma must be in [{NOISE_SIGMA_MIN}, {NOISE_SIGMA_MAX}], got {noise_sigma}"
            )
        self.blur_sigma = blur_sigma
        self.noise_sigma = noise_sigma
        self.seed = seed

    def apply(self, clean: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        generator = rng if rng is not None else np.random.default_rng(self.seed)
        blurred = bilinear_downsample(gaussian_blur(clean, self.blur_sigma), self.scale)
        noise = generator.normal(0.0, self.noise_sigma, size=blurred.shape)
        return blurred + noise

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "blur_sigma": self.blur_sigma,
            "noise_sigma": self.noise_sigma,
            "seed": self.seed,
        }


def build_operator_family(
    config: ForwardConfig | None = None,
    *,
    blur_sigma: float | None = None,
    noise_sigma: float | None = None,
    seed: int = 0,
) -> list[ForwardOperator]:
    """Build the declared operator family from a validated config (or defaults)."""
    cfg = config or ForwardConfig()
    cfg.validate()
    blur = cfg.blur_sigma if blur_sigma is None else blur_sigma
    noise = cfg.noise_sigma if noise_sigma is None else noise_sigma
    operators: list[ForwardOperator] = []
    for name in cfg.deterministic_operators:
        if name == "bilinear":
            operators.append(BilinearDownsample(cfg.scale))
        elif name == "area":
            operators.append(AreaDownsample(cfg.scale))
        elif name == "blur":
            operators.append(BlurDownsample(blur, cfg.scale))
    for name in cfg.stochastic_operators:
        if name == "noisy-blur":
            operators.append(NoisyBlurDownsample(blur, noise, cfg.scale, seed=seed))
    return operators


def operator_by_name(operators: list[ForwardOperator], name: str) -> ForwardOperator:
    """Return the operator with the given name (raises if absent)."""
    for operator in operators:
        if operator.name == name:
            return operator
    raise ForwardError(f"operator not in family: {name}")


# ---------------------------------------------------------------------------
# Non-identifiability and misspecification cases (Phase 7 stress suite)
# ---------------------------------------------------------------------------


def non_identifiable_stripe_pair(
    size: int = 256, sigma: float = 1.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two clean images that differ everywhere but re-degrade near-identically.

    Clean A is alternating 0/1 columns (period 2); clean B is its inverse.
    Under Gaussian blur + down-sample both observations collapse toward the
    same ~0.5 field, so the family cannot distinguish them — the canonical
    non-identifiability case (modality contract section 10).
    """
    columns = (np.arange(size) % 2).astype(np.float64)
    clean_a = np.tile(columns, (size, 1))
    clean_b = 1.0 - clean_a
    operator = BlurDownsample(sigma)
    return clean_a, clean_b, operator.apply(clean_a), operator.apply(clean_b)


def non_identifiable_line_pair(
    size: int = 256, sigma: float = 1.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A single line present vs absent, blurred below family resolution.

    The two clean images differ by one full-width structure; after
    blur + down-sample the observations differ only by a faint smear, below
    the family's resolution. Used as a review case: the family cannot certify
    whether the line existed.
    """
    clean_a = np.zeros((size, size), dtype=np.float64)
    clean_a[:, size // 2] = 1.0
    clean_b = np.zeros((size, size), dtype=np.float64)
    operator = BlurDownsample(sigma)
    return clean_a, clean_b, operator.apply(clean_a), operator.apply(clean_b)
