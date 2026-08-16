"""Acquisition artifact suite (Phase 10, structural-risk-v1 section 3).

Modality-plausible pre-inference artifacts applied to the degraded input:
sensor noise, column striping, gain non-uniformity, dead pixels, and a local
blur patch. All are labeled ``acquisition`` — a separate threat model from
candidate manipulations, and reports must keep them apart. Parameters come
from the frozen hidden stress definitions; outputs are clipped to [0, 1].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np

from evidence_net.stress_tests.forward import gaussian_blur
from evidence_net.stress_tests.hidden_stress import HiddenStressError, stress_params


class AcquisitionError(ValueError):
    """Raised for invalid acquisition artifacts or parameters."""


def _as_2d(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise AcquisitionError(f"image must be (H, W), got shape {array.shape}")
    return array.astype(np.float64, copy=True)


class AcquisitionArtifact(ABC):
    """A modality-plausible pre-inference artifact."""

    name: str
    threat = "acquisition"

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    @abstractmethod
    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return a modified copy of the input (values clipped to [0, 1])."""

    @abstractmethod
    def effect(self) -> str:
        """Human description of the artifact."""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "threat": self.threat, "effect": self.effect()}


class SensorNoise(AcquisitionArtifact):
    """Additive sensor noise, bounded by the frozen sigma."""

    name = "sensor-noise"

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        sigma = float(self.params.get("noise_sigma", 0.02))
        array = _as_2d(image)
        return np.clip(array + rng.normal(0.0, sigma, size=array.shape), 0.0, 1.0)

    def effect(self) -> str:
        return "adds bounded additive sensor noise"


class ColumnStripe(AcquisitionArtifact):
    """Column-wise striping (sensor line artifacts)."""

    name = "column-stripe"

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        amplitude = float(self.params.get("stripe_amplitude", 0.05))
        array = _as_2d(image)
        columns = np.arange(array.shape[1])
        offset = amplitude * np.sin(2.0 * np.pi * columns / 8.0)
        return np.clip(array + offset[None, :], 0.0, 1.0)

    def effect(self) -> str:
        return "adds column-wise striping"


class GainNonuniformity(AcquisitionArtifact):
    """Multiplicative smooth gain field (sensor shading)."""

    name = "gain-nonuniformity"

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        low, high = self.params.get("gain_range", [0.8, 1.2])
        array = _as_2d(image)
        height, width = array.shape
        rows = np.linspace(low, high, height)
        cols = np.linspace(high, low, width)
        gain = rows[:, None] * cols[None, :]
        return np.clip(array * gain, 0.0, 1.0)

    def effect(self) -> str:
        return "applies a smooth multiplicative gain field"


class DeadPixels(AcquisitionArtifact):
    """Zero out a small frozen fraction of pixels (dead sensor elements)."""

    name = "dead-pixels"

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        fraction = float(self.params.get("dead_pixel_fraction", 0.001))
        array = _as_2d(image)
        mask = rng.random(array.shape) < fraction
        array[mask] = 0.0
        return np.clip(array, 0.0, 1.0)

    def effect(self) -> str:
        return "zeros a small fraction of pixels (dead sensor elements)"


class LocalBlurPatch(AcquisitionArtifact):
    """Blur a central patch (out-of-focus region)."""

    name = "local-blur-patch"

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        sigma = float(self.params.get("blur_sigma", 1.0))
        array = _as_2d(image)
        height, width = array.shape
        blurred = gaussian_blur(array, sigma)
        patch_h = max(4, height // 4)
        patch_w = max(4, width // 4)
        top = height // 2 - patch_h // 2
        left = width // 2 - patch_w // 2
        blended = array.copy()
        blended[top : top + patch_h, left : left + patch_w] = blurred[
            top : top + patch_h, left : left + patch_w
        ]
        return np.clip(blended, 0.0, 1.0)

    def effect(self) -> str:
        return "blurs a central patch (out-of-focus region)"


ARTIFACT_TYPES: dict[str, type[AcquisitionArtifact]] = {
    "sensor-noise": SensorNoise,
    "column-stripe": ColumnStripe,
    "gain-nonuniformity": GainNonuniformity,
    "dead-pixels": DeadPixels,
    "local-blur-patch": LocalBlurPatch,
}


def build_acquisition_suite(
    params: Mapping[str, Any] | None = None,
    names: tuple[str, ...] | None = None,
) -> list[AcquisitionArtifact]:
    """Build the frozen acquisition suite from hidden-stress acquisition params."""
    if params is None:
        try:
            loaded = stress_params()
        except HiddenStressError as exc:
            raise AcquisitionError(f"cannot build acquisition suite: {exc}") from exc
        acquisition_params = loaded["acquisition"]
        assert isinstance(acquisition_params, dict)
        params = acquisition_params
    selected = names or tuple(ARTIFACT_TYPES)
    unknown = set(selected) - set(ARTIFACT_TYPES)
    if unknown:
        raise AcquisitionError(f"unknown artifacts: {sorted(unknown)}")
    return [ARTIFACT_TYPES[name](params) for name in selected]
