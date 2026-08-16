"""Distribution familiarity diagnostic (Phase 9, familiarity-v1 draft).

A reference-distance baseline: degraded inputs are embedded in a frozen
6-component feature vector (pixel stats, band energy, edge density), the
reference population is fit from development inputs only, and familiarity is
the RMS standardized distance to that population against a threshold. The
diagnostic reports shift detection per declared shift group (source,
severity, degradation, acquisition) and evaluates rare valid structures
**separately** so Gate 8's no-systematic-suppression rule is visible.
Familiarity is never correctness: unfamiliar is not wrong, familiar is not
right, and neither certifies the output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evidence_net.evaluation.metrics import edge_magnitude
from evidence_net.stress_tests.forward import (
    AreaDownsample,
    BlurDownsample,
    NoisyBlurDownsample,
)

FEATURE_NAMES = (
    "mean",
    "std",
    "energy_low",
    "energy_mid",
    "energy_high",
    "edge_density",
)
N_FEATURES = len(FEATURE_NAMES)
FREQUENCY_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 1.0 / 8.0),
    (1.0 / 8.0, 1.0 / 2.0),
    (1.0 / 2.0, 1.0),
)
STD_EPSILON = 1e-6
POWER_EPSILON = 1e-12

REFERENCE_GROUP = "reference"
SHIFT_GROUP_NAMES = ("source", "severity", "degradation", "acquisition")
RARE_VALID_GROUP = "rare-valid"


class FamiliarityError(ValueError):
    """Raised for invalid familiarity configurations or diagnostics."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamiliarityConfig:
    """Validated familiarity-diagnostic configuration (familiarity-v1)."""

    version: str = "familiarity-v2"
    threshold: float = 2.0
    calibration_quantile: float = 0.9
    n_reference: int = 64
    n_per_shift: int = 32
    rare_valid_max_false_warning_rate: float = 0.5
    n_boot: int = 1000
    seed: int = 0

    def validate(self) -> None:
        if self.version not in ("familiarity-v1", "familiarity-v2"):
            raise FamiliarityError(
                f"version must be familiarity-v1 or familiarity-v2, got {self.version}"
            )
        if self.threshold <= 0.0:
            raise FamiliarityError(f"threshold must be > 0, got {self.threshold}")
        if not 0.0 < self.calibration_quantile < 1.0:
            raise FamiliarityError(
                f"calibration_quantile must be in (0, 1), got {self.calibration_quantile}"
            )
        if self.n_reference < 1:
            raise FamiliarityError(f"n_reference must be >= 1, got {self.n_reference}")
        if self.n_per_shift < 1:
            raise FamiliarityError(f"n_per_shift must be >= 1, got {self.n_per_shift}")
        if not 0.0 <= self.rare_valid_max_false_warning_rate <= 1.0:
            raise FamiliarityError(
                "rare_valid_max_false_warning_rate must be in [0, 1], "
                f"got {self.rare_valid_max_false_warning_rate}"
            )
        if self.n_boot < 1:
            raise FamiliarityError(f"n_boot must be >= 1, got {self.n_boot}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "threshold": self.threshold,
            "calibration_quantile": self.calibration_quantile,
            "n_reference": self.n_reference,
            "n_per_shift": self.n_per_shift,
            "rare_valid_max_false_warning_rate": self.rare_valid_max_false_warning_rate,
            "n_boot": self.n_boot,
            "seed": self.seed,
        }


def load_familiarity_config(path: Path) -> FamiliarityConfig:
    """Load and validate a familiarity config from YAML (familiarity-v1)."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise FamiliarityError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise FamiliarityError(f"config root must be a mapping: {path}")
    allowed = {
        "version",
        "threshold",
        "calibration_quantile",
        "n_reference",
        "n_per_shift",
        "rare_valid_max_false_warning_rate",
        "n_boot",
        "seed",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise FamiliarityError(f"unknown config keys: {sorted(unknown)}")

    def pick(key: str, default: Any) -> Any:
        return raw.get(key, default)

    def as_float(value: Any, key: str) -> float:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise FamiliarityError(f"{key} must be a number, got {type(value).__name__}")
        return float(value)

    def as_int(value: Any, key: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise FamiliarityError(f"{key} must be an int, got {type(value).__name__}")
        return value

    config = FamiliarityConfig(
        version=str(pick("version", "familiarity-v2")),
        threshold=as_float(pick("threshold", 2.0), "threshold"),
        calibration_quantile=as_float(pick("calibration_quantile", 0.9), "calibration_quantile"),
        n_reference=as_int(pick("n_reference", 64), "n_reference"),
        n_per_shift=as_int(pick("n_per_shift", 32), "n_per_shift"),
        rare_valid_max_false_warning_rate=as_float(
            pick("rare_valid_max_false_warning_rate", 0.5),
            "rare_valid_max_false_warning_rate",
        ),
        n_boot=as_int(pick("n_boot", 1000), "n_boot"),
        seed=as_int(pick("seed", 0), "seed"),
    )
    config.validate()
    return config


# ---------------------------------------------------------------------------
# Feature representation (familiarity-v1 section 1)
# ---------------------------------------------------------------------------


def _as_2d(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise FamiliarityError(f"image must be (H, W) or (1, H, W), got shape {array.shape}")
    return array.astype(np.float64, copy=False)


def _band_energy_fractions(array: np.ndarray) -> np.ndarray:
    """Relative radial power per band, fractions summing to 1 (guarded)."""
    centered = array - array.mean()
    power = np.abs(np.fft.rfft2(centered)) ** 2
    height, width = array.shape
    freq_y = np.fft.fftfreq(height)[:, None]
    freq_x = np.fft.rfftfreq(width)[None, :]
    radial = np.sqrt(freq_y**2 + freq_x**2)
    normalized = np.minimum(radial / 0.5, 1.0)
    fractions: list[float] = []
    for lo, hi in FREQUENCY_BANDS:
        mask = (normalized >= lo) & (normalized < hi)
        fractions.append(float(power[mask].mean()) if mask.any() else 0.0)
    total = float(np.sum(fractions))
    if total <= POWER_EPSILON:
        # A flat image has no power anywhere: neutral equal thirds, not NaN.
        return np.full(3, 1.0 / 3.0)
    return np.asarray(fractions, dtype=np.float64) / total


def feature_vector(image: np.ndarray) -> np.ndarray:
    """Frozen 6-component feature vector on the input grid (familiarity-v1)."""
    array = _as_2d(image)
    mean = float(array.mean())
    std = float(array.std())
    fractions = _band_energy_fractions(array)
    if fractions.shape[0] != 3:
        raise FamiliarityError("internal error: band energy fractions must have 3 bands")
    edge_density = float(edge_magnitude(array).mean())
    return np.asarray(
        [mean, std, fractions[0], fractions[1], fractions[2], edge_density],
        dtype=np.float64,
    )


def _normalized_grid(array: np.ndarray) -> np.ndarray:
    """Brightness-invariant z-scored grid (zero mean, unit std)."""
    std = float(array.std())
    if std <= STD_EPSILON:
        return np.zeros_like(array, dtype=np.float64)
    return (array - array.mean()) / std


def feature_vector_v2(image: np.ndarray) -> np.ndarray:
    """Brightness-invariant structure feature vector (familiarity-v2).

    Computes structure statistics on a z-scored (brightness-invariant) grid
    so that dark or bright inputs are not flagged merely for their global
    brightness — the Gate 8 systematic-suppression failure mode of v1. The
    band-energy fractions are already relative (brightness-invariant); the
    global pixel mean is deliberately excluded.
    """
    array = _as_2d(image)
    grid = _normalized_grid(array)
    fractions = _band_energy_fractions(array)
    if fractions.shape[0] != 3:
        raise FamiliarityError("internal error: band energy fractions must have 3 bands")
    edge_density = float(edge_magnitude(grid).mean())
    # Local structure: distribution of 16x16 patch standard deviations on the
    # normalized grid — captures texture grain independent of brightness.
    height, width = grid.shape
    patch = 16
    ph, pw = height // patch, width // patch
    patches = (
        grid[: ph * patch, : pw * patch]
        .reshape(ph, patch, pw, patch)
        .transpose(0, 2, 1, 3)
        .reshape(ph * pw, patch * patch)
    )
    local_std = patches.std(axis=1)
    return np.asarray(
        [
            float(grid.std()),
            fractions[0],
            fractions[2],
            edge_density,
            float(local_std.mean()),
            float(np.percentile(local_std, 90)),
            float(local_std.std()),
        ],
        dtype=np.float64,
    )


V2_FEATURE_NAMES = (
    "std",
    "energy_low",
    "energy_high",
    "edge_density",
    "local_std_mean",
    "local_std_p90",
    "local_std_spread",
)


# ---------------------------------------------------------------------------
# Reference-distance baseline (familiarity-v1 sections 2-3)
# ---------------------------------------------------------------------------


class ReferenceFamiliarity:
    """Per-feature mean/std over the reference population + RMS z-distance."""

    def __init__(self, reference_features: np.ndarray, *, threshold: float = 2.0) -> None:
        if reference_features.ndim != 2 or reference_features.shape[1] != N_FEATURES:
            raise FamiliarityError(
                f"reference_features must be (n, {N_FEATURES}), got {reference_features.shape}"
            )
        if reference_features.shape[0] < 1:
            raise FamiliarityError("reference population must have at least one image")
        self.mean = reference_features.mean(axis=0)
        self.std = np.where(
            reference_features.std(axis=0) < STD_EPSILON,
            STD_EPSILON,
            reference_features.std(axis=0),
        )
        self.threshold = threshold
        self.n_reference = reference_features.shape[0]

    @classmethod
    def fit(cls, images: Sequence[np.ndarray], *, threshold: float = 2.0) -> ReferenceFamiliarity:
        """Fit the reference population from development inputs."""
        if not images:
            raise FamiliarityError("cannot fit familiarity on an empty reference population")
        features = np.stack([feature_vector(image) for image in images])
        return cls(features, threshold=threshold)

    def distance(self, image: np.ndarray) -> float:
        """RMS standardized distance of one input to the reference population."""
        vector = feature_vector(image)
        z = (vector - self.mean) / self.std
        return float(np.sqrt(np.mean(z**2)))

    def is_familiar(self, distance: float) -> bool:
        """Familiar when ``distance <= threshold``."""
        return distance <= self.threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_reference": self.n_reference,
            "threshold": self.threshold,
            "feature_mean": self.mean.tolist(),
            "feature_std": self.std.tolist(),
        }


class ReferenceFamiliarityV2:
    """Familiarity-v2 reference: brightness-invariant features + calibrated threshold.

    v2 fixes the two Gate 8 failure modes of v1: (1) the feature vector is
    brightness-invariant (structure on a z-scored grid), so dark or bright
    inputs are not flagged merely for global brightness — rare **valid**
    structures are no longer systematically suppressed; and (2) the threshold
    is **calibrated from the reference population's own leave-one-out
    distance spread** (e.g. its 90th percentile) instead of a fixed global
    constant, which is the standard OOD practice and makes detection
    meaningful rather than all-or-nothing.
    """

    def __init__(self, reference_features: np.ndarray, *, threshold: float) -> None:
        if reference_features.ndim != 2 or reference_features.shape[1] != len(V2_FEATURE_NAMES):
            raise FamiliarityError(
                f"reference_features must be (n, {len(V2_FEATURE_NAMES)}), "
                f"got {reference_features.shape}"
            )
        if reference_features.shape[0] < 2:
            raise FamiliarityError(
                "reference population must have at least two images for LOO calibration"
            )
        self.mean = reference_features.mean(axis=0)
        self.std = np.where(
            reference_features.std(axis=0) < STD_EPSILON,
            STD_EPSILON,
            reference_features.std(axis=0),
        )
        self.threshold = threshold
        self.n_reference = reference_features.shape[0]

    @classmethod
    def fit(
        cls,
        images: Sequence[np.ndarray],
        *,
        calibration_quantile: float = 0.90,
    ) -> ReferenceFamiliarityV2:
        """Fit the v2 reference and calibrate the threshold from its own spread.

        The threshold is the ``calibration_quantile`` of the leave-one-out
        distances of the reference population itself — fit on development
        inputs only, never on probes. Standard OOD calibration.
        """
        if not images:
            raise FamiliarityError("cannot fit familiarity on an empty reference population")
        if not 0.0 < calibration_quantile < 1.0:
            raise FamiliarityError(
                f"calibration_quantile must be in (0, 1), got {calibration_quantile}"
            )
        features = np.stack([feature_vector_v2(image) for image in images])
        if len(features) < 2:
            raise FamiliarityError(
                "reference population must have at least two images for LOO calibration"
            )
        loo_distances: list[float] = []
        for index in range(len(features)):
            leave_out = np.delete(features, index, axis=0)
            mean_loo = leave_out.mean(axis=0)
            std_loo = np.where(
                leave_out.std(axis=0) < STD_EPSILON, STD_EPSILON, leave_out.std(axis=0)
            )
            z = (features[index] - mean_loo) / std_loo
            loo_distances.append(float(np.sqrt(np.mean(z**2))))
        threshold = float(np.percentile(loo_distances, 100.0 * calibration_quantile))
        return cls(features, threshold=threshold)

    def distance(self, image: np.ndarray) -> float:
        """RMS standardized v2 distance of one input to the reference population."""
        vector = feature_vector_v2(image)
        z = (vector - self.mean) / self.std
        return float(np.sqrt(np.mean(z**2)))

    def is_familiar(self, distance: float) -> bool:
        """Familiar when ``distance <= threshold``."""
        return distance <= self.threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_reference": self.n_reference,
            "threshold": self.threshold,
            "feature_version": "familiarity-v2",
            "feature_names": list(V2_FEATURE_NAMES),
            "feature_mean": self.mean.tolist(),
            "feature_std": self.std.tolist(),
        }


# ---------------------------------------------------------------------------
# Synthetic shift suite (familiarity-v1 section 4)
# ---------------------------------------------------------------------------


def _rare_valid_images(n: int, size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Thin lines, isolated points, and small defects on a dark background.

    These are *valid* structures (in-domain content) that the diagnostic must
    evaluate separately so it never systematically suppresses them (Gate 8).
    """
    images: list[np.ndarray] = []
    for i in range(n):
        image = np.full((size, size), 0.02)
        template = i % 3
        if template == 0:
            image[:, size // 2] = 0.9  # thin vertical line
        elif template == 1:
            image[size // 2, :] = 0.9  # thin horizontal line
        else:
            image[size // 4, size // 4] = 0.8  # isolated point + small defect
            image[3 * size // 4, 3 * size // 4 : 3 * size // 4 + 2] = 0.6
        image = image + rng.normal(0.0, 0.005, size=image.shape)
        images.append(np.clip(image, 0.0, 1.0))
    return images


def inject_rare_valid(images: Sequence[np.ndarray], *, seed: int = 0) -> list[np.ndarray]:
    """Inject rare **valid** structures into in-domain images (familiarity-v2).

    v1 evaluated rare-valid on synthetic dark-flat fixtures whose global
    brightness was far outside the reference — conflating 'dark' with
    'unfamiliar'. v2 injects thin lines, isolated points, and small defects
    into real (or real-like) inputs so the diagnostic is evaluated on
    structures that appear *within the in-domain brightness distribution*,
    which is the actual Gate 8 no-suppression question.
    """
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for index, image in enumerate(images):
        array = _as_2d(image).copy()
        height, width = array.shape
        template = index % 3
        if template == 0:
            column = width // 2 + (index // 3) % 4 - 2
            array[:, column] = np.clip(array[:, column] + 0.5, 0.0, 1.0)  # thin line
        elif template == 1:
            row = height // 2 + (index // 3) % 4 - 2
            array[row, :] = np.clip(array[row, :] + 0.5, 0.0, 1.0)  # thin line
        else:
            array[height // 4, width // 4] = 1.0  # isolated point
            array[3 * height // 4, 3 * width // 4 : 3 * width // 4 + 2] = 0.8  # defect
        array = array + rng.normal(0.0, 0.005, size=array.shape)
        out.append(np.clip(array, 0.0, 1.0))
    return out


def build_shift_suite(
    *,
    n_per_shift: int = 32,
    size: int = 64,
    seed: int = 0,
) -> dict[str, list[np.ndarray]]:
    """Seeded synthetic population and declared shift groups.

    Reference: blur+downsample of smooth noise fields (in-domain). Shifts:
    ``severity`` (stronger blur + noise), ``degradation`` (area downsample —
    a different family), ``acquisition`` (intensity scale/offset), ``source``
    (a different texture family), and ``rare-valid`` (thin lines / points /
    defects). All synthetic, software-only; never used in scientific reports.
    """
    rng = np.random.default_rng(seed)
    blur = BlurDownsample(blur_sigma=0.5)
    severe = NoisyBlurDownsample(blur_sigma=1.5, noise_sigma=0.03, seed=seed)
    area = AreaDownsample()

    def noise_field() -> np.ndarray:
        return rng.random((size, size))

    def smooth_field() -> np.ndarray:
        base = rng.random((size, size))
        blurred = base.copy()
        for _ in range(3):
            blurred = (blurred + np.roll(blurred, 1, axis=0) + np.roll(blurred, -1, axis=1)) / 3.0
        return np.clip(blurred, 0.0, 1.0)

    reference = [blur.apply(noise_field()) for _ in range(n_per_shift)]
    severity = [severe.apply(noise_field()) for _ in range(n_per_shift)]
    degradation = [area.apply(noise_field()) for _ in range(n_per_shift)]
    acquisition = [
        np.clip(0.8 * blur.apply(noise_field()) + 0.1, 0.0, 1.0) for _ in range(n_per_shift)
    ]
    source = [blur.apply(smooth_field()) for _ in range(n_per_shift)]
    rare_valid = _rare_valid_images(n_per_shift, size, rng)
    return {
        REFERENCE_GROUP: reference,
        "source": source,
        "severity": severity,
        "degradation": degradation,
        "acquisition": acquisition,
        RARE_VALID_GROUP: rare_valid,
    }


# ---------------------------------------------------------------------------
# Report (familiarity-v1 sections 3-4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamiliarityReport:
    """Per-group distances, shift detection, and the rare-valid gate input."""

    n_reference: int
    threshold: float
    per_group: dict[str, float]
    predictions: dict[str, bool]
    shift_groups: dict[str, dict[str, float]]
    rare_valid: dict[str, float]
    applicability: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_reference": self.n_reference,
            "threshold": self.threshold,
            "per_group": self.per_group,
            "predictions": self.predictions,
            "shift_groups": self.shift_groups,
            "rare_valid": self.rare_valid,
            "applicability": self.applicability,
        }


def build_familiarity_report(
    reference: ReferenceFamiliarity | ReferenceFamiliarityV2,
    probes: Mapping[str, Sequence[np.ndarray]],
    group_ids: Mapping[str, Sequence[str]],
    *,
    rare_valid_max_false_warning_rate: float = 0.5,
) -> FamiliarityReport:
    """Score all probes, aggregate shift detection, and report rare-valid behavior.

    ``probes`` maps a group name to a list of images; ``group_ids`` maps the
    group name to one id per image. The ``rare-valid`` group is evaluated
    separately: its false-warning rate (fraction flagged unfamiliar) is
    reported against the declared cap so systematic suppression is visible.
    """
    per_group: dict[str, float] = {}
    predictions: dict[str, bool] = {}
    shift_groups: dict[str, dict[str, float]] = {}
    for group_name, images in probes.items():
        ids = group_ids.get(group_name, [])
        if len(ids) != len(images):
            raise FamiliarityError(f"group '{group_name}': {len(images)} images but {len(ids)} ids")
        distances: list[float] = []
        unfamiliar = 0
        for image, group_id in zip(images, ids, strict=True):
            distance = reference.distance(image)
            familiar = reference.is_familiar(distance)
            per_group[group_id] = distance
            predictions[group_id] = familiar
            distances.append(distance)
            if not familiar:
                unfamiliar += 1
        shift_groups[group_name] = {
            "detection_rate": unfamiliar / len(images),
            "mean_distance": float(np.mean(distances)),
            "n": len(images),
        }
    rare = shift_groups.get(RARE_VALID_GROUP, {"detection_rate": 0.0, "n": 0})
    false_warning_rate = float(rare["detection_rate"])
    rare_valid = {
        "false_warning_rate": false_warning_rate,
        "n": int(rare["n"]),
        "exceeds_cap": false_warning_rate > rare_valid_max_false_warning_rate,
        "max_allowed": rare_valid_max_false_warning_rate,
    }
    return FamiliarityReport(
        n_reference=reference.n_reference,
        threshold=reference.threshold,
        per_group=per_group,
        predictions=predictions,
        shift_groups=shift_groups,
        rare_valid=rare_valid,
        applicability=(
            "Familiarity is valid only within the feature domain of the "
            "reference population. Unfamiliar is not wrong; familiar is not "
            "correct; neither certifies the restored output. Rare valid "
            "structures are evaluated separately so they are never "
            "systematically suppressed (Gate 8)."
        ),
    )
