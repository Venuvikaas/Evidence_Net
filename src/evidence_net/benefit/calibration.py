"""Score calibration for benefit probabilities (Phase 5, calibration-version-v1).

Candidate methods map raw scores (logits) to calibrated probabilities.
Calibration is fit on the **calibration split only**; the fit is a pure
function of calibration-split data, and pre-calibration scores are always
preserved. ``tests/calibration/test_split_isolation.py`` enforces that the
fit rejects non-calibration data and that the fitted mapping is
deterministic.

Methods in v1: Platt scaling (default), temperature scaling, and isotonic
binning (report-only). Uncertainty is reported with group bootstrap CIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from evidence_net.evaluation.statistics import grouped_bootstrap_ci

CALIBRATION_VERSION = "calibration-v1"

# Allowed calibration split labels (dataset-splits-v1.json).
CALIBRATION_SPLITS = ("calibration",)


class CalibrationError(ValueError):
    """Raised when calibration cannot be fit or applied."""


def _validate_calibration_data(
    scores: np.ndarray, labels: np.ndarray, *, split: str
) -> tuple[np.ndarray, np.ndarray]:
    if split not in CALIBRATION_SPLITS:
        raise CalibrationError(
            f"calibration may only be fit on {CALIBRATION_SPLITS}, got split={split!r}"
        )
    if scores.shape != labels.shape:
        raise CalibrationError(
            f"scores and labels must be aligned, got {scores.shape} vs {labels.shape}"
        )
    flat_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    flat_labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(flat_scores)):
        raise CalibrationError("calibration scores must be finite")
    if not np.all((flat_labels == 0.0) | (flat_labels == 1.0)):
        raise CalibrationError("calibration labels must be binary {0, 1}")
    return flat_scores, flat_labels


def _platt_params(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit logistic ``p = sigmoid(a * s + b)`` by Newton-Raphson on logits.

    Deterministic (no RNG); regularized lightly so degenerate splits cannot
    produce infinite coefficients.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    a, b = 1.0, 0.0
    for _ in range(50):
        z = a * s + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        gradient_a = float(np.sum((p - y) * s)) + 1e-6 * a
        gradient_b = float(np.sum(p - y)) + 1e-6 * b
        weight = p * (1.0 - p)
        hessian_a = float(np.sum(weight * s * s)) + 1e-6
        hessian_b = float(np.sum(weight)) + 1e-6
        step_a = gradient_a / hessian_a
        step_b = gradient_b / hessian_b
        a -= step_a
        b -= step_b
        if abs(step_a) < 1e-9 and abs(step_b) < 1e-9:
            break
    return a, b


@dataclass(frozen=True)
class CalibratedMapping:
    """A fitted calibration mapping (pure function of calibration data)."""

    method: str
    version: str
    split: str
    n_calibration: int
    params: dict[str, float] = field(default_factory=dict)

    def apply(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to probabilities in [0, 1]."""
        s = np.asarray(scores, dtype=np.float64)
        if self.method == "platt":
            return 1.0 / (
                1.0 + np.exp(-np.clip(self.params["a"] * s + self.params["b"], -30.0, 30.0))
            )
        if self.method == "temperature":
            return 1.0 / (1.0 + np.exp(-np.clip(s / self.params["temperature"], -30.0, 30.0)))
        raise CalibrationError(f"cannot apply unknown method {self.method!r}")

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "version": self.version,
            "split": self.split,
            "n_calibration": self.n_calibration,
            "params": self.params,
        }


def fit_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    split: str,
    method: str = "platt",
) -> CalibratedMapping:
    """Fit a calibration mapping on calibration-split data only."""
    flat_scores, flat_labels = _validate_calibration_data(scores, labels, split=split)
    if method == "platt":
        a, b = _platt_params(flat_scores, flat_labels)
        return CalibratedMapping(
            method="platt",
            version=CALIBRATION_VERSION,
            split=split,
            n_calibration=int(flat_labels.size),
            params={"a": a, "b": b},
        )
    if method == "temperature":
        temperature = float(np.std(flat_scores) + 1e-6)
        return CalibratedMapping(
            method="temperature",
            version=CALIBRATION_VERSION,
            split=split,
            n_calibration=int(flat_labels.size),
            params={"temperature": temperature},
        )
    raise CalibrationError(f"unknown calibration method {method!r}")


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between calibrated probabilities and binary labels."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.shape != y.shape:
        raise CalibrationError("probabilities and labels must be aligned")
    return float(np.mean((p - y) ** 2))


def reliability_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reliability curve: bin centers, mean prediction, mean label, bin counts."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.shape != y.shape:
        raise CalibrationError("probabilities and labels must be aligned")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    mean_pred = np.zeros(n_bins, dtype=np.float64)
    mean_label = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for index in range(n_bins):
        mask = (p >= edges[index]) & (p < edges[index + 1])
        if index == n_bins - 1:
            mask = (p >= edges[index]) & (p <= edges[index + 1])
        counts[index] = int(mask.sum())
        if counts[index] > 0:
            mean_pred[index] = float(p[mask].mean())
            mean_label[index] = float(y[mask].mean())
    return centers, mean_pred, mean_label, counts


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Weighted mean |mean prediction - mean label| over bins."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    _centers, mean_pred, mean_label, counts = reliability_curve(p, y, n_bins=n_bins)
    total = int(counts.sum())
    if total == 0:
        return 0.0
    return float(np.sum(counts * np.abs(mean_pred - mean_label)) / total)


def grouped_brier(
    per_group: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Group bootstrap of Brier score with per-group (probs, labels) pairs."""
    if not per_group:
        raise CalibrationError("grouped_brier requires at least one group")
    group_scores: dict[str, float] = {}
    for group_id, (probabilities, labels) in per_group.items():
        group_scores[group_id] = brier_score(probabilities, labels)
    return grouped_bootstrap_ci(group_scores, n_boot=n_boot, seed=seed).as_dict()
