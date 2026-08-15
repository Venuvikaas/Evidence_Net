"""Structural effect summaries for the detail proposal (Phase 4, box 4).

Per ``docs/proposal-contract.md`` section 5, the proposal's effect is
summarized by magnitude, edge content, multi-scale energy, and structural
change between Base and candidate. All functions operate on ``[0, 1]``
floating arrays on the output grid and are deterministic per image.
"""

from __future__ import annotations

import numpy as np

from evidence_net.evaluation.metrics import (
    FREQUENCY_BANDS,
    _radial_power_profile,
    edge_displacement,
    edge_magnitude,
    ssim,
)

_EPSILON = 1e-12


def proposal_magnitude(base: np.ndarray, proposal: np.ndarray) -> dict[str, float]:
    """Magnitude summary of the proposal relative to the Base output.

    Returns mean/max ``|d|`` and the relative mean magnitude
    ``mean(|d|) / (range(b) + eps)``.
    """
    b = np.asarray(base, dtype=np.float64)
    d = np.asarray(proposal, dtype=np.float64)
    abs_d = np.abs(d)
    base_range = float(b.max() - b.min())
    return {
        "mean_abs": float(abs_d.mean()),
        "max_abs": float(abs_d.max()),
        "relative_mean": float(abs_d.mean() / (base_range + _EPSILON)),
    }


def proposal_edge(proposal: np.ndarray) -> dict[str, float]:
    """Edge summary of the proposal (normalized Sobel magnitude of d)."""
    magnitude = edge_magnitude(np.asarray(proposal, dtype=np.float64))
    return {
        "edge_mean": float(magnitude.mean()),
        "edge_max": float(magnitude.max()),
        "edge_fraction": float((magnitude >= 0.5).mean()),
    }


def proposal_energy(proposal: np.ndarray) -> dict[str, float]:
    """Multi-scale energy summary: relative power of d per frequency band."""
    d = np.asarray(proposal, dtype=np.float64)
    power = np.abs(np.fft.rfft2(d)) ** 2
    radial = _radial_power_profile(power, d.shape)
    result: dict[str, float] = {}
    total = float(power.sum())
    for lo, hi in FREQUENCY_BANDS:
        # The radial profile clamps at 1.0 (Nyquist corner bin); the last
        # band is inclusive so the clamped bins are not dropped.
        mask = (radial >= lo) & (radial < hi) if hi < 1.0 else (radial >= lo) & (radial <= hi)
        band_power = float(power[mask].sum()) if mask.any() else 0.0
        result[f"band[{lo:.3f},{hi:.3f})"] = band_power / (total + _EPSILON)
    return result


def structural_change(base: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Structural change from Base to candidate.

    Returns edge displacement (px, capped) between base and candidate edges,
    the SSIM between them, and the mean edge-magnitude difference
    ``mean(|grad b| - |grad c|)``.
    """
    b = np.asarray(base, dtype=np.float64)
    c = np.asarray(candidate, dtype=np.float64)
    return {
        "edge_displacement_px": edge_displacement(b, c),
        "ssim": ssim(b, c),
        "edge_magnitude_delta": float((edge_magnitude(b) - edge_magnitude(c)).mean()),
    }


def proposal_effect_summary(
    base: np.ndarray, proposal: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    """All structural effect summaries for one image in a flat dict."""
    summary: dict[str, float] = {}
    summary.update(proposal_magnitude(base, proposal))
    summary.update(proposal_edge(proposal))
    for key, value in proposal_energy(proposal).items():
        summary[f"energy_{key}"] = value
    summary.update(structural_change(base, candidate))
    return summary


def connected_components(mask: np.ndarray) -> int:
    """Number of 4-connected components in a binary mask (0/False = background)."""
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return 0
    height, width = binary.shape
    labels = np.zeros(binary.shape, dtype=np.int32)
    component = 0
    # Union-find over row-major labels with 4-neighborhood connectivity.
    parent: dict[int, int] = {}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(height):
        for j in range(width):
            if not binary[i, j]:
                continue
            left = labels[i, j - 1] if j > 0 and binary[i, j - 1] else -1
            up = labels[i - 1, j] if i > 0 and binary[i - 1, j] else -1
            if left < 0 and up < 0:
                component += 1
                labels[i, j] = component
                parent[component] = component
            elif left >= 0:
                labels[i, j] = left
            elif up >= 0:
                labels[i, j] = up
            if left >= 0 and up >= 0 and left != up:
                union(left, up)
    roots = {find(label) for label in labels[binary]}
    return len(roots)
