"""Deterministic benefit label behavior (support-definition-v1, Phase 5)."""

from __future__ import annotations

import numpy as np
import pytest

from evidence_net.benefit.labels import (
    OUTPUT_GRID,
    PATCH_GRID,
    PATCH_SIZE,
    BenefitLabelsError,
    benefit_fraction,
    label_samples,
    patch_benefit_labels,
    write_label_manifest,
)


def _grid(offset: float = 0.0, value: float = 0.5) -> np.ndarray:
    return np.full((OUTPUT_GRID, OUTPUT_GRID), value) + offset


def test_beneficial_when_candidate_closer() -> None:
    # Proposal moves every patch toward the target: all patches beneficial.
    base = _grid(value=0.4)
    target = _grid(value=0.5)
    proposal = np.full((OUTPUT_GRID, OUTPUT_GRID), 0.1)
    labels = patch_benefit_labels(base, proposal, target)
    assert labels.shape == (PATCH_GRID, PATCH_GRID)
    assert labels.dtype == np.uint8
    assert labels.sum() == PATCH_GRID * PATCH_GRID
    assert benefit_fraction(labels) == 1.0


def test_harmful_when_candidate_farther() -> None:
    # Proposal moves every patch away from the target: no patch beneficial.
    base = _grid(value=0.5)
    target = _grid(value=0.6)
    proposal = np.full((OUTPUT_GRID, OUTPUT_GRID), -0.1)
    labels = patch_benefit_labels(base, proposal, target)
    assert labels.sum() == 0
    assert benefit_fraction(labels) == 0.0


def test_ties_are_not_beneficial() -> None:
    # Candidate exactly equals Base: strict rule rejects the tie.
    base = _grid(value=0.5)
    target = _grid(value=0.5)
    proposal = np.zeros((OUTPUT_GRID, OUTPUT_GRID))
    labels = patch_benefit_labels(base, proposal, target)
    assert labels.sum() == 0


def test_region_beneficial_mixed_patch() -> None:
    # One patch is improved, the rest are harmed: exactly one label.
    # base = 0.5, target = 0.4: moving +0.1 pushes the candidate away from
    # the target (harmful); moving -0.1 pulls it to 0.4 (beneficial).
    base = _grid(value=0.5)
    target = _grid(value=0.4)
    proposal = np.full((OUTPUT_GRID, OUTPUT_GRID), 0.1)
    proposal[:PATCH_SIZE, :PATCH_SIZE] = -0.1
    labels = patch_benefit_labels(base, proposal, target)
    assert int(labels[0, 0]) == 1
    assert int(labels.sum()) == 1


def test_requires_output_grid() -> None:
    with pytest.raises(BenefitLabelsError, match="output grid"):
        patch_benefit_labels(np.zeros((32, 32)), np.zeros((32, 32)), np.zeros((32, 32)))


def test_requires_aligned_shapes() -> None:
    with pytest.raises(BenefitLabelsError, match="share the output grid"):
        patch_benefit_labels(
            np.zeros((OUTPUT_GRID, OUTPUT_GRID)),
            np.zeros((16, 16)),
            np.zeros((OUTPUT_GRID, OUTPUT_GRID)),
        )


def test_deterministic_and_versioned() -> None:
    base = _grid(value=0.5)
    target = _grid(value=0.5)
    rng = np.random.default_rng(0)
    proposal = rng.normal(0.0, 0.05, size=(OUTPUT_GRID, OUTPUT_GRID))
    first = patch_benefit_labels(base, proposal, target)
    second = patch_benefit_labels(base, proposal, target)
    assert np.array_equal(first, second)


def test_label_samples_alignment() -> None:
    rng = np.random.default_rng(1)
    bases = [rng.uniform(0.0, 1.0, size=(OUTPUT_GRID, OUTPUT_GRID)) for _ in range(3)]
    proposals = [rng.normal(0.0, 0.05, size=(OUTPUT_GRID, OUTPUT_GRID)) for _ in range(3)]
    targets = [np.clip(b + d, 0.0, 1.0) for b, d in zip(bases, proposals, strict=False)]
    samples = label_samples(["a", "b", "c"], bases, proposals, targets)
    assert [sample.sample_id for sample in samples] == ["a", "b", "c"]
    assert all(sample.labels.shape == (PATCH_GRID, PATCH_GRID) for sample in samples)

    with pytest.raises(BenefitLabelsError, match="aligned"):
        label_samples(["a", "b"], bases, proposals, targets)


def test_write_label_manifest(tmp_path) -> None:
    base = _grid(value=0.5)
    target = _grid(value=0.5)
    proposal = np.full((OUTPUT_GRID, OUTPUT_GRID), 0.1)
    samples = label_samples(["s1"], [base], [proposal], [target])
    path = write_label_manifest(tmp_path / "benefit-labels-v1.json", samples)
    assert path.is_file()
    payload = path.read_text(encoding="utf-8")
    assert "labels-v1" in payload
    assert '"sample_id": "s1"' in payload
