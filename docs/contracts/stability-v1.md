# Contract: stability-v1

- **Name:** `stability-v1`
- **Version:** v1
- **Status:** draft (promotes to frozen after Research Gate 7)
- **Owner:** Lane B (diagnostics); reviewed by A and C at promotion
- **Governed by:** `docs/modality-contract.md`, `docs/tensor-contract.md`,
  `docs/evaluation-protocol.md`

## Purpose

Fix what the model-stability diagnostic measures (Phase 8): **agreement**.
A restored output is called stable when it changes little under invertible
input perturbations (after inverting the output back) and when checkpoints of
the same architecture agree. Stability is a review diagnostic — agreement is
**never** correctness, never a probability of truth, and never calibration.

## 1. What stability measures

### 1.1 Invertible perturbation stability

- A small family of **invertible spatial perturbations** is applied to the
  degraded input on the input grid (128x128 for the official dataset); the
  model runs; the output on the output grid (256x256) is inverted back with
  the perturbation's known inverse.
- Frozen family for v1: pixel shifts `(dy, dx)` with `|dy|, |dx| <= 1` on the
  input grid (inverse: opposite shift scaled by 2 on the output grid) and
  horizontal / vertical flips (self-inverse). The zero shift `(0, 0)` is the
  identity and must measure zero deviation (sanity check).
- Per-group deviation is the MAE between the unperturbed output and the
  inverted perturbed output, aggregated with the grouped bootstrap CI
  (`metrics-v1` discipline; pixels are never sample counts).
- The report shows the deviation distribution across the perturbation family
  (mean / max and the arg-max perturbation), never a single friendly
  perturbation.

### 1.2 Checkpoint stability

- Checkpoints compared must be snapshots of the **same architecture and
  contract** (e.g. `best.pt` vs `last.pt` of one training run).
- Per-group agreement is the MAE between the two checkpoints' outputs on
  identical inputs, aggregated with grouped bootstrap CIs. The pairwise
  agreement matrix is reported.
- Synthetic checkpoint pairs (same architecture, small weight perturbation)
  are allowed for tests and smoke runs but must be labeled synthetic; no
  scientific report may use synthetic pairs.

### 1.3 Diverse-model comparison (only with measured error diversity)

- A model may join a comparison set only when its **error diversity** versus
  the existing set is measured and exceeds the declared threshold
  (`min_diversity_threshold`).
- Error diversity metrics (per-pixel error maps vs the same reference):
  pairwise error correlation, disagreement rate (errors of opposite sign),
  and complementarity (fraction of pixels where the two models make
  different-magnitude mistakes).
- Diversity is never accuracy: a diverse model may still be wrong.

## 2. Prohibited interpretations (Gate 7)

- Agreement is not correctness; a stable wrong output is still wrong.
- Agreement across checkpoints is not a probability that the output is true.
- Stability does not calibrate any score and does not certify restored
  detail physically existed.
- A diverse ensemble is not automatically more reliable; its incremental
  value must be measured separately (EXP-006).

## 3. Implementation references

- Code: `src/evidence_net/stress_tests/stability.py`
- Config: `configs/modality/stability-v1.yaml`
- Script: `scripts/measure_stability.py`
- Tests: `tests/numerical/test_stability.py`
- Experiment: EXP-006 (incremental-value question, Research Gate 7)

## 4. Change procedure

Changing the perturbation family, checkpoint semantics, or diversity rule
requires `stability-v2`, an ADR, and review by lanes A (benefit/decision
consume stability features) and C (review UI renders the diagnostic).
Unproven stability sources stay disabled by default (Integration II).
