# Contract: tensor-v1

- **Name:** `tensor-v1`
- **Version:** v1
- **Status:** frozen
- **Frozen at:** Phase 4 handoff (Research Gate 3: continue, ADR-007)
- **Owner:** ALL lanes
- **Governed by:** `docs/tensor-contract.md`

## Purpose

Fix tensor conventions so every lane reads and writes interchangeable
tensors on identical grids.

## Frozen fields

1. **Layout.** Channel-first `(C, H, W)`; batch tensors `(N, C, H, W)`;
   grayscale `C = 1`. Sample-level spatial indices are `(row, col)` with row 0
   at the top, consistent across input, Base, proposal, candidate, final,
   diagnostics, and masks.
2. **Output grid.** Restored outputs live on the output grid (256x256 for the
   official dataset). All diagnostic maps are produced on the same grid with
   no silent resampling between layers.
3. **Dtype.** Internal working tensors are `float32`; raw inputs keep their
   on-disk dtype in the preserved raw tensor. Masks are `uint8` `{0, 1}` (or
   `{0, 1, 2, 3}` for action maps once defined).
4. **Range.** Working image tensors are `[0, 1]` (clamped); the raw tensor's
   true range is recorded in the manifest. Proposals (residuals) are signed
   and bounded by amplitude `alpha` (see `proposal-output-v1`).
5. **Raw preservation.** The raw corrupted tensor is preserved exactly as
   read; derived views are separate tensors explicitly labeled as derived.
6. **Alignment.** Tiled inference must preserve grid alignment within a
   documented tolerance (Phase 15 parity tests).

## Implementation references

- Code: `src/evidence_net/data/loaders.py`, `src/evidence_net/models/base.py`,
  `src/evidence_net/models/proposal.py`, `src/evidence_net/evaluation/`
- Tests: `tests/unit/test_loaders.py`, `tests/unit/test_models.py`,
  `tests/unit/test_proposal.py`

## Change procedure

Dimension, dtype, grid, or alignment changes are cross-lane and require
`tensor-v2`, an ADR, a migration note, and affected-owner review. No lane may
silently change grids or dtype between layers.
