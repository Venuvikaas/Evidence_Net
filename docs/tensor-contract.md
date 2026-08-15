# Tensor Contract v1 (initial)

- **Status:** Draft — confirmed against observed files in Phase 1.
- **Version:** v1
- **Governed by:** `EXECUTION.md` Phases 0 (contract) and 1 (verification).

This document fixes tensor conventions: dimensions, channel order, data type,
value range, masks, and spatial alignment. Code that loads or produces tensors
must conform to this contract; changes require a decision-log entry and version
increment.

## 1. Raw input preservation

- The raw corrupted tensor is preserved **exactly as read** from disk: dtype,
  value range, and dimensions are never silently cast or normalized at load
  time.
- Any derived view (normalized, gradients, frequency components) is a separate
  tensor explicitly labeled as derived, never written back over the raw
  tensor.

## 2. Dimensions and channel order

- Internal tensor layout is **channel-first**: `(C, H, W)` for 2D imagery.
- Grayscale inputs use `C = 1`. Multi-channel inputs follow the documented
  source order; channel order is recorded per dataset in the manifest.
- Batch tensors (training) use `(N, C, H, W)`.
- Sample-level spatial indices are `(row, col)` with row 0 at the top,
  consistent across input, Base, proposal, candidate, final, diagnostics, and
  masks.

## 3. Data type

- Internal tensors are `float32` unless a contract explicitly declares
  otherwise.
- Raw inputs keep their on-disk dtype in the preserved raw tensor; the
  internal working tensor is a float32 conversion of the preserved raw tensor.
- Masks are `uint8` with values `{0, 1}` (or `{0, 1, 2, 3}` for
  action maps: accept / attenuate / reject / abstain once defined).

## 4. Value range

- Working tensors default to `[0, 1]`; the raw tensor's true range is recorded
  in the manifest and preserved.
- Proposals (residuals) are signed and **bounded** by a configured amplitude
  `alpha`; the bound is part of the proposal contract (Phase 4).
- Metrics and diagnostics are never computed on tensors whose range is
  ambiguous; preprocessing version is recorded with every derived view.

## 5. Masks and spatial alignment

- The unresolved mask is a separate tensor on the same grid as the input;
  rejecting a proposal does **not** automatically clear the unresolved mask.
- All diagnostic maps (benefit, consistency, stability, familiarity, decision
  actions) are produced on the **same spatial grid** as the input, with no
  silent resampling between layers.
- Tiled inference (Phase 15) must preserve this alignment within a documented
  tolerance.

## 6. File naming for tensors

- Tensors saved as artifacts use `.npy` with names matching the artifact
  contract (`docs/run-and-artifact-contract.md`), e.g. `base.npy`,
  `proposal.npy`, `final.npy`, `unresolved.npy`, `decision_map.npy`.
- Each artifact file records its own metadata (dtype, shape, range) next to
  the tensor when stored.

## 7. Contract verification

- Phase 1 adds tests for dimensions, type, channel order, mask, and range
  preservation (`test(data): verify tensor contract`).
- A dry-run loader must be able to read every supported `Test_NoisyLR/` input
  under this contract without running final evaluation.
