# Contract: error-and-optional-fields-v1

- **Name:** `error-and-optional-fields-v1`
- **Version:** v1
- **Status:** frozen
- **Frozen at:** Phase 4 handoff (Research Gate 3: continue, ADR-007)
- **Owner:** ALL lanes
- **Governed by:** `docs/run-and-artifact-contract.md` section 5,
  `docs/tensor-contract.md`

## Purpose

Fix how post-handoff lanes add **optional** fields (Benefit, diagnostics,
Decision, Unresolved) and how artifacts behave when a field is **absent** or
an error occurs. This is the forward-compatibility contract that lets lanes
A, B, and C land independently without breaking the frozen Phase 4 slice.

## Frozen fields

1. **Backward compatibility.** The Phase 4 vertical slice is complete without
   any optional field. Every consumer must render, evaluate, and report when
   optional fields are absent.
2. **Optional fields.** Allowed optional artifact keys (when their contract
   is promoted): `proposal_benefit.npy`, `measurement_consistency.npy`,
   `model_stability.npy`, `distribution_familiarity.npy`, `decision_map.npy`,
   `unresolved.npy`. Each optional field names its own contract version in
   provenance (e.g. `"support_definition": "support-definition-v1"`); absent
   optional fields are recorded as `"not-defined"`, never omitted.
3. **Field metadata.** Every optional tensor carries dtype, shape, range, and
   hash metadata beside the tensor, on the same grid as the input
   (`tensor-v1`).
4. **Errors.** Error payloads are structured JSON with `error_code`,
   `message`, and `details`; they never expose paths, secrets, or raw
   tensors. Failed optional-field computation leaves the field absent
   (`"not-defined"`) rather than corrupting the run bundle or the frozen
   fields.
5. **No silent version mixing.** A report may combine fields from different
   contract versions only if every field's version is recorded.
6. **Synthetic fixtures.** Optional-field and error behavior is exercised by
   synthetic, software-only fixtures (see `data/fixtures/manifest-v1.json`);
   no scientific report may use synthetic software-only fixtures.

## Implementation references

- Example payload: `data/fixtures/error-and-optional-fields-v1-example.json`
- Tests: `tests/unit/test_handoff.py`
- Future consumers: lane C (Phase 11 unified inference and optional fields),
  lane D (Phase 15-17 parity and monitoring)

## Change procedure

Adding a new optional field requires a new named contract version (e.g.
`proposal-benefit-v1`) plus an ADR; it must remain optional and backward
compatible. Changing the error payload schema requires
`error-and-optional-fields-v2`, an ADR, and C/D review.
