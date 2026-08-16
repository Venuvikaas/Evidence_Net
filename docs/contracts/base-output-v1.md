# Contract: base-output-v1

- **Name:** `base-output-v1`
- **Version:** v1
- **Status:** frozen
- **Frozen at:** Phase 4 handoff (Research Gate 3: continue, ADR-007)
- **Owner:** ALL lanes
- **Governed by:** `docs/proposal-contract.md` section 1, `docs/tensor-contract.md`

## Purpose

Fix the interface of the frozen Base Reconstruction output that every lane
reads: the Phase 3 promoted model is the comparison floor for all later
scientific and product claims.

## Frozen fields

1. **Model.** Base Reconstruction `b = U(y) + h_b(f(y))`: deterministic
   bilinear 2x anchor plus a learned refinement, clamped to `[0, 1]`.
   Promoted checkpoint: `checkpoints/train-base-gate2/best.pt` (tag
   `v0.2-base-reconstruction`; sha256 in `docs/handoff/checkpoint-registry.md`).
2. **Output tensor.** `b` on the output grid (256x256 for the official
   dataset), single channel, `float32`, values in `[0, 1]`.
3. **Frozen semantics.** The Base is frozen (stop-gradient) inside the
   proposal wrapper; no later lane retrains or mutates it. It remains the
   comparison floor in every later phase and is re-evaluated through the same
   harness.
4. **Lower-intervention claim.** The deterministic-anchor construction bounds
   the refinement by design; the claim is stated with those bounds, never as
   automatic safety.
5. **Known behavior.** Structural failure catalogue is frozen:
   periodic-region MAE 0.096, edge-band 0.084, flat 0.030 (`docs/base-failures.md`).

## Implementation references

- Code: `src/evidence_net/models/base.py`, `src/evidence_net/models/validate.py`
- Configs: `configs/model/base-gate2.yaml`
- Docs: `docs/base-failures.md`, `docs/decision-base-reconstruction.md`
- Experiments: EXP-003, `runs/compare-gate2/`

## Change procedure

Replacing or retraining the Base requires `base-output-v2`, an ADR, a rerun
decision for EXP-003/EXP-004 consumers, and a re-freeze of proposal targets
that were generated from the old Base. All four lanes must be notified.
