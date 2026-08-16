# Contract: proposal-output-v1

- **Name:** `proposal-output-v1`
- **Version:** v1
- **Status:** frozen
- **Frozen at:** Phase 4 handoff (Research Gate 3: continue, ADR-007)
- **Owner:** ALL lanes
- **Governed by:** `docs/proposal-contract.md`

## Purpose

Fix the tensor interface of the bounded Detail Proposal and its fusion rule,
which all post-handoff lanes consume.

## Frozen fields

1. **Parameterization.** `d = alpha * tanh(h_d(f(y), b))` with
   `|d| <= alpha` elementwise; `alpha` is a model configuration value
   (`model.amplitude`, validated `> 0`). The Base is frozen inside the
   wrapper (stop-gradient).
2. **Tensors.** Base `b`, proposal `d`, ungated candidate `c`, and gated
   reconstruction `x_hat` live on the output grid (256x256 for the official
   dataset), single channel, `float32`. Image tensors `b`, `c`, `x_hat` are
   clamped to `[0, 1]`; `d` is a signed residual bounded by `alpha`.
3. **Fusion rule.** `c = b + d` (ungated candidate, gate `g = 1`);
   `x_hat = b + g * d` (gated, `g in [0, 1]`). Identities are tested:
   `g = 0` returns exactly `b`; `g = 1` returns exactly `c`.
4. **Promoted checkpoint.** `checkpoints/train-proposal-gate3v2/best.pt`
   (tag `v0.3-proposal-oracle`; sha256 in
   `docs/handoff/checkpoint-registry.md`).
5. **What it does not claim.** Proposed detail is the learned prior's best
   residual under the training objective, not proof the detail physically
   existed. Benefit is measured on declared outcomes, never visual appeal.
6. **Oracle is a study tool only.** Oracle decisions see ground truth and are
   never used at inference; Phase 5 (lane A) estimates the probability the
   oracle would accept.

## Implementation references

- Code: `src/evidence_net/models/proposal.py`, `src/evidence_net/proposal/targets.py`
- Configs: `configs/model/proposal-gate3.yaml`
- Tests: `tests/unit/test_proposal.py`
- Experiments: EXP-004, `runs/oracle-gate3-20260815-205601/`

## Change procedure

Changing the parameterization, amplitude semantics, or fusion rule requires
`proposal-output-v2`, an ADR, and re-running the Phase 4 oracle study. Lane A
(benefit prediction) and lane C (inference/UI) must review the migration.
