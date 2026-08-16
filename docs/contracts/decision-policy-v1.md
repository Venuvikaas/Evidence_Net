# Contract: decision-policy-v1

- **Name:** `decision-policy-v1`
- **Version:** v1
- **Status:** draft (promotes to frozen after Research Gate 5)
- **Owner:** Lane A (benefit prediction and decision science)
- **Governed by:** `docs/proposal-contract.md`, `support-definition-v1`,
  `calibration-version-v1`, `docs/evaluation-protocol.md`

## Purpose

Fix the semantics of selective action (Phase 6): what **accept**, **attenuate**,
**reject**, and **abstain** mean, how thresholds are chosen, and — the
kill-switch rule of this phase — that **a rejected proposal never
automatically implies a resolved Base output**. The policy consumes the
calibrated benefit probability (`calibration-version-v1`) and the frozen
Base / bounded proposal (`proposal-contract.md`).

## 1. Actions

Each 16x16 patch (same grid as `support-definition-v1`) is assigned exactly
one action from the calibrated benefit probability `p`:

- **accept** (`p >= accept_threshold`): emit the ungated candidate
  `c = clamp(b + d, 0, 1)`.
- **attenuate** (`reject_threshold <= p < accept_threshold`): emit
  `b + g(p) * d` with a **documented monotone mapping** `g: [0, 1] -> [0, 1]`
  (v1: linear in `p`, clamped). The mapping is part of the policy version.
- **reject** (`p < reject_threshold`): emit the frozen Base `b`.

### 1.1 Unresolved mask (separate from the action)

- A patch is **unresolved** when the policy has no basis to trust **either**
  output there: v1 uses input-side difficulty (patch edge density above a
  declared threshold), because the EXP-004 failure catalogue concentrates
  harm in periodic high-edge-density regions.
- The unresolved mask is **orthogonal** to the accept/attenuate/reject
  action: a rejected patch may be unresolved, an accepted patch may be
  unresolved. **Rejection emits the Base but never certifies it.** The mask
  is reported separately and downstream consumers must not treat rejected
  regions as resolved.

## 2. Thresholds, costs, and critical regions

- `accept_threshold` and `reject_threshold` are chosen on **validation and
  calibration data only**, then **frozen** before any held-out evaluation
  (`decision-policy-v1` freeze). No test / held-out data ever selects a
  threshold (same isolation rule as `calibration-version-v1`).
- **Costs** (declared, v1): `false_accept` (accepting where the proposal
  harms) and `false_reject` (rejecting where the proposal would help) are
  weights used by the comparison report, never hidden.
- **Critical regions** (declared, v1): high edge-density patches; the report
  breaks out outcomes on critical vs non-critical regions separately.

## 3. Prohibited interpretations (Gate 5)

- Rejecting the proposal does not mean the Base is correct; unresolved
  regions carry that uncertainty explicitly.
- Attenuation is not calibrated probability; `g(p)` is a policy mapping with
  documented semantics, not a probability.
- Selective gating that improves a pre-declared endpoint is not a claim that
  restored detail existed (see `support-definition-v1`).
- The policy never runs on `Test_NoisyLR/` during development.

## 4. Implementation references

- Code: `src/evidence_net/decision/policy.py`
- Config: `configs/decision_policy/decision-policy-v1.yaml`
- Script: `scripts/measure_policy.py`
- Tests: `tests/calibration/test_policy.py` (fallback-uncertainty rule,
  action maps, coverage-risk reports, threshold freeze)
- Experiment: EXP-010 (Research Gate 5)

## 5. Change procedure

Changing action semantics, the attenuation mapping, threshold-selection
population, or the unresolved-mask rule requires `decision-policy-v2`, an
ADR, and review by Lane B (structural diagnostics consume the action maps)
and Lane C (review UI renders actions and the unresolved mask). The frozen
thresholds of v1 stay valid until all consumers migrate.
