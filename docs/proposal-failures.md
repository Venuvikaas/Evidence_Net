# Natural proposal failures (Phase 4, box 13)

This archive collects proposal outputs that hurt rather than help, taken from
the governed oracle study and the effect analysis. They are **natural**
failures — no adversarial inputs were constructed. They are archived so
Phase 5's benefit predictor and later stress tests have concrete negatives.

Run: `runs/proposal-effects-20260815-205856/` (12 validation groups, seed 0),
`runs/oracle-gate3-20260815-205601/`. Proposal checkpoint:
`checkpoints/train-proposal-gate3v2/best.pt` (frozen Base `train-base-gate2`,
amplitude 0.1, 12 epochs, composite + residual-fidelity loss).

## Where the proposal hurts

Across the 12-group sample the ungated candidate is never worse than the
Base on overall MAE (aggregate benefit +0.0024), but harm exists **per
structural region**:

| region | mean benefit | mean harm | oracle accept |
| --- | --- | --- | --- |
| edge_band | 0.00175 | 0.00007 | 0.791 |
| flat | 0.00263 | 0.00001 | 0.877 |
| periodic | 0.00153 | 0.00070 | 0.783 |
| all | 0.00239 | 0.00000 | 0.868 |

Harm concentrates in **periodic (high edge-density) regions**: the proposal
adds sharp detail that does not match the target in dense-texture areas.
The oracle patch gate accepts only 78% of periodic-region patches vs 88% of
flat patches, so a gate that cannot see ground truth must be conservative
there.

## Worst groups by regional harm

| sample | region harmed | regional harm | overall gain |
| --- | --- | --- | --- |
| 000893 | periodic | 0.00529 | +0.00479 |
| 002672 | periodic | 0.00135 | +0.00116 |
| 000250 | periodic | 0.00094 | +0.00021 |
| 002051 | periodic | 0.00079 | +0.00079 |
| 001977 | periodic | 0.00005 | +0.00174 |

Note `000893`: the candidate improves the overall image (MAE -0.0048) while
degrading the periodic regions (MAE +0.0053). Overall averages would hide
this; regional decomposition exposes it.

## Structural characterization

- **Flat regions** are the safest: benefit is largest and harm negligible;
  the proposal mostly denoises/refines smooth areas.
- **Periodic regions** are the riskiest: harm is 70x the flat-region harm.
- **Edge bands** are intermediate: modest benefit, low harm, oracle accepts
  79%.

## Consequence for design

- The Phase 5 benefit predictor and the decision policy must be **regional**,
  not image-global, and must treat periodic (high edge-density) regions as
  high-risk for acceptance.
- These five groups (and especially `000893`) are archived as stress-test
  cases for the Phase 5 gated pipeline: a gating policy that accepts
  `000893`'s periodic patches must be treated as over-confident.
