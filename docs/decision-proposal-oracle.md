# Research Gate 3 — Does the decomposition have value?

- Status: **continue**
- Date: 2026-08-16
- Gate: `EXECUTION.md` Phase 4 (Bounded Detail Proposal and Oracle Study)
- Evidence: EXP-004 (`runs/oracle-gate3-20260815-205601/`,
  `runs/proposal-effects-20260815-205856/`,
  `checkpoints/train-proposal-gate3v2/best.pt`)

## Decision

**Continue** with the gated decomposition: Base + bounded Detail Proposal +
oracle-informed selective acceptance. The predeclared acceptance rules of
EXP-004 are met, and the oracle finds headroom well beyond both the frozen
Base and the equal-capacity direct restoration.

## Evidence against the predeclared rule

12 validation groups, seed 0, group-bootstrap CIs.

| Condition | Result | Verdict |
| --- | --- | --- |
| (1) oracle patch PSNR >= Base + 0.5 dB **or** MAE >= 5% better | PSNR +0.45 dB; **MAE -6.3%** | ✅ (via MAE) |
| (2) oracle patch not worse than equal-capacity direct | 25.66 dB vs 22.60 dB | ✅ |
| (3) patch coverage in [10%, 90%] | 86.8% (risk 13.2%) | ✅ |
| (4) no increase in edge displacement vs Base | 6.75 vs 6.68 px, CIs overlap | ✅ (no detectable increase) |

The pixel oracle reaches 26.16 dB (+0.95 over Base), so finer gating
granularity holds even more headroom; the patch grid is retained because it
matches the Phase 5 benefit event.

## Why continue rather than redesign or abandon

- **Meaningful headroom exists**: the oracle can improve MAE by 6.3% (patch)
  and 0.95 dB PSNR (pixel) over the frozen Phase 3 floor, and sits 3 dB
  above the equal-capacity direct CNN. The decomposition is not redundant.
- **Selection is informative**: 87% patch acceptance means the oracle
  rejects 13% of proposals that would harm; the risk measure is
  non-degenerate, so a benefit predictor has a real signal to learn.
- **Failures are understood, not hidden**: harm concentrates in periodic
  high-edge-density regions (0.0007 vs 0.0001 flat); worst groups are
  archived (FAIL-001) and become stress tests for Phase 5.
- **The first trained proposal was nearly inert** (mean |d| ~ 0.002) because
  the frozen Base already minimized the composite candidate loss; adding
  residual fidelity to the objective (per product definition 10.3) produced
  a real residual (mean |d| ~ 0.011) and the headroom above. This is the
  documented redesign of the *objective*, not of the decomposition itself.

## Consequences

- Phase 5 proceeds: define the proposal-benefit event, build simple
  predictors first (residual magnitude, local signal, attention gate), and
  require calibration/selective-risk reports before any learned predictor is
  trusted.
- The decision policy must be **regional**: periodic regions are
  high-risk (78% oracle accept vs 88% flat); the Phase 5 predictor should
  receive per-region evidence.
- The oracle remains a study tool; it is never used at inference.

## Alternatives considered

- **Redesign proposal**: not needed at this gate — the objective fix
  (residual fidelity) already produced meaningful headroom. A redesign would
  be re-litigating a passing gate.
- **Change spatial unit**: the pixel oracle shows more headroom than the
  patch oracle (+0.95 vs +0.45 dB), but the patch unit matches the Phase 5
  benefit event and the release-gate requirement; revisit only if Phase 5
  shows the patch unit is unlearnable.
- **Abandon gated decomposition**: rejected — the direct model is 3 dB below
  the oracle output, so the decomposition clearly adds value over an
  equal-capacity alternative.
