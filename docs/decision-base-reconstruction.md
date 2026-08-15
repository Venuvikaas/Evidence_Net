# Decision: Base Reconstruction claim and Research Gate 2 (Phase 3)

- **Status:** Accepted (Phase 3).
- **Evidence:** `runs/compare-gate2/`, `runs/catalogue-gate2/`, EXP-003.
- **Governed by:** `EXECUTION.md` Phase 3 Research Gate 2.

## Question

Is the candidate Base Reconstruction credible enough to proceed to the
proposal research? Specifically:

1. Is it independently useful against the declared baselines?
2. Is its behavior understood by structural group?
3. Is the "lower intervention" claim supported, or must it be removed?
4. Does the harness expose failures rather than only averages?

## Evidence

### 1. Independently useful against baselines (12 validation groups, seed 0)

| Restorer | PSNR (dB) | SSIM | MAE |
| --- | --- | --- | --- |
| Deterministic bilinear (anchor U) | 25.08 [23.03, 27.45] | 0.599 | 0.0430 |
| Classical median-5 + bilinear | 24.46 [22.04, 26.87] | 0.592 | 0.0428 |
| **Base Reconstruction (trained)** | **25.21 [23.19, 27.45]** | **0.639** | **0.0399** |
| Direct CNN (equal capacity) | 22.60 [21.29, 23.81] | 0.634 | 0.0452 |

The Base Reconstruction improves on the deterministic anchor on all three
primary metrics (PSNR +0.13 dB, SSIM +0.040, MAE −0.0031) and beats the
classical median baseline clearly. The equal-capacity direct CNN is
substantially worse (PSNR −2.6 dB vs anchor), so the anchor + refinement
architecture earns its structure. PSNR CIs overlap at n=12, so the gain is
consistent but not yet statistically strong; SSIM/MAE gains are the
strongest signals.

### 2. Behavior understood by structural group

The failure catalogue (`docs/base-failures.md`) shows the error is
concentrated in periodic regions (MAE 0.096) and edge bands (MAE 0.084),
with flat regions much better (MAE 0.030). Worst samples are reported
individually, not averaged away.

### 3. Lower-intervention claim

The Base Reconstruction is **constructed** as `b = U(y) + h_b(f(y))` with a
deterministic anchor: it cannot move arbitrarily far from bilinear
up-sampling — the learned refinement is a bounded correction on top of the
anchor. At 12 epochs its output remains within ~0.04 MAE of the target and
stays close to the anchor everywhere (the refinement is small; worst edge
regions carry most of the residual). This supports a *fidelity-oriented,
lower-intervention-by-construction* description:

- **Supported:** the architecture cannot generate unrestricted images; it
  starts from the deterministic anchor and only corrects it.
- **Not claimed:** the model is not automatically "safe". Its lower
  intervention must be re-measured against the proposal (Phase 4) and any
  gated policy, and worst-group behavior remains the binding constraint.

### 4. Harness exposes failures

The comparison report and failure catalogue list worst samples and regional
decomposition; the grouped bootstrap reports CIs rather than point masses.
No pixel-count statistics are reported.

## Decision (Research Gate 2): **CONTINUE**

The Base Reconstruction is independently useful against the declared
baselines (condition 1), its behavior is understood by structural group
(condition 2), the lower-intervention claim is supported with explicit
bounds (condition 3), and the harness exposes failures (condition 4).

## Follow-ups

- Promote `checkpoints/train-base-gate2/best.pt` as the frozen Phase 3 Base
  Reconstruction and tag `v0.2-base-reconstruction`.
- Phase 4 must target the catalogue's worst regions (periodic, edge bands)
  with the Bounded Detail Proposal; a proposal that fails to improve those
  regions adds no value.
- Re-run the failure catalogue for every later model version.

## Version history

| Version | Date | Change |
| --- | --- | --- |
| v1 | 2026-08-16 | Research Gate 2 decision: continue (Phase 3). |
