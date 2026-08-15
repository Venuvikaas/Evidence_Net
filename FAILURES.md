# Failure Archive (FAILURES.md)

Negative results are project assets. Do not delete failed artifacts that
explain later design decisions. Categories follow `EXECUTION.md` (Part 1,
Section 10).

Tracked categories:

- Failed model configurations.
- Data-quality incidents.
- Misleading metrics.
- Natural hallucination examples.
- Calibration failures.
- Operator misspecification cases.
- UI interpretation failures.
- Deployment parity failures.

```markdown
## FAIL-XXX — Short title
- Category:
- Context:
- What was tried:
- Observed result:
- Root cause hypothesis:
- Consequence for design:
- Artifact path:
```

---

_Empty._
## FAIL-001 — Proposal adds spurious detail in periodic regions
- Category: Natural hallucination examples.
- Context: Phase 4 oracle study and effect analysis
  (`runs/proposal-effects-20260815-205856/`, 12 validation groups, seed 0).
- What was tried: bounded Detail Proposal (amplitude 0.1, 12 epochs,
  composite + residual-fidelity loss) on top of the frozen Base
  (`checkpoints/train-proposal-gate3v2/best.pt`).
- Observed result: the ungated candidate never degrades overall MAE on the
  sample, but harm concentrates in periodic (high edge-density) regions:
  mean harm 0.00070 vs 0.00007 in edge bands and 0.00001 in flat regions.
  Worst case `000893`: overall MAE improves -0.0048 while periodic-region
  MAE degrades +0.0053. Oracle accepts only 78% of periodic patches vs 88%
  of flat patches.
- Root cause hypothesis: dense-texture regions are hardest to reconstruct,
  so the learned residual there reflects training noise more than signal;
  sharpening detail in those regions does not match the target.
- Consequence for design: Phase 5 benefit prediction and the decision policy
  must be regional; periodic regions are high-risk for acceptance. `000893`,
  `002672`, `000250`, `002051`, `001977` are archived stress cases.
- Artifact path: `docs/proposal-failures.md`, `runs/proposal-effects-20260815-205856/`.
