# Experiment Ledger (EXPERIMENTS.md)

Governed experiments are registered here **before** they run. Acceptance rules
are written before examining final test results. The format is defined in
`EXECUTION.md` (Part 1, Section 9). Every experiment produces a run bundle in
`runs/<run_id>/`.

```markdown
## EXP-XXX — Hypothesis
- Question:
- Primary metric:
- Secondary diagnostics:
- Baselines:
- Dataset manifest:
- Configs:
- Acceptance rule:
- Result:
- Confidence interval / uncertainty:
- Decision:
- Artifact path:
```

---

## EXP-001 — Official dataset validity for supervised restoration
- Question: Is the official `train/` dataset fit for supervised restoration
  (pairing, alignment, leakage, target meaning, rights)?
- Primary metric: pair integrity (unmatched + duplicated + ambiguous = 0),
  readable fraction = 1.0, exact and near-duplicate groups = 0.
- Secondary diagnostics: alignment phase distribution; target range;
  resolution ratio; train/test input compatibility; Test_NoisyLR isolation.
- Baselines: n/a (data-validity gate before model comparison).
- Dataset manifest: `official-train-source-v1.json`
  (hash c504b2dded0f3a04...) and `official-test-noisylr-source-v1.json`
  (hash aab75186e9a46982...).
- Configs: `scripts/audit_dataset.py` (align-sample 200, near-sample 0,
  fixed seeds 0/1/2/3); `scripts/build_splits.py --seed 0`.
- Acceptance rule (predeclared): **continue** if (1) 100% of files readable;
  (2) pairing clean (0 unmatched/duplicated/ambiguous); (3) 0 exact and 0
  near duplicates; (4) train and test inputs compatible in extension, shape,
  channels, dtype, and range family; (5) no Test_NoisyLR path in any
  development manifest; (6) target meaning and alignment uncertainty
  documented and recorded. **Repair / change scope** if any of (1)-(5)
  fails; **benchmark-only** if the pairing or target meaning cannot be
  trusted for supervised learning.
- Result: all acceptance conditions met. 3200/3200 clean pairs; 0 exact and
  0 near duplicates; 100% readable; compatibility confirmed; 400/400 test
  inputs dry-run readable; isolation enforced by tests. Alignment: no
  dominant 2x phase (offsets 0,0: 56 / 0,1: 61 / 1,0: 42 / 1,1: 41 of 200),
  mean best-offset MAE residual ≈ 0.067 — recorded as dataset-level target
  uncertainty in the train source manifest. Degradation labels absent;
  degradation-held-out group reserved with zero members.
- Confidence interval / uncertainty: deterministic audit; only
  alignment/degradation use fixed-seed sampling (n = 200 pairs); statistics
  grouped by pair, never by pixel.
- Decision: **continue** (ADR-005).
- Artifact path: `runs/audit-*/` (metrics, summary, alignment examples).

---

## EXP-002 — Classical baselines through the trusted evaluation harness
- Question: Do deterministic and classical restorers give usable comparison
  anchors on the validation split, and does the harness produce grouped,
  pixel-safe statistics?
- Primary metric: PSNR and MAE per source group, aggregated with a 95%
  seeded group bootstrap (groups = samples, never pixels).
- Secondary diagnostics: SSIM, edge displacement, structural error,
  frequency-band relative power differences.
- Baselines: deterministic bilinear 2x up-sampling;
  classical median-5x5 + bilinear 2x.
- Dataset manifest: `official-train-source-v1.json` +
  `dataset-splits-v1.json` (validation split only; Test_NoisyLR untouched).
- Configs: `scripts/evaluate_baselines.py --n-samples 8 --seed 0
  --split validation` (sample selection seeded; n_boot = 1000, seed 0).
- Acceptance rule (predeclared): **continue** if the harness reproduces
  per-group metrics with finite CIs and both baselines complete without
  error; use the deterministic anchor as the Phase 3 comparison floor.
- Result: harness green. Deterministic bilinear: PSNR 24.67 dB
  (CI 23.23–26.20), SSIM 0.572, MAE 0.043; classical median+bilinear:
  PSNR 24.41 dB (CI 21.80–26.92), SSIM 0.516, MAE 0.043. Edge displacement
  lower for the deterministic anchor (4.66 vs 9.69 px). Both baselines show
  large mid/high-band power deficits (classical −0.66 / −0.79 relative),
  consistent with the inputs being pre-degraded (Phase 1 alignment
  uncertainty), so low PSNR is expected before learned models.
- Confidence interval / uncertainty: group bootstrap over 8 validation
  groups; CIs reflect cross-sample spread, not pixel counts.
- Decision: **continue** — deterministic anchor accepted as the Phase 3
  floor; harness reusable for every later model (Phase 8/9 onward).
- Artifact path: `runs/baseline-eval-20260815-171136/` (comparison sheets,
  comparison-report.md, metrics.json).

---

## EXP-003 — Learned Base Reconstruction vs classical baselines
- Question: Is the learned Base Reconstruction independently useful against
  the deterministic and classical anchors, and is its behavior understood by
  structural region (Research Gate 2)?
- Primary metric: PSNR / SSIM / MAE on a seeded validation sample, with 95%
  group bootstraps; all models on identical paired groups.
- Secondary diagnostics: edge displacement, structural error, failure
  catalogue by region (edge band / periodic / flat).
- Baselines: deterministic bilinear 2x; classical median-5 + bilinear;
  direct-restoration CNN of equal capacity.
- Dataset manifest: `official-train-source-v1.json` +
  `dataset-splits-v1.json`; training on 256 train groups (seed 0),
  evaluation on 12 validation groups (seed 0).
- Configs: `configs/model/base-gate2.yaml`, `configs/model/direct-gate2.yaml`
  (12 epochs, batch 8, lr 1e-3, composite loss pixel 1.0 / structural 0.25 /
  edge 0.25 / frequency 0.1); `scripts/compare_restoration.py --n-samples 12`;
  `scripts/catalogue_failures.py --n-samples 12`.
- Acceptance rule (predeclared): **continue** if (1) the Base Reconstruction
  is not statistically worse than the deterministic anchor on PSNR/SSIM/MAE;
  (2) its learned refinement improves over the anchor on at least one primary
  metric; (3) failures are characterized by structural region rather than
  only averaged; (4) the harness exposes worst groups, not just means.
  **Redesign the objective / do not imply safety** if the Base is merely
  weaker or indistinguishable everywhere.
- Status: predeclared (results recorded on execution).
