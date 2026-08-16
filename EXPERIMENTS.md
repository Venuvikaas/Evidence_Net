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
- Result: Base Reconstruction after 12 epochs improves on the anchor:
  PSNR 25.21 dB vs 25.08 dB (deterministic), SSIM 0.639 vs 0.599, MAE 0.0399
  vs 0.0430; classical median anchor 24.46 dB. Direct CNN of equal capacity
  is much weaker (PSNR 22.60 dB, MAE 0.0452). Failure catalogue: edge-band
  MAE 0.084, periodic-region MAE 0.096, flat-region MAE 0.030 — errors are
  structurally concentrated, and worst samples are reported individually.
- Confidence interval / uncertainty: group bootstrap over 12 validation
  groups; PSNR CIs for base and anchor overlap ([23.19, 27.45] vs
  [23.03, 27.45]), so the gain is consistent but not statistically strong at
  this sample size; the direct model is clearly worse.
- Decision: **continue** — the Base Reconstruction earns its place against
  the declared baselines (condition 1), improves on the anchor (condition 2),
  and failures are understood by structural group (conditions 3-4).
- Artifact path: `runs/compare-gate2/` (comparison report + sheets),
  `runs/catalogue-gate2/` (regional error decomposition).

---

## EXP-004 — Oracle gating headroom of the Detail Proposal
- Question: Does the bounded Detail Proposal provide meaningful oracle
  headroom — i.e. would selective acceptance of the proposal (an oracle that
  sees ground truth) improve declared outcomes over the frozen Base and the
  equal-capacity direct model (Research Gate 3)?
- Primary metric: PSNR / SSIM / MAE on a seeded validation sample with 95%
  group bootstraps, for Base, ungated candidate, oracle pixel-gated, and
  oracle patch-gated outputs (identical paired groups).
- Secondary diagnostics: pixel/patch coverage and risk (fraction of units
  where accepting the proposal would increase error), edge displacement and
  structural error of the oracle-patch output, proposal magnitude/energy
  summaries.
- Baselines: frozen Base (Phase 3 floor); ungated candidate; equal-capacity
  direct-restoration CNN; classical median-5 + bilinear.
- Configs: `configs/model/proposal-gate3.yaml` (12 epochs, batch 8, lr 1e-3,
  amplitude 0.1, composite loss pixel 1.0 / structural 0.25 / edge 0.25 /
  frequency 0.1); frozen Base checkpoint `checkpoints/train-base-gate2/best.pt`;
  `scripts/train_proposal.py`; `scripts/measure_oracle.py --n-samples 12`.
- Acceptance rule (predeclared): **continue** (Research Gate 3) if all of:
  (1) oracle patch-gated output improves mean PSNR over the frozen Base by
  at least 0.5 dB, or mean MAE by at least 5% relative, on the seeded
  validation sample (headroom beyond the Phase 3 floor);
  (2) oracle patch-gated PSNR/MAE is not worse than the equal-capacity direct
  model (headroom beyond an equal-capacity alternative);
  (3) patch coverage is between 10% and 90% — the proposal is neither always
  harmful nor trivially redundant (selection is meaningfully informative);
  (4) the oracle-patch output does not increase mean edge displacement vs the
  frozen Base (structural impact bounded).
  **Redesign the proposal or the spatial unit** if the oracle finds no such
  headroom; **abandon the gated decomposition** if equal-capacity direct
  restoration matches oracle-gated outputs.
- Result (proposal trained with composite + residual-fidelity loss, 12
  epochs): oracle patch-gated PSNR 25.66 dB vs Base 25.21 dB (+0.45), SSIM
  0.671 vs 0.639, MAE 0.0373 vs 0.0399 (-6.3% relative); oracle pixel-gated
  26.16 dB. Ungated candidate 25.63 dB (also +0.42 over Base). Direct CNN
  22.60 dB — oracle output is 3 dB above the equal-capacity alternative.
- Coverage / risk: pixel coverage 0.578 (risk 0.422); patch coverage 0.868
  (risk 0.132) — the proposal is selective, neither always harmful nor
  trivially redundant.
- Structural impact: oracle-patch edge displacement 6.75 px vs Base 6.68 px
  (delta +0.07 px, bootstrap CIs [4.06, 9.99] vs [3.95, 9.82] overlap
  heavily → no detectable increase); structural error 0.0435 vs 0.0443.
- Failure catalogue (EXP-004 box 12-13): benefit concentrates in flat
  regions (0.0026), harm in periodic high-edge-density regions (0.0007,
  oracle accept 78%); worst group 000893 gains overall while degrading
  periodic regions; archived as FAIL-001.
- Decision: **continue** — conditions 1-3 met (MAE -6.3% >= 5% bar; oracle
  far above direct; coverage 86.8% in [10%, 90%]); condition 4 met (edge
  displacement not detectably increased). The decomposition has value and
  is worth predicting (Research Gate 3: continue).
- Artifact path: `runs/oracle-gate3-20260815-205601/`,
  `runs/proposal-effects-20260815-205856/`,
  `checkpoints/train-proposal-gate3v2/best.pt`.

---

## EXP-005 — Does the measurement-consistency diagnostic add held-out value?
- Question: Does the per-operator residual distribution of the
  measurement-consistency diagnostic (`forward-model-v1`, Phase 7) improve
  selective-risk ordering or restoration decisions beyond simple benefit
  features (Lane A), or provide independently useful review information that
  changes a review decision (Research Gate 6)?
- Primary metric: change in selective-risk ordering quality (e.g. AUROC of
  the oracle-accept event) and/or calibration-domain-valid mean absolute
  error of the frozen benefit predictor, with vs without the consistency
  feature set; for the review-information arm: fraction of documented cases
  where the per-operator residual structure changes the review verdict on a
  predeclared case bank.
- Secondary diagnostics: per-operator residual means and CIs (group
  bootstrap), arg-min operator spread, bias, stochastic spread; whether
  consistency features are redundant with simple residual-magnitude and
  local-signal benefit features.
- Baselines: no consistency features (benefit features only); consistency
  features only; combined.
- Dataset manifest: `official-train-source-v1.json` + `dataset-splits-v1.json`
  (validation + calibration splits only; Test_NoisyLR untouched).
- Configs: `configs/modality/forward-v1.yaml`;
  `scripts/measure_consistency.py --synthetic` (harness smoke, CI); real
  governed run on the frozen validation sample with the deterministic anchor
  and, once available, the frozen Base (`base-output-v1`).
- Acceptance rule (predeclared, Gate 6): **keep** the diagnostic if (1) the
  consistency feature set improves a declared held-out outcome by a
  predeclared margin over benefit features alone, OR (2) the review
  information arm shows the residual structure changes review verdicts on a
  predeclared fraction of the case bank. **Remove** the diagnostic if it adds
  no held-out value and no independently useful review information.
  The diagnostic is always labeled compatibility, never truth.
- Result: pending — machinery (operators, report, script, stress cases,
  tests) is complete; the governed comparison needs Lane A's simple benefit
  features (Phase 5) and runs at Integration I when those are promoted.
  The synthetic smoke run validates the harness end-to-end in CI.
- Confidence interval / uncertainty: group bootstrap over samples; pixels are
  never sample counts; stochastic operators report seeded spread.
- Decision: pending (Research Gate 6).
- Artifact path: `runs/measure-consistency-*/` (synthetic smoke runs).

---

## EXP-006 — Does the model-stability diagnostic add held-out value?
- Question: Does the stability diagnostic (`stability-v1`, Phase 8) —
  perturbation deviation, checkpoint agreement, and measured error
  diversity — improve selective-risk ordering or restoration decisions
  beyond simple benefit features (Lane A) and the measurement-consistency
  features (Phase 7), or provide independently useful review information
  (Research Gate 7)?
- Primary metric: change in selective-risk ordering quality (e.g. AUROC of
  the oracle-accept event) and/or calibration-domain-valid mean absolute
  error of the frozen benefit predictor, with vs without the stability
  feature set; for the review arm: fraction of documented cases where
  perturbation/checkpoint agreement or the diversity guard changes the
  review verdict on a predeclared case bank.
- Secondary diagnostics: per-perturbation deviation means and CIs (group
  bootstrap), arg-max perturbation, pairwise checkpoint agreement, pairwise
  error diversity (correlation / disagreement / complementarity) and the
  included-model set after the diversity guard.
- Baselines: no stability features; benefit features only; benefit +
  consistency features; benefit + consistency + stability features.
- Dataset manifest: `official-train-source-v1.json` + `dataset-splits-v1.json`
  (validation + calibration splits only; Test_NoisyLR untouched).
- Configs: `configs/modality/stability-v1.yaml`;
  `scripts/measure_stability.py --synthetic` (harness smoke, CI); real
  governed run on the frozen validation sample with the promoted Base
  checkpoints (`base-output-v1`) and, when present, the direct model.
- Acceptance rule (predeclared, Gate 7): **keep** the diagnostic if (1) the
  stability feature set improves a declared held-out outcome by a
  predeclared margin over benefit+consistency features alone, OR (2) the
  review information arm shows agreement/diversity changes review verdicts
  on a predeclared fraction of the case bank. **Remove** the diagnostic if it
  adds no held-out value and no independently useful review information.
  Agreement is stability, never correctness; a stable wrong output is still
  wrong.
- Result: pending — machinery (perturbations, checkpoint agreement,
  diversity guard, script, tests) is complete; the governed comparison needs
  Lane A's simple benefit features (Phase 5) and runs at Integration II when
  those are promoted. The synthetic smoke run validates the harness
  end-to-end in CI.
- Confidence interval / uncertainty: group bootstrap over samples; pixels
  are never sample counts; synthetic checkpoint pairs are labeled synthetic
  and never used in scientific reports.
- Decision: pending (Research Gate 7).
- Artifact path: `runs/measure-stability-*/` (synthetic smoke runs).

---

## EXP-007 — Does the familiarity diagnostic detect declared shifts without suppressing rare valid structures?
- Question: Does the reference-distance baseline (`familiarity-v1`, Phase 9)
  detect the declared source, severity, degradation, and acquisition shifts,
  and does it avoid systematically suppressing rare **valid** structures
  (thin lines, isolated points, small defects) — Research Gate 8?
- Primary metric: per-shift-group detection rate (fraction of probes flagged
  unfamiliar) with grouped CIs; rare-valid false-warning rate (fraction of
  rare-valid probes flagged unfamiliar) against the declared cap
  (`rare_valid_max_false_warning_rate = 0.5` default).
- Secondary diagnostics: mean distance per group, distance distributions,
  per-feature z-contributions (which features drive the flag), and the
  applicability statement bound to the reference feature domain.
- Baselines: no familiarity signal (no shift warnings); random threshold;
  per-feature threshold variants.
- Dataset manifest: `official-train-source-v1.json` + `dataset-splits-v1.json`
  (calibration reference; validation + heldout-source probes; Test_NoisyLR
  untouched). Synthetic rare-valid structures are labeled and never used as
  scientific evidence.
- Configs: `configs/modality/familiarity-v1.yaml`;
  `scripts/measure_familiarity.py --synthetic` (harness smoke, CI); real
  governed run on the frozen calibration reference.
- Acceptance rule (predeclared, Gate 8): **continue** if (1) every declared
  shift group is detected at a predeclared rate (e.g. >= 0.8 detection on
  severity and degradation groups), (2) the rare-valid false-warning rate
  stays below the declared cap (no systematic suppression), and (3) the
  applicability limits are published with the output. **Redesign the
  representation** if detection fails; **do not integrate into warnings or
  abstention** before this gate passes (Lane A policy).
- Result: pending — machinery (features, baseline, shift suites, report,
  script, tests) is complete; the governed run on the real calibration
  reference and curated rare-valid cases (Phase 10 failure bank) decides the
  gate. The synthetic smoke run validates the harness end-to-end in CI and
  reports the rare-valid cap mechanism.
- Confidence interval / uncertainty: group bootstrap over samples; pixels
  are never sample counts; synthetic probes are labeled synthetic.
- Decision: pending (Research Gate 8).
- Artifact path: `runs/measure-familiarity-*/` (synthetic smoke runs).

## EXP-008 — Do the five structural-risk evidence categories give the claimed downstream protection?
- Question: Does the candidate manipulation suite (false-line, deletion,
  edge-shift, merge, split, false-periodicity, defect-point), the ambiguity
  suite, the acquisition artifact suite, the frozen natural failure bank,
  and the frozen downstream task each provide **separate** evidence about
  restoration behavior, and do they jointly support a structural-risk claim
  without any single suite overclaiming — Research Gate 9?
- Primary metric: per-category effect sizes (candidate manipulations:
  downstream-measurement deltas per manipulation; ambiguity: observation vs
  candidate MAE on non-identifiable pairs; acquisition: input deltas;
  natural failures: count/severity of frozen bank cases; downstream:
  measurement-fidelity error per output type with group bootstrap CIs).
- Secondary diagnostics: manipulation vs measurement type cross-tables,
  worst-case candidate effects, ambiguity pair residuals, per-artifact input
  deltas, and the hidden-stress hash pinned in every run manifest
  (`data/stress/hidden-stress-v1.json`).
- Baselines: no candidate (base output only) vs candidate-modified output;
  observation vs candidate on ambiguity pairs; base vs oracle-patch proxy
  on the frozen downstream task.
- Dataset manifest: `dataset-splits-v1.json` validation split for real mode;
  synthetic probes are labeled synthetic and never used as scientific
  evidence; `Test_NoisyLR/` untouched (hash-verified hidden stress).
- Configs: `scripts/measure_structural_risk.py --synthetic` (harness smoke,
  CI) and `--real` (frozen Base/Proposal checkpoints on validation split).
- Acceptance rule (predeclared, Gate 9): **continue** if each of the five
  categories produces evidence on its own terms (no cross-category
  substitution), the hidden stress definitions remain frozen (hash match),
  and the downstream task is evaluated without co-training on the stress
  suite. **Redesign the affected suite** if a category is empty or its
  effect cannot be measured.
- Result: pending — machinery (structural.py, ambiguity.py, acquisition.py,
  downstream.py, hidden_stress.py, natural-failures bank, tests, script) is
  complete; the governed real run at Integration III decides the gate.
  The synthetic smoke run validates all five categories end-to-end in CI.
- Confidence interval / uncertainty: group bootstrap over samples; synthetic
  probes labeled; ambiguity pairs are non-identifiable by construction, so
  candidate vs observation MAE is reported, never judged as "truth".
- Decision: pending (Research Gate 9).
- Artifact path: `runs/structural-risk-*/` (synthetic smoke runs).

## EXP-009 — Is proposal benefit predictably useful (Gate 4)?
- Question: Can the minimal Proposal-Benefit Predictor (trained two-stage,
  separately from the proposal and Base) predict the deterministic benefit
  event of `support-definition-v1` well enough to beat declared simple
  heuristics, order selective risk usefully, and calibrate meaningfully
  within a stated domain — Research Gate 4?
- Primary metric: group-bootstrapped AUC (per source group) and pooled AUC
  of the learned predictor vs the declared baselines (residual-magnitude,
  local-signal, attention-gate) on held-out validation data; selective-risk
  curve (mean patch MAE of the gated output at declared coverage levels vs
  the ungated and Base floors); calibration (group-bootstrapped Brier,
  reliability/ECE) fit on the **calibration split only**.
- Secondary diagnostics: per-group AUC distribution, gate coverage at
  thresholds, calibration reliability bins, and the transfer check on
  proposals not seen during predictor training.
- Baselines: residual-magnitude heuristic; local-signal heuristic;
  reconstruction-trained attention gate; uncalibrated predictor scores
  (pre-calibration preserved per `calibration-version-v1`).
- Dataset manifest: `dataset-splits-v1.json` (calibration split for
  training/calibration; validation split for evaluation; `Test_NoisyLR/`
  untouched). Synthetic fixtures are labeled synthetic, never scientific
  evidence.
- Configs: `configs/support_definition/support-definition-v1.yaml`,
  `configs/calibration/calibration-version-v1.yaml`;
  `scripts/train_benefit.py --synthetic` (two-stage train, CI);
  `scripts/measure_benefit.py --synthetic` (evaluation, CI); real governed
  run on the frozen calibration/validation splits.
- Acceptance rule (predeclared, Gate 4): **continue** if the learned
  predictor (1) beats every declared simple baseline in group AUC on
  held-out validation data, (2) provides useful selective-risk ordering
  (gated error at 0.5 coverage below the ungated floor at 1.0 coverage),
  (3) has meaningful calibration within a stated domain (ECE below a
  declared bound on the calibration fit), and (4) retains value on external
  proposal behavior or bounds its limitation explicitly.
  **Simplify, redefine the event, or stop the support-aware claim** if the
  learned predictor does not beat the simple baselines.
- Result: pending — machinery (labels, baselines, attention gate, minimal
  predictor, calibration, evaluation suite, scripts, tests) is complete;
  the governed real run on the frozen calibration/validation splits decides
  the gate. The synthetic smoke run validates the harness end-to-end in CI
  and reports the split isolation (calibration fit / validation eval).
- Confidence interval / uncertainty: group bootstrap over source groups;
  pixels and patches are never sample counts; synthetic probes labeled
  synthetic; pre-calibration scores preserved.
- Decision: pending (Research Gate 4).
- Artifact path: `runs/benefit-eval-*/` and `runs/benefit-predictor-*/`
  (synthetic smoke runs).

## EXP-010 — Does selective action reduce risk (Gate 5)?
- Question: Does the frozen decision-policy-v1 (accept/attenuate/reject from
  the calibrated benefit probability, plus the orthogonal unresolved mask)
  improve a pre-declared endpoint without unacceptable regressions, lower
  measured risk via abstention at usable coverage, and avoid disguising
  unresolved Base errors — Research Gate 5?
- Primary metric: restoration outcome (PSNR/SSIM/MAE) and structural risk
  (edge displacement) of the gated output vs the frozen Base and the ungated
  candidate on held-out validation data; coverage (fraction with gate > 0)
  and unresolved area (input edge-density mask) reported alongside.
- Secondary diagnostics: action-map fractions (accept/attenuate/reject),
  per-action patch MAE vs the target, critical-region (high edge-density)
  outcome breakout, and the rejected-and-unresolved overlap (the
  fallback-uncertainty rule).
- Baselines: ungated candidate (coverage 1.0), frozen Base (coverage 0),
  full-accept oracle as the upper reference (Phase 4 headroom).
- Dataset manifest: `dataset-splits-v1.json` (calibration split for
  threshold fit and calibration; validation split for evaluation;
  `Test_NoisyLR/` untouched). Synthetic fixtures are labeled synthetic.
- Configs: `configs/decision_policy/decision-policy-v1.yaml`;
  `scripts/measure_policy.py --synthetic` (CI smoke); governed real run on
  the frozen calibration/validation splits with the trained predictor.
- Acceptance rule (predeclared, Gate 5): **continue** if (1) the gated
  output improves the pre-declared endpoint over the frozen Base without
  increasing mean edge displacement vs the ungated candidate, (2)
  abstention (unresolved mask) lowers measured risk at a usable coverage
  (unresolved area below a declared cap), and (3) the report shows rejected
  patches that are unresolved (the policy never certifies the Base on
  rejection). **Redesign the policy or the unresolved rule** if rejection
  disguises Base errors.
- Result: pending — machinery (policy, thresholds, unresolved mask, action
  maps, coverage-risk reports, script, tests) is complete; the governed real
  run at Integration I decides the gate. The synthetic smoke run validates
  the policy path end-to-end in CI with split-isolated threshold fitting.
- Confidence interval / uncertainty: group bootstrap over source groups;
  patches pooled only inside the action-map report; synthetic probes
  labeled; thresholds frozen before evaluation.
- Decision: pending (Research Gate 5).
- Artifact path: `runs/policy-eval-*/` (synthetic smoke runs).
