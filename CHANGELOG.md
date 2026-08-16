# Changelog

All notable changes to this project are recorded in this file. The governing
execution plan lives in `EXECUTION.md`.

## [Unreleased]

### Added (Phase 18 — Final Validation and Release)
- `scripts/run_final_inference.py`: frozen one-pass evaluation on all 400
  supported `Test_NoisyLR/` inputs, preserving original relative names,
  verifying one-output-per-input coverage and the output contract
  ((256,256) float32 in [0,1]), and recording provenance hashes for the
  source manifest, checkpoints, and outputs.
- `docs/release-report-v1.md`: frozen release report recording the
  candidate versions, final evaluation protocol and results, all gate
  decisions (ADR-009..015), and published failures/limitations.

### Decided (Research Gates 4–10)
- Gate 4 **simplify**: benefit prediction is at chance on the frozen event
  (all predictors pooled AUC 0.48–0.59; the event is the norm — 79.4% of
  validation patches beneficial); per-patch benefit claim not promoted.
  ADR-009, EXP-009.
- Gate 5 **continue (simplified)**: default-accept + unresolved abstention
  beats the frozen Base (PSNR 25.376 vs 24.888 dB, MAE 0.0382 vs 0.0408).
  ADR-010, EXP-010.
- Gate 6 **keep** (consistency), Gate 7 **keep** (stability). ADR-011/012.
- Gate 8 **NOT promoted**: familiarity detects 0% of declared shifts and
  flags 100% of rare-valid structures (cap 0.50 exceeded). ADR-013.
- Gate 9 **continue** (structural-risk). ADR-014.
- Gate 10 **pending participants** (human interpretation untested; protocol
  and capture ready). ADR-015, EXP-011.

### Added (Lane A, Phase 6 — Decision Policy, Gating, and Abstention)
- `decision-policy-v1` draft contract (`docs/contracts/decision-policy-v1.md`):
  accept / attenuate / reject semantics from the calibrated benefit
  probability, an **orthogonal unresolved mask** (input edge density, from
  the EXP-004 failure catalogue), and the kill-switch rule that rejection
  never certifies the Base. Thresholds fit on validation/calibration only
  and frozen before held-out evaluation.
- Policy implementation (`src/evidence_net/decision/policy.py`): validated
  `PolicyConfig`, linear attenuation mapping, per-patch action maps,
  unresolved-mask computation, threshold fitting with split isolation, and
  action-map / coverage-risk reports.
- `configs/decision_policy/decision-policy-v1.yaml`;
  `scripts/measure_policy.py` (synthetic smoke for CI; real mode at
  Integration I) writing run bundles with the frozen thresholds.
- Tests in `tests/calibration/test_policy.py` (11 tests): action assignment,
  attenuation gate, the rejected-and-unresolved coexistence rule, threshold
  split isolation, and coverage-risk reports.
- EXP-010 registered with a predeclared Gate 5 acceptance rule; CI gains the
  policy smoke step.

### Added (Lane A, Phase 5 — Proposal-Benefit Definition, Predictor, and Calibration)
- `support-definition-v1` draft contract (`docs/contracts/support-definition-v1.md`):
  the benefit event is a strict patch-level comparison (16x16 grid, ungated
  candidate patch MAE < Base patch MAE), deterministic, versioned `labels-v1`,
  population limited to development data (Test_NoisyLR never enters).
- `calibration-version-v1` draft contract
  (`docs/contracts/calibration-version-v1.md`): calibrated probabilities
  within a stated domain, fit on the **calibration split only** (kill-switch),
  pre-calibration scores always preserved.
- Deterministic label generation (`src/evidence_net/benefit/labels.py`),
  declared baselines (residual-magnitude, local-signal), a
  reconstruction-trained attention gate, and the minimal two-stage
  Proposal-Benefit Predictor (`benefit/predictors.py`).
- Calibration (`benefit/calibration.py`: Platt/temperature, Brier,
  reliability/ECE) and the separate evaluation suite
  (`benefit/evaluate.py`: per-group AUC, pooled AUC, selective-risk curves)
  with the statistical-unit discipline of the evaluation protocol.
- `scripts/train_benefit.py` (two-stage, calibration-split-only, synthetic
  smoke for CI) and `scripts/measure_benefit.py` (calibration-fit / eval-split
  split isolation, synthetic smoke for CI) writing run bundles.
- Tests in `tests/calibration/` (labels, predictors, calibration, evaluation,
  and split-isolation regression tests) — 35 tests.
- EXP-009 registered with a predeclared Gate 4 acceptance rule; CI gains the
  two Lane A smoke steps (flagged for lane-D review of `.github/workflows`).

### Added (Lane B, Phase 10 — Structural-Risk and Downstream Validation)
- `structural-risk-v1` draft contract (`docs/contracts/structural-risk-v1.md`)
  defining five **separate** threat-model evidence categories (candidate
  manipulation, ambiguity, acquisition, natural failure, downstream) and the
  hidden-test rule (definitions frozen, never trained on). Registered in the
  contracts index as a lane-B draft.
- Candidate manipulation suite (`src/evidence_net/stress_tests/structural.py`):
  false-line, deletion, edge-shift, merge, split, false-periodicity,
  defect-point with validated parameters.
- Ambiguity suite (`ambiguity.py`): clean-candidate pairs with
  near-identical observations (non-identifiable by construction).
- Acquisition artifacts (`acquisition.py`): pre-inference degradations
  (sensor noise, compression, hot pixels, downsampling).
- Frozen hidden stress (`data/stress/hidden-stress-v1.json`) with
  content-hash verification (`hidden_stress.py`), frozen natural failure
  bank (`data/failures/natural-failures-v1.json`) curated into
  `FAILURES.md`, and downstream task (`downstream.py`, `docs/downstream-validation.md`)
  evaluated without co-training on the stress suite.
- `scripts/measure_structural_risk.py` (synthetic smoke, CI-safe; real mode
  on frozen Base/Proposal validation split) writing run bundles with the
  hidden-stress hash pinned in the manifest.
- Analytical tests in `tests/numerical/` (test_structural.py, test_ambiguity.py,
  test_acquisition.py, test_downstream.py, test_stress_isolation.py)
  covering all five evidence categories plus training isolation.

### Added (Lane B, Phase 9 — Distribution Familiarity and Shift)
- `familiarity-v1` draft contract (`docs/contracts/familiarity-v1.md`):
  frozen 6-component feature representation on the input grid, reference
  population rule (development data only), RMS standardized reference
  distance, configurable threshold, applicability limits, and the separate
  evaluation of rare valid structures (no systematic suppression, Gate 8).
- Familiarity diagnostics (`src/evidence_net/stress_tests/familiarity.py`):
  feature extraction with epsilon guards (flat images never NaN), the
  reference-distance baseline (`ReferenceFamiliarity`), a seeded synthetic
  shift suite (source / severity / degradation / acquisition / rare-valid),
  and a report with per-group detection rates and the rare-valid
  false-warning rate against the declared cap.
- `scripts/measure_familiarity.py` (synthetic smoke, CI-safe; real mode fits
  the calibration split and probes validation / heldout-source / shifts /
  rare-valid) writing run bundles; `configs/modality/familiarity-v1.yaml`.
- Analytical tests in `tests/numerical/test_familiarity.py` (feature guards,
  in-distribution familiarity, severity monotonicity, threshold behavior,
  report determinism, rare-valid in-domain vs out-of-domain gate behavior).
- EXP-007 registered (Gate 8 shift-detection and rare-structure rule).

### Added (Lane B, Phase 8 — Model Stability)
- `stability-v1` draft contract (`docs/contracts/stability-v1.md`):
  stability = agreement under invertible perturbations, across same-
  architecture checkpoints, and measured error diversity before combining
  models; prohibited interpretations (agreement is never correctness,
  never a probability of truth, never calibration); promotes at Gate 7.
- Stability diagnostics (`src/evidence_net/stress_tests/stability.py`):
  invertible perturbation family (bounded shifts + flips) with output-grid
  inverses, perturbation deviation distribution with grouped bootstrap CIs,
  pairwise checkpoint agreement, and error-diversity metrics (correlation /
  disagreement / complementarity) with the `add_if_diverse` guard.
- `scripts/measure_stability.py` (seeded synthetic smoke, CI-safe; real mode
  uses the promoted Base checkpoints) writing run bundles;
  `configs/modality/stability-v1.yaml`.
- Analytical tests in `tests/numerical/test_stability.py` (identity
  perturbation, flip equivariance, subpixel shift sensitivity, checkpoint
  self-agreement, diversity metrics, guard behavior).
- EXP-006 registered (Gate 7 incremental-value question); governed
  comparison pending Lane A's simple benefit features (Integration II).

### Added (Lane B, Phase 7 — Measurement Consistency)
- `forward-model-v1` draft contract (`docs/contracts/forward-model-v1.md`):
  bounded operator family (bilinear / area / blur / noisy-blur), parameter
  bounds, operation order, seeded stochastic treatment, non-identifiability
  threat model; promotes to frozen at Research Gate 6.
- Bounded forward operators (`src/evidence_net/stress_tests/forward.py`) with
  construction-time bounds validation (misspecification raises
  `ForwardError`), config loading, and canonical non-identifiability cases
  (stripe and line-present/absent pairs).
- Measurement-consistency compatibility report
  (`src/evidence_net/stress_tests/consistency.py`): per-operator residual
  distribution (min / median / max, never minimum only) with grouped
  bootstrap CIs, per-image feature extraction for Lane A, and stochastic
  spread reporting.
- `scripts/measure_consistency.py` (synthetic smoke mode, CI-safe) writing
  run bundles; `configs/modality/forward-v1.yaml`.
- Analytical tests in `tests/numerical/` (exact pooling, constants,
  bounds/misspecification, seeded stochasticity, operation-order
  sensitivity, non-identifiability, report discipline).
- EXP-005 registered (Gate 6 incremental-value question); governed
  comparison pending Lane A's simple benefit features (Integration I).

### Added (Four-Developer Handoff, after Phase 4)
- Frozen handoff contracts in `docs/contracts/`: `dataset-v1`, `tensor-v1`,
  `metrics-v1`, `artifacts-v1`, `base-output-v1`, `proposal-output-v1`,
  `structural-summary-v1`, `oracle-report-v1`, and
  `error-and-optional-fields-v1`, with a registry README and contract-change
  procedure (ADR-008).
- Four-lane ownership and workflow: `CODEOWNERS` (lanes A/B/C/D), PR
  template naming consumed contract versions, `CONTRIBUTING.md`, and
  `docs/four-developer-workflow.md` (branch/PR rules, integration
  checkpoints I-V, promotion, current-work rule).
- Kill switches: `docs/kill-switches.md` (per-lane Gates 4-10, global
  process switches) mechanically enforced by `scripts/verify_handoff.py`
  (new CI step) and `tests/unit/test_handoff.py`.
- Handoff artifacts: checkpoint registry
  (`docs/handoff/checkpoint-registry.md`, sha256-pinned Base and Proposal
  checkpoints with reproduction commands) and fixture registry
  (`data/fixtures/manifest-v1.json`) with a synthetic
  `error-and-optional-fields-v1` example fixture.

## [0.5.0] - 2026-08-16

### Added (Phase 4)
- Bounded Detail Proposal (`d = alpha*tanh(h_d(y, b))`, |d| <= alpha) with
  the ungated candidate `c = b + d` and fusion rule `x = b + g*d`; the Base
  is frozen (stop-gradient) inside the wrapper.
- Target residual generation (`d* = x - stopgrad(b)`) and structural effect
  summaries (magnitude, edge, multi-scale energy, structural change,
  4-connected components).
- Ground-truth oracle gating at pixel and 16x16 patch granularity with
  coverage/risk and headroom reports (group-bootstrap CIs).
- Proposal training/comparison scripts and configs; the proposal objective
  includes residual fidelity after the first governed run showed the
  composite-only objective was locally flat (documented objective fix).
- EXP-004 oracle study: oracle patch MAE -6.3% vs Base, oracle PSNR 3 dB
  above the equal-capacity direct model, coverage 86.8%; Research Gate 3
  continue (ADR-007). Natural harmful proposals archived (FAIL-001).

## [0.4.0] - 2026-08-16

### Added (Phase 3)
- PyTorch training stack: structured config validation, reproducible trainer
  with checkpointing/resume/mixed-precision/seeded runs, numerical failure
  guards (NaN, exploding gradients, empty batches).
- Experiment provenance: run bundles with environment capture, config,
  training history, and checkpoint references.
- Base Reconstruction (`b = U(y) + h_b(f(y))`, deterministic anchor +
  learned refinement) and an equal-capacity direct-restoration CNN.
- Composite base loss with configurable pixel, structural, edge, and
  frequency terms (differentiable; sqrt-epsilon guard on edge magnitudes).
- Model path validation: output contract, gradient flow, checkpoint
  roundtrip, and tiled-parity inference for fully convolutional models.
- Training/comparison scripts (`scripts/train_base.py`,
  `scripts/compare_restoration.py`, `scripts/catalogue_failures.py`) and
  model configs under `configs/model/`.
- EXP-003 records the governed comparison; failure catalogue and Research
  Gate 2 decision doc (continue); ADR-006.

## [0.3.0] - 2026-08-15

### Added (Phase 2)
- Evaluation metric contracts (`docs/evaluation-protocol.md`) with PSNR,
  SSIM, MAE, edge displacement, structural error, and frequency-band
  diagnostics.
- Grouped statistics: seeded group bootstrap CIs that resample source
  groups, with guards rejecting pixel-level aggregation.
- Deterministic bilinear reference reconstruction and classical
  median+bilinear restoration baseline (`models/reference.py`).
- Baseline inference pipeline (`inference/baseline.py`) and restoration
  comparison reporting with dependency-free PNG comparison sheets
  (`reporting/comparison_report.py`, `scripts/evaluate_baselines.py`).
- Smoke pipeline extended through baseline evaluation and report
  generation; EXP-002 records the first harness results.

## [0.2.0] - 2026-08-15

### Added (Phase 1)
- Official local dataset path validation and resolution (execution-file
  parent or `TRAIN_DATA_DIR` / `TEST_NOISY_LR_DIR`).
- Frozen source manifests for `train/` (3200 pairs) and `Test_NoisyLR/`
  (400 inputs) with per-file hashes and target-alignment uncertainty.
- Pairing adapter, pair-integrity audit, alignment audit, duplicate
  detection, and dataset audit pipeline (`scripts/audit_dataset.py`).
- Deterministic grouped development splits (train / validation / calibration
  / heldout-source; heldout-degradation reserved) and the frozen
  `dataset-manifest-v1`.
- Raw-preserving `.npy` loader with tensor-contract validation and a
  dry-run loader for the isolated test inputs.
- Test_NoisyLR isolation tests and docs: data card, train/test structure,
  provenance, grouping-and-splits.

## [0.1.0] - 2026-08-15

### Added
- Repository skeleton with the target folder layout (Phase 0).
- Repository ignore rules, environment template, and pre-commit quality checks.
- Python project metadata and tooling (ruff, mypy, pytest, pre-commit).
- Governance ledgers: `DECISIONS.md`, `EXPERIMENTS.md`, `FAILURES.md`,
  `BACKLOG.md`.
- Initial scientific contracts: modality, dataset manifest, tensor, and
  run/artifact contracts.
- Product definition copied from the final idea document.
- Environment check script and initial smoke pipeline with run-bundle support.
- Core quality CI workflow (lint, format, type check, unit tests, smoke).
- Initial runbook (`README.md`) with setup and reproduction commands.
