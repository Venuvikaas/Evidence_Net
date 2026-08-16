# Changelog

All notable changes to this project are recorded in this file. The governing
execution plan lives in `EXECUTION.md`.

## [Unreleased]

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
