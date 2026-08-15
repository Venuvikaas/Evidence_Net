# Changelog

All notable changes to this project are recorded in this file. The governing
execution plan lives in `EXECUTION.md`.

## [Unreleased]

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
