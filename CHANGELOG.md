# Changelog

All notable changes to this project are recorded in this file. The governing
execution plan lives in `EXECUTION.md`.

## [Unreleased]

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
