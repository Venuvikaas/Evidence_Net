# Run and Artifact Contract v1

- **Status:** Accepted (Phase 0).
- **Version:** v1
- **Governed by:** `EXECUTION.md` Part 1 Section 3 (required experiment
  bundle) and Part 2 (repository layout).

## 1. Run bundle

Every governed run (smoke, experiment, or evaluation) produces a run bundle
under `runs/<run_id>/`:

```text
runs/<run_id>/
  config.yaml              # configuration used for the run (YAML, sorted keys)
  manifest.json            # data manifest reference + run metadata (JSON)
  environment.txt          # python, platform, package versions
  metrics.json             # primary metrics and diagnostics (JSON)
  summary.md               # human-readable conclusion
  artifacts/               # tensors, images, reports produced by the run
  logs/                    # run logs (no raw tensors)
  checkpoint-or-reference.txt  # checkpoint path or reference input, or "no-checkpoint"
```

A result does not count if its configuration, data manifest, seed policy, code
commit, and output artifacts cannot be recovered.

## 2. Run IDs

- Format: `<kind>-<YYYYMMDD>-<HHMMSS>` (UTC), e.g. `smoke-20260815-213000`,
  `exp-base-vs-direct-20260901-120000`.
- `kind` describes the purpose: `smoke`, `exp`, `eval`, `release`.

## 3. Required fields per file

- `config.yaml`: every configurable value that affects outputs (model,
  loss, data, seed, device policy). Secrets never appear in configs.
- `manifest.json`: dataset manifest hash, split labels used, and
  `test_final` isolation confirmation where applicable.
- `metrics.json`: metric name → `{value, unit, ci_method, ci, n_groups}`.
  Pixels are never reported as independent sample counts; statistics are
  grouped by image or source group.
- `checkpoint-or-reference.txt`: the exact checkpoint used for inference or
  `no-checkpoint` for non-model runs; for fixture/smoke runs, the reference
  input path.

## 4. Artifact naming

- Restored outputs and diagnostics use the tensor contract names
  (`docs/tensor-contract.md`): `input.npy`, `base.npy`, `proposal.npy`,
  `candidate.npy`, `proposal_benefit.npy`, `measurement_consistency.npy`,
  `model_stability.npy`, `distribution_familiarity.npy`, `decision_map.npy`,
  `final.npy`, `unresolved.npy`.
- Reports: `report.md`, `report.json`, `comparison_sheet.png`.
- Artifact metadata (dtype, shape, range, hash) is recorded with each artifact.

## 5. Provenance

Every run bundle records the semantic versions that produced it:

- Model version / commit.
- Dataset manifest version and hash.
- Support definition version (Phase 5).
- Calibration version (Phase 5).
- Forward-model version (Phase 7).
- Decision-policy version (Phase 6).

Where a version does not yet exist, the field is recorded as
`"not-defined"` rather than omitted.

## 6. Storage discipline

- `runs/` and `artifacts/` are never committed to Git (see `.gitignore`).
- Only promoted checkpoints, required comparisons, and failure exemplars are
  retained (EXECUTION.md Part 1 Section 11).
