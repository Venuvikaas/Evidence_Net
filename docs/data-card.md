# Data Card — Official Local Datasets (Phase 1)

- **Status:** Registered (Phase 1). Extended with audit findings at the
  Phase 1 checkpoint.
- **Execution parent:** `<project-parent>` (the directory containing
  `EVIDENCE_NET_EXECUTION_WITH_DATASET.md`).
- **Resolution:** `TRAIN_DATA_DIR` / `TEST_NOISY_LR_DIR` environment
  variables or `.env`, falling back to `<project-parent>/train` and
  `<project-parent>/Test_NoisyLR` (`scripts/resolve_dataset_paths.py`).

## Dataset overview

| Dataset | Root | Content | Files | Role |
| --- | --- | --- | --- | --- |
| `official-train-source-v1` | `<parent>/train` | `train/GT` + `train/NoisyLR` | 3200 pairs | Development data (train / validation / calibration / held-out) |
| `official-test-noisylr-source-v1` | `<parent>/Test_NoisyLR` | `NoisyLR/` | 400 inputs | Isolated final evaluation input; **never** development |

See `docs/train-structure.md` and `docs/test-noisylr-structure.md` for the
observed structures.

## Access method

- The official directories are provided as local directories (and as
  `train.zip` / `Test_NoisyLR.zip` in the execution parent).
- They are resolved from configuration, never from an assumed current working
  directory.
- They are **never committed to Git** (`.gitignore`); only frozen manifest
  files under `data/manifests/` are committed.

## Provenance information available to the developer

- The datasets accompany the KLA problem statement
  (`KLA Problem Statement_explanation.pptx` in the execution parent) and the
  project's final-idea documents.
- No license text, acquisition description, sensor model, or degradation
  pipeline documentation is present in the local files.
- The local files contain only the `.npy` tensors and macOS extraction junk;
  no README or metadata files were shipped.

## Observed contents

- Inputs (`NoisyLR`): 128×128 `float32`, values outside `[0, 1]`
  (min ≈ −0.28, max ≈ 2.16 across the train set).
- Targets (`GT`): 256×256 `float32`, every file exactly within `[0, 1]`
  (clipped/normalized clean references).
- Resolution ratio: exactly 2.0 (super-resolution-style pairing).
- Junk excluded: `__MACOSX/`, `.DS_Store`, hidden entries.

## Unresolved restrictions

- License and permitted-use terms are **not stated** in the local files; the
  datasets are treated as official challenge materials (see
  `docs/data-provenance.md` for the quarantine policy).
- Target provenance (real vs. synthetic) is not documented locally
  (see `docs/data-provenance.md`).
- Degradation labels are absent; degradation family is a hypothesis from
  statistics, not a confirmed fact.
- Required output naming for `Test_NoisyLR/` evaluation is unresolved (no
  evaluation contract in the local files).
- Target alignment between inputs and targets is not a clean 2× grid
  relationship (see `docs/train-structure.md` and the audit).

## Audit findings (Phase 1, `runs/audit-*/metrics.json`)

| Check | Result |
| --- | --- |
| Pair integrity | 3200 pairs; 0 unmatched, 0 duplicated, 0 ambiguous |
| Readability | 6400/6400 train files, 400/400 test files readable |
| Exact duplicates | 0 groups |
| Near duplicates (32×32 pooled signature) | 0 groups |
| Train/test input compatibility | compatible (extension, shape, channels, dtype, range family) |
| Resolution ratio (GT/NoisyLR) | exactly 2.0 for all pairs |
| Alignment (2× block-offset) | no dominant phase (56/61/42/41 of 200); mean best-offset MAE ≈ 0.067 |
| Input range (train NoisyLR) | min −0.279 … max 2.158 (per-file vary) |
| Target range (train GT) | exactly [0.0, 1.0] for all 3200 |
| Test input range | min −0.225 … max 2.158 |
| Clipping | all targets in [0, 1]; 18/3200 train inputs and 2/400 test inputs fully in [0, 1] |

### Alignment and target uncertainty

- Method: 2× block-offset search over offsets {(0,0),(0,1),(1,0),(1,1)}
  comparing NoisyLR against GT sub-blocks; lowest-MAE offset recorded;
  MAE at the best offset is the alignment residual (documented in
  `src/evidence_net/data/audit.py`).
- Finding: no offset dominates (56/61/42/41 of 200 pairs), so the inputs are
  **not a clean 2× subsample** of the targets; residual MAE ≈ 0.067
  (min 0.012, max 0.151). Box-2×2 pooling reduces MAE slightly
  (≈ 0.03 on sampled pairs), consistent with anti-aliased down-sampling plus
  noise.
- Consequence: dataset-level target-alignment uncertainty is recorded in
  every train manifest file (`target_uncertainty`); per-pair uncertainty is
  not yet available.

### Duplicates

- Exact (sha256): 0 groups. Near (32×32 mean-pooled, 3-decimal quantized
  signature): 0 groups. No duplicate handling is required in the splits.

### Splits and grouping

- Frozen in `data/manifests/dataset-splits-v1.json` (seed 0): train 2551,
  validation 328, calibration 164, heldout-source 157,
  heldout-degradation 0 (reserved). Grouping is sample-level because no
  session metadata exists (`docs/grouping-and-splits.md`).

## Frozen manifests and hashes (dataset-manifest-v1)

- `official-train-source-v1.json` — 6400 files (3200 inputs + 3200 targets),
  sha256 `c504b2dded0f3a04…`.
- `official-test-noisylr-source-v1.json` — 400 inputs, sha256
  `aab75186e9a46982…`.
- `dataset-splits-v1.json` — seed 0, counts above.
- The aggregate `data/manifests/dataset-manifest-v1.json` records the sha256
  of every referenced manifest; `scripts/verify_manifests.py` re-verifies
  them.

## Limitations (honest statement)

- License terms are unstated in the local files; use is limited to the
  official challenge scope.
- Target provenance (real vs. synthetic) is unknown; degradation labels are
  absent.
- Target alignment is not a clean 2× grid; supervised training should treat
  per-pixel target correspondence as approximate.
- `Test_NoisyLR/` has no local targets; final evaluation is output-only until
  the evaluation contract supplies ground truth or submission rules.
- These limitations do not block supervised development (EXP-001, ADR-005),
  but any metric or calibration claim must stay within this stated domain.
