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


