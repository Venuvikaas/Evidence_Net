# Official `Test_NoisyLR/` Structure (Observed)

- **Status:** Observed facts only. This directory is **isolated evaluation
  input**: it must never influence training, validation, calibration,
  hyperparameter search, or threshold selection.
- **Resolved root:** `<project-parent>/Test_NoisyLR` (from
  `TEST_NOISY_LR_DIR` / execution-file parent discovery, see
  `docs/data-card.md`).
- **Manifest:** `data/manifests/official-test-noisylr-source-v1.json`.

## Observed layout

```text
<project-parent>/Test_NoisyLR/
├── __MACOSX/              # macOS zip-extraction junk — excluded
└── NoisyLR/               # isolated evaluation inputs
    ├── 000000.npy
    ├── 000001.npy
    └── ... 000399.npy
```

## Observed file facts (from the frozen source manifest)

| Property | NoisyLR (inputs) |
| --- | --- |
| Count | 400 |
| Names | `000000.npy` … `000399.npy` |
| Extension | `.npy` |
| Shape | `(128, 128)` for all 400 |
| Dtype | `float32` |
| Range (dataset) | min −0.225 … max 2.158; per-file ranges vary |
| Readable | 400 / 400 |

## Targets

- **Targets are absent** from the local `Test_NoisyLR/` directory: there is no
  `GT` directory and no target files. Targets are not inaccessible or
  separately governed locally — they are simply not provided.
- Final evaluation therefore measures **outputs only** (no local ground
  truth). Any required submission/output naming convention must come from the
  challenge evaluation contract; none is present in the local files, so the
  required output naming is recorded as **unresolved** until the evaluation
  contract is consulted.

## Isolation guarantees

- The test manifest carries **no development labels** (no split labels, no
  roles) and is kept free of metrics.
- Automated checks (`tests/unit/test_isolation.py`) fail if any
  `Test_NoisyLR/` path enters a training, validation, calibration,
  hyperparameter-search, or threshold-selection manifest.
- `scripts/dryrun_loader.py --dataset test` reads every input through the
  raw-preserving loader without running final evaluation.

## Compatibility with train inputs

Train and test inputs share extension, shape (`(128, 128)`), channels (1),
dtype (`float32`), and overlapping raw ranges (audit result: compatible).
