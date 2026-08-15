# Official `train/` Structure (Observed)

- **Status:** Observed facts only; no pairing rules are invented.
- **Resolved root:** `<project-parent>/train` (from `TRAIN_DATA_DIR` /
  execution-file parent discovery, see `docs/data-card.md`).
- **Manifest:** `data/manifests/official-train-source-v1.json`.

## Observed layout

```text
<project-parent>/train/
├── __MACOSX/              # macOS zip-extraction junk (AppleDouble ._ files) — excluded
├── .DS_Store              # macOS metadata — excluded
└── train/                 # dataset content root
    ├── GT/                # clean targets
    │   ├── 000000.npy
    │   ├── 000001.npy
    │   └── ... 003199.npy
    └── NoisyLR/           # degraded low-resolution inputs
        ├── 000000.npy
        ├── 000001.npy
        └── ... 003199.npy
```

## Observed file facts (from the frozen source manifest)

| Property | GT (targets) | NoisyLR (inputs) |
| --- | --- | --- |
| Count | 3200 | 3200 |
| Names | `000000.npy` … `003199.npy` | `000000.npy` … `003199.npy` |
| Extension | `.npy` | `.npy` |
| Shape | `(256, 256)` for all 3200 | `(128, 128)` for all 3200 |
| Dtype | `float32` | `float32` |
| Range (dataset) | exactly `[0.0, 1.0]` for all 3200 (clipped) | min −0.279 … max 2.158; per-file ranges vary |
| Readable | 3200 / 3200 | 3200 / 3200 |

## Candidate pairing rule (from observed names and structure)

The noisy input and target relationship is identified only from observed
names, metadata, and the directory structure:

- Noisy inputs live in `train/NoisyLR/`, targets in `train/GT/`.
- Files are paired **by zero-padded base name**: `NoisyLR/000123.npy`
  ↔ `GT/000123.npy`.
- The pairing adapter (`src/evidence_net/data/pairing.py`) enforces exactly
  one `GT` and one `NoisyLR` directory, pairs by base name, and **reports**
  unmatched, duplicated, or ambiguous files rather than silently skipping
  them.
- Resolution ratio is exactly 2.0 (GT 256×256, input 128×128).

## Junk handling

- `__MACOSX/` and hidden entries (names starting with `.`) are excluded from
  inventory, pairing, and splitting.
- `train/train/.DS_Store` is excluded (non-`.npy`).

## Unresolved

- No acquisition/session metadata is present; each sample is treated as its
  own source unit (see `docs/grouping-and-splits.md`).
- The exact degradation pipeline is not documented in the local files;
  statistics only (see `docs/data-card.md`).
