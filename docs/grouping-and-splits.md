# Grouping and Development Splits (Phase 1)

## Source-group hierarchy

The official `train/` directory contains no acquisition/session metadata:
files are flat 6-digit ids under `GT/` and `NoisyLR/`. The leakage-safe
grouping hierarchy is therefore:

```text
source_group (= sample id)   # highest observable unit; no session metadata exists
    └── acquisition          # not present in the data
        └── sample           # one NoisyLR + one GT pair
```

Because no higher-level grouping is observable, **each sample is its own
source unit**. This is the conservative choice: it prevents any assumption of
structure that the data does not expose, and it is documented as a
limitation — if acquisition sessions exist but were not encoded in the file
names, split assignments are still deterministic and can be re-derived when
session metadata becomes available.

## Split policy (dataset-splits-v1, seed 0)

| Split | Fraction | Samples |
| --- | --- | --- |
| train | 0.80 | 2551 |
| validation | 0.10 | 328 |
| calibration | 0.05 | 164 |
| heldout-source | 0.05 | 157 |
| heldout-degradation | 0.00 (reserved) | 0 |

- Assignment is deterministic: `sha256("<seed>:<sample_id>") mod 1000`
  bucketed by cumulative fraction (`src/evidence_net/data/splits.py`).
- Splits are frozen in `data/manifests/dataset-splits-v1.json` (committed,
  immutable; changes require a decision-log entry and rerun decision).
- The **heldout-source** group is a development-time robustness reserve.
- The **heldout-degradation** group is **reserved with zero members** because
  no degradation labels exist; it must be populated from documented
  degradation labels before it can be used. Do not invent degradation labels.

## Isolation rules

- Development splits are built **only** from
  `official-train-source-v1.json`.
- `official-test-noisylr-source-v1` is never consulted for development
  decisions; the split builder raises if any test path is present, and
  `tests/unit/test_isolation.py` enforces this over committed manifests.
- `Test_NoisyLR/` is used only for the final evaluation inference in
  Phase 18.

## Changing the policy

Any change to fractions, seed, or grouping requires:

1. A decision-log entry (ADR) recording the rationale and evidence.
2. A version increment of the splits manifest.
3. A rerun decision for affected experiments (trained models, calibration,
   policies).
