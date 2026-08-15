# Data Provenance — Inventory, Classification, and Quarantine (Phase 1)

## Candidate dataset inventory

| Dataset | Source | License | Modality | Pairing | Resolution | Target meaning |
| --- | --- | --- | --- | --- | --- | --- |
| `train/` (official) | KLA problem statement, local | **Not stated in local files** | Semiconductor-like structural imagery (NoisyLR → GT) | 6-digit base-name pairing | 128×128 → 256×256 | Clean reference, clipped to [0, 1] |
| `Test_NoisyLR/` (official) | KLA problem statement, local | **Not stated in local files** | Same | Isolated inputs (no targets) | 128×128 | None locally (targets absent) |
| `KLA Problem Statement_explanation.pptx` | KLA problem statement | Documentation | — | — | — | Problem statement, not data |

No other candidate datasets are present in the execution parent.

## Target provenance classification

The exact provenance of the clean targets is **not documented** in the local
files. Observed evidence supports a *synthetic-style degradation pipeline*
hypothesis, but this is a hypothesis, not a confirmed fact:

- Input mean ≈ target mean (0.2184 vs 0.2182 on sampled pairs): consistent
  with zero-mean-ish additive degradation plus down-sampling.
- Resolution ratio is exactly 2.0 for every pair.
- Targets are exactly clipped to `[0, 1]`; inputs are not clipped and contain
  out-of-range values.
- Residuals after 2× pooling/offset alignment remain substantial (MAE
  ≈ 0.03–0.07), i.e., inputs are not a clean deterministic function of the
  targets.

**Classification:** real-vs-synthetic **unknown**; targets are treated as the
official clean reference for supervised training, and calibration claims will
be restricted to the stated dataset domain.
