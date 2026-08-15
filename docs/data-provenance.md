# Data Provenance — Inventory, Classification, and Quarantine (Phase 1)

## Candidate dataset inventory

| Dataset | Source | License | Modality | Pairing | Resolution | Target meaning |
| --- | --- | --- | --- | --- | --- | --- |
| `train/` (official) | KLA problem statement, local | **Not stated in local files** | Semiconductor-like structural imagery (NoisyLR → GT) | 6-digit base-name pairing | 128×128 → 256×256 | Clean reference, clipped to [0, 1] |
| `Test_NoisyLR/` (official) | KLA problem statement, local | **Not stated in local files** | Same | Isolated inputs (no targets) | 128×128 | None locally (targets absent) |
| `KLA Problem Statement_explanation.pptx` | KLA problem statement | Documentation | — | — | — | Problem statement, not data |

No other candidate datasets are present in the execution parent.
