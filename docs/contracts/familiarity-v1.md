# Contract: familiarity-v1

- **Name:** `familiarity-v1`
- **Version:** v1
- **Status:** draft (promotes to frozen after Research Gate 8)
- **Owner:** Lane B (diagnostics); reviewed by A and C at promotion
- **Governed by:** `docs/modality-contract.md`, `docs/tensor-contract.md`,
  `docs/evaluation-protocol.md`

## Purpose

Fix the distribution-familiarity diagnostic (Phase 9): a reference-distance
measure of how far a degraded input lies from a frozen reference population
of development inputs. Familiarity reports **where the input sits in feature
space**, never what the output means. Unfamiliar is not wrong; familiar is
not correct; and the diagnostic must never systematically suppress rare
**valid** structures.

## 1. Representation (frozen features)

- Features are computed on the **input grid** (128x128 for the official
  dataset) from the degraded observation — the acquisition/degradation
  domain the model actually consumes.
- Frozen feature vector `f(y)` (6 bounded components, `float64`):
  - `mean` — pixel mean;
  - `std` — pixel standard deviation;
  - `energy_low`, `energy_mid`, `energy_high` — relative radial power per
    band (same bands as `metrics-v1`: `[0, 1/8)`, `[1/8, 1/2)`,
    `[1/2, 1]` of Nyquist; fractions sum to 1);
  - `edge_density` — mean normalized Sobel gradient magnitude.
- Feature computation is deterministic; a constant image yields `std = 0`
  with the epsilon guard, never NaN.

## 2. Reference population

- The reference population is a **frozen set of development inputs** only:
  the calibration split of `train/` in real mode; a seeded synthetic
  population in smoke mode (labeled synthetic, never used in scientific
  reports). `Test_NoisyLR/` never enters the reference population.
- The baseline fits per-feature mean and standard deviation over the
  reference (epsilon-guarded); the fit is part of the diagnostic contract.

## 3. Distance and threshold

- **Distance:** RMS of per-feature standardized deviations
  `sqrt(mean((f - mean_ref)^2 / (std_ref + eps)^2))` — the simplest
  reference-distance baseline.
- **Threshold:** a configurable value (`familiarity-v1` config); an input is
  `familiar` when `distance <= threshold`, `unfamiliar` otherwise.
- Distance is reported per image / source group with the grouped-statistics
  discipline of `metrics-v1`; pixels are never sample counts.

## 4. Applicability limits (frozen)

- The diagnostic is valid **only within the feature domain of the reference
  population**; any calibration claim is bound to that validated domain.
- Shift detection is reported per declared shift group (source, severity,
  degradation, acquisition). Detection rate = fraction of probes in a group
  flagged unfamiliar.
- **Rare valid structures** (thin lines, isolated points, small defects) are
  evaluated **separately**: the report carries their false-warning rate (the
  fraction of rare-valid probes flagged unfamiliar) so systematic suppression
  is visible and gate-checkable (Gate 8).

## 5. Prohibited claims

- Unfamiliar is not wrong; familiar is not correct; neither certifies the
  restored output.
- Familiarity is not a context-free trust score and never calibrates the
  benefit predictor.
- The diagnostic does not claim to detect every possible shift — only the
  declared shift types within the declared feature representation.

## 6. Implementation references

- Code: `src/evidence_net/stress_tests/familiarity.py`
- Config: `configs/modality/familiarity-v1.yaml`
- Script: `scripts/measure_familiarity.py`
- Tests: `tests/numerical/test_familiarity.py`
- Experiment: EXP-007 (shift detection and rare-structure behavior,
  Research Gate 8)

## 7. Change procedure

Changing the feature representation, reference population rule, distance, or
threshold semantics requires `familiarity-v2`, an ADR, and review by lanes A
(warnings/abstention policy consume familiarity) and C (review UI renders the
diagnostic). Unproven familiarity outputs stay disabled by default
(Integration II).
