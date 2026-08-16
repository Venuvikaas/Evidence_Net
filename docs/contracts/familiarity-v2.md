# Contract: familiarity-v2

- **Name:** `familiarity-v2`
- **Version:** v2 (supersedes the v1 draft; v1 is retained for reference)
- **Status:** draft (freezes after Research Gate 8 re-run)
- **Owner:** Lane B (diagnostics); reviewed by A and C at promotion
- **Governed by:** `docs/modality-contract.md`, `docs/tensor-contract.md`,
  `docs/evaluation-protocol.md`

## Purpose

Fix the distribution-familiarity diagnostic (Phase 9) after Gate 8 identified
two failure modes in `familiarity-v1` on the real governed run
(`runs/familiarity-gate8-real/`):

1. **Systematic suppression of rare valid structures** — v1's feature vector
   was dominated by global brightness, so dark/bright inputs were flagged as
   unfamiliar for their brightness alone (rare-valid false-warning rate
   1.000 against a declared 0.50 cap).
2. **No shift detection at all** — v1's threshold was a fixed constant
   (2.0) never calibrated to the reference population's own spread, so
   every group scored 0.000 detection.

v2 fixes both: a **brightness-invariant feature representation** and a
**threshold calibrated from the reference population's own leave-one-out
distance spread**. Familiarity still reports **where the input sits in
feature space**, never what the output means. Unfamiliar is not wrong;
familiar is not correct; the diagnostic never certifies outputs.

## 1. Representation (frozen features)

- Features are computed on the **input grid** (128x128 for the official
  dataset) from the degraded observation — the acquisition/degradation
  domain the model actually consumes.
- Frozen feature vector `g(y)` (7 components, `float64`):
  - `std` — pixel standard deviation of the **z-scored** grid (scale
    information, brightness-normalized);
  - `energy_low`, `energy_high` — relative radial power per band on the
    raw grid (same bands as `metrics-v1`; fractions already
    brightness-invariant);
  - `edge_density` — mean normalized Sobel gradient magnitude on the
    **z-scored** grid;
  - `local_std_mean`, `local_std_p90`, `local_std_spread` — mean, 90th
    percentile, and spread of 16x16 patch standard deviations on the
    **z-scored** grid (texture-grain signature).
- The global pixel `mean` is deliberately **excluded**: global brightness
  is not familiarity. A dark input and a bright input with the same
  structure receive the same v2 vector (verified by unit test).
- Feature computation is deterministic; constant images are epsilon-guarded
  (z-scored grid is all zeros), never NaN.

## 2. Reference population

- Same rule as v1: a **frozen set of development inputs** only — the
  calibration split of `train/` in real mode; a seeded synthetic population
  in smoke mode (labeled synthetic, never used in scientific reports).
  `Test_NoisyLR/` never enters the reference population.

## 3. Distance and threshold (changed in v2)

- **Distance:** RMS of per-feature standardized deviations
  `sqrt(mean((g - mean_ref)^2 / (std_ref + eps)^2))` in the v2 feature
  space.
- **Threshold (calibrated, not fixed):** the threshold is the
  `calibration_quantile` (default 0.90) of the **leave-one-out** distances
  of the reference population itself — the standard OOD calibration
  practice. It is fit on development inputs only, never on probes, so no
  post-hoc tuning is possible. An input is `familiar` when
  `distance <= threshold`, `unfamiliar` otherwise.
- Distance is reported per image / source group with the grouped-statistics
  discipline of `metrics-v1`; pixels are never sample counts.

## 4. Applicability limits (frozen)

- The diagnostic is valid **only within the feature domain of the reference
  population**; any calibration claim is bound to that validated domain.
- Shift detection is reported per declared shift group (source, severity,
  degradation, acquisition). Detection rate = fraction of probes in a group
  flagged unfamiliar.
- **Rare valid structures** (thin lines, isolated points, small defects)
  are evaluated **separately** and **in-domain**: v2 injects them into real
  validation inputs rather than scoring synthetic dark-flat fixtures, so
  the no-suppression property is tested where it matters. The report
  carries their false-warning rate against the declared cap (Gate 8).
- Data-truth limit published from the real run: the official `train/`
  manifest records **no acquisition/session metadata — each sample is its
  own source unit** (`dataset-splits-v1.json` grouping note), so the
  declared **source** shift is not a measurable distribution shift in this
  dataset and its detection rate is not evidence of diagnostic failure.

## 5. Prohibited claims

- Unfamiliar is not wrong; familiar is not correct; neither certifies the
  restored output.
- Familiarity is not a context-free trust score and never calibrates the
  benefit predictor.
- The diagnostic does not claim to detect every possible shift — only the
  declared shift types within the declared feature representation.
- Detection rates are bounded by what the reference population's spread
  supports: near-identity perturbations of already-degraded inputs (weak
  severity/acquisition probes) are expected to rank above chance but flag
  below the calibrated threshold by design.

## 6. Implementation references

- Code: `src/evidence_net/stress_tests/familiarity.py`
  (`feature_vector_v2`, `ReferenceFamiliarityV2`, `inject_rare_valid`)
- Config: `configs/modality/familiarity-v2.yaml`
- Script: `scripts/measure_familiarity.py`
- Tests: `tests/numerical/test_familiarity.py`
- Experiment: EXP-007 (shift detection and rare-structure behavior,
  Research Gate 8 re-run)

## 7. Change procedure

Changing the feature representation, reference population rule, distance,
or threshold semantics requires `familiarity-v3`, an ADR, and review by
lanes A (warnings/abstention policy consume familiarity) and C (review UI
renders the diagnostic). Unproven familiarity outputs stay disabled by
default (Integration II).
