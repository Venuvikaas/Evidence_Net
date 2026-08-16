# Contract: support-definition-v1

- **Name:** `support-definition-v1`
- **Version:** v1
- **Status:** draft (promotes to frozen after Research Gate 4)
- **Owner:** Lane A (benefit prediction and decision science)
- **Governed by:** `docs/proposal-contract.md`, `docs/evaluation-protocol.md`,
  `docs/metrics-v1` discipline

## Purpose

Fix exactly what "the proposal benefits this region" means (Phase 5). A
**benefit event** is a region where applying the ungated proposal strictly
reduces restoration error relative to the frozen Base output. The definition
is deliberately narrow: it is a **measured property of an output pair**
(Base vs candidate) against the ground truth, evaluated by the same oracle
rule used in Phase 4 — never a claim that restored detail physically existed.

## 1. The benefit event

### 1.1 Region and unit

- **Spatial unit:** the 16x16 patch grid on the 256x256 output grid
  (same patch size as the Phase 4 oracle, `PATCH_SIZE = 16`). A region is
  one patch.
- **Event:** patch `r` is **beneficial** when the patch-level MAE of the
  ungated candidate is lower than the patch-level MAE of the Base output
  by **more than a declared margin** `m`:

      beneficial(r)  <=>  MAE(x_r, c_r) + m < MAE(x_r, b_r)

  where `x` is the clean target, `b` the frozen Base output, `c = clamp(b+d, 0, 1)`
  the ungated candidate, and `d` the bounded proposal. Ties, increases, and
  sub-margin improvements are **not** beneficial.
- **Margin versions:** `labels-v1` uses `m = 0` (strict, matching
  `docs/proposal-contract.md`). `labels-v2` declares `m = 0.005` — Gate 4
  evidence (EXP-009 revision, ADR-016) showed the strict event is dominated
  by sub-margin noise (mean delta 0.0026; all predictors at chance, pooled
  AUC 0.49-0.59), while the meaningful-benefit event is predictable
  (group AUC 0.85-0.93). The promoted event is `labels-v2` with `m = 0.005`.
- The label map is the binary per-patch event over the patch grid; the same
  rule at pixel resolution is reported separately and never merged with the
  patch labels.

### 1.2 Determinism and versioning

- Labels are a pure function of `(b, d, x, m)` on the output grid; they are
  generated deterministically and written as versioned JSON artifacts
  (`benefit-labels-v1.json`) alongside the run bundle.
- The label generator version is fixed at `labels-v1` / `labels-v2` (margin
  parameter); changing the event rule, patch size, margin, or strictness
  requires a new label version (an ADR and a rerun decision).

### 1.3 Utility and population

- **Utility:** the event is the target for the benefit predictor and the
  basis for the Phase 6 selective policy. A beneficial region is one where
  the oracle would accept the proposal (headroom measured in EXP-004).
- **Population:** development data only (train/validation/calibration
  splits of `dataset-splits-v1.json`). `Test_NoisyLR/` never enters label
  generation, predictor training, or calibration. Labels on synthetic
  fixtures are labeled synthetic and never used as scientific evidence.
- **Limitations:** the event says nothing about whether the restored detail
  physically existed; it compares two outputs against ground truth. Harm
  (proposal increases error) is the complement under the strict rule only
  when the proposal is applied ungated; attenuation changes the comparison.

## 2. Prohibited interpretations (Gate 4)

- A beneficial region is not proof of physical detail; the proposal may win
  by chance or by a different trade-off.
- Patch-level benefit is not pixel-level benefit; a mixed patch contains both.
- The label is not a probability; probabilities come from the calibrated
  predictor (CalibrationVersion-v1) and carry their own domain statement.
- No product claim ("the proposal is useful") follows from label quality
  alone; that claim needs the predictor comparison of EXP-009.

## 3. Implementation references

- Code: `src/evidence_net/benefit/labels.py` (deterministic label
  generation, version `labels-v1`)
- Config: `configs/support_definition/support-definition-v1.yaml`
- Script: `scripts/measure_benefit.py`
- Tests: `tests/calibration/test_labels.py`
- Experiment: EXP-009 (Research Gate 4)

## 4. Change procedure

Changing the event rule, patch size, strictness, or population requires
`support-definition-v2`, an ADR, and review by Lane B (diagnostics consume
benefit labels for the incremental-value comparisons of EXP-005/006) and
Lane C (review UI renders benefit regions). The old version stays valid
until all consumers migrate; no report may mix label versions.
