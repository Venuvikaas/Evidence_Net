# Contract: calibration-version-v1

- **Name:** `calibration-version-v1`
- **Version:** v1
- **Status:** draft (promotes to frozen after Research Gate 4)
- **Owner:** Lane A (benefit prediction and decision science)
- **Governed by:** `docs/evaluation-protocol.md`, `docs/metrics-v1` discipline

## Purpose

Fix what a calibrated benefit probability means (Phase 5): a probability
**within a stated domain and population** that the patch-level benefit event
(SupportDefinition-v1) holds, estimated by a method fit on **calibration
data only** and evaluated on data never used for fitting. Pre-calibration
scores are always preserved; calibration never replaces them silently.

## 1. Calibration contract

### 1.1 Inputs and output

- **Input:** raw predictor scores `s` (any monotone score, e.g. logits) for
  patch events, plus the deterministic labels of `support-definition-v1`.
- **Output:** calibrated probability `p = f(s)` in [0, 1] with the fitted
  mapping `f` recorded (version, parameters, calibration split id).
- **Preservation:** the raw score `s` is always retained alongside `p`; a
  report may show either, but must state which one it shows.

### 1.2 Split isolation (kill-switch rule)

- Calibration is fit on the **calibration split** of
  `dataset-splits-v1.json` only.
- **No test / held-out data ever enters fitting** — regression tests in
  `tests/calibration/test_split_isolation.py` enforce that the fit function
  rejects data that is not on the calibration split, and that the fitted
  mapping is a pure function of calibration-split data (same scores in ->
  same probabilities out regardless of when they are evaluated).
- Validation data may be used to *choose* between candidate calibration
  methods (model selection), but the chosen method is then re-fit on
  calibration data only before any evaluation; the report states which data
  served which role.

### 1.3 Domain and uncertainty

- The probability is meaningful only within the stated domain: the feature
  population the predictor was trained on and the degradation/geometry
  coverage of the calibration split. Applicability limits are published
  with every report.
- Uncertainty: group bootstrap over source groups (never pixels) for
  Brier and reliability statistics; the calibration curve itself is
  reported with per-bin counts and binomial intervals.

## 2. Candidate methods (v1)

- **Platt scaling** (logistic on logits, 1-D) — v1 default.
- **Temperature scaling** (single multiplicative temperature).
- **Isotonic binning** (monotone step function; only for reports, never the
  v1 default).
- Each candidate preserves pre-calibration scores; the report compares them
  with Brier score and reliability diagrams on the same held-out groups.

## 3. Prohibited interpretations

- A calibrated probability is not a statement that restored detail existed.
- Calibration within the domain does not extend outside it; extrapolated
  claims are prohibited without a new calibration statement.
- Brier score is not accuracy; a well-calibrated weak predictor can still
  order poorly (ranking is reported separately in EXP-009).

## 4. Implementation references

- Code: `src/evidence_net/benefit/calibration.py`
- Config: `configs/calibration/calibration-version-v1.yaml`
- Script: `scripts/measure_benefit.py`
- Tests: `tests/calibration/test_calibration.py`,
  `tests/calibration/test_split_isolation.py`
- Experiment: EXP-009 (Research Gate 4)

## 5. Change procedure

Changing the fitting population, the default method, or the preservation
rule requires `calibration-version-v2`, an ADR, and review by Lane B
(diagnostics consume calibrated scores for the incremental-value
comparisons) and Lane C (review UI renders probabilities). A report may
never mix calibration versions.
