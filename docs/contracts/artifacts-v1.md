# Contract: artifacts-v1

- **Name:** `artifacts-v1`
- **Version:** v1
- **Status:** frozen
- **Frozen at:** Phase 4 handoff (Research Gate 3: continue, ADR-007)
- **Owner:** ALL lanes
- **Governed by:** `docs/run-and-artifact-contract.md`

## Purpose

Fix the run-bundle layout, artifact naming, and provenance rules so every
lane produces and consumes interchangeable, traceable artifacts.

## Frozen fields

1. **Run bundle.** Every governed run writes
   `runs/<run_id>/` with `config.yaml`, `manifest.json`, `environment.txt`,
   `metrics.json`, `summary.md`, `artifacts/`, `logs/`, and
   `checkpoint-or-reference.txt`. Run IDs: `<kind>-<YYYYMMDD>-<HHMMSS>`
   (UTC); kinds: `smoke`, `exp`, `eval`, `release`.
2. **Artifact names.** Tensors use the `tensor-v1` names: `input.npy`,
   `base.npy`, `proposal.npy`, `candidate.npy`, `proposal_benefit.npy`,
   `measurement_consistency.npy`, `model_stability.npy`,
   `distribution_familiarity.npy`, `decision_map.npy`, `final.npy`,
   `unresolved.npy`. Reports: `report.md`, `report.json`,
   `comparison_sheet.png`. Each artifact records metadata (dtype, shape,
   range, hash) beside the tensor.
3. **Provenance.** Every run bundle records semantic versions: model,
   dataset manifest (hash), support definition, calibration, forward model,
   and decision policy; a version that does not exist yet is recorded as
   `"not-defined"`, never omitted.
4. **Never in Git.** `runs/` and `artifacts/` are never committed. Only
   promoted checkpoints, required comparisons, and failure exemplars are
   retained (storage discipline).
5. **Metrics schema.** `metrics.json` maps metric name ->
   `{value, unit, ci_method, ci, n_groups}`; pixels are never sample counts.

## Implementation references

- Code: `src/evidence_net/reporting/run_bundle.py`,
  `src/evidence_net/reporting/comparison_report.py`,
  `src/evidence_net/training/provenance.py`
- Tests: `tests/unit/test_run_bundle.py`, `tests/unit/test_report.py`

## Change procedure

Adding an artifact name or changing the bundle layout requires
`artifacts-v2`, an ADR, and affected-owner review (C and D are the primary
consumers; A and B are the primary producers of new fields).
