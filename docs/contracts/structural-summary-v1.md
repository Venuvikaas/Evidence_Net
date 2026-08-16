# Contract: structural-summary-v1

- **Name:** `structural-summary-v1`
- **Version:** v1
- **Status:** frozen
- **Frozen at:** Phase 4 handoff (Research Gate 3: continue, ADR-007)
- **Owner:** ALL lanes
- **Governed by:** `docs/proposal-contract.md` section 5, `docs/evaluation-protocol.md`

## Purpose

Fix the per-image structural summaries of a proposal so lanes B (diagnostics)
and C (review UI/reports) consume identical definitions of where and how the
proposal changes the reconstruction.

## Frozen fields

For a proposal `d`, Base `b`, and candidate `c`, per image:

1. **Magnitude.** mean and max of `|d|`; relative mean magnitude
   `mean(|d|) / (range(b) + epsilon)`.
2. **Edge.** mean and max of the normalized edge magnitude of `d` (Sobel
   gradient magnitude, same normalization as `metrics-v1`).
3. **Multi-scale energy.** relative power of `d` per frequency band (low
   `[0, 1/8)`, mid `[1/8, 1/2)`, high `[1/2, 1]` of Nyquist) — where the
   proposal adds or removes energy.
4. **Structural change.** `edge_displacement(b, c)` in px (capped at 16),
   `ssim(b, c)`, and the edge-magnitude difference
   `mean(|grad b| - |grad c|)` between Base and candidate.
5. **Spatial unit.** Per image on the output grid; aggregation follows the
   grouped-statistics discipline of `metrics-v1`.

## Implementation references

- Code: `src/evidence_net/evaluation/proposal_metrics.py`,
  `src/evidence_net/evaluation/metrics.py`
- Tests: `tests/unit/test_proposal_metrics.py`
- Experiments: EXP-004, `runs/proposal-effects-20260815-205856/`

## Change procedure

Redefining a summary (e.g. edge normalization, energy bands, displacement
cap) requires `structural-summary-v2`, an ADR, and a rerun decision for the
proposal-effects analysis. Lanes B and C review the migration.
