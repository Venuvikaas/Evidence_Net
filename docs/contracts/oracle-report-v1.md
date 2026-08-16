# Contract: oracle-report-v1

- **Name:** `oracle-report-v1`
- **Version:** v1
- **Status:** frozen
- **Frozen at:** Phase 4 handoff (Research Gate 3: continue, ADR-007)
- **Owner:** ALL lanes
- **Governed by:** `docs/proposal-contract.md` section 4, `docs/evaluation-protocol.md`

## Purpose

Fix the structure and statistical discipline of oracle headroom reports so
the Phase 4 evidence is reproducible and comparable, and so later selective-
risk claims use the same report shape.

## Frozen fields

1. **Decisions.** Oracle gating is computed from ground truth at two
   granularities: **pixel** (accept `p` when
   `|c_p - x_p| < |b_p - x_p|`; ties and increases rejected) and **patch**
   (fixed 16x16 patches on the 256x256 grid; accept `r` when
   `MAE_r(c, x) < MAE_r(b, x)`). The patch grid is the primary region unit
   because it matches the Phase 5 benefit event.
2. **Report fields.** `HeadroomReport` JSON contains `n_groups`, `coverage`
   (pixel/patch), `risk` (pixel/patch), `base_metrics`, `candidate_metrics`,
   `oracle_pixel_metrics`, `oracle_patch_metrics` (each with
   `{mean, ci_lo, ci_hi, n_groups, n_boot}` per primary metric), and
   `structural_impact` (`edge_displacement_px` and `structural_error` for
   base and oracle-patch outputs).
3. **Coverage and risk.** Coverage = fraction of accepted units; risk =
   fraction of units where accepting the proposal increases error (the
   oracle rejects these; an ungated system would take the harm).
4. **Statistics.** Group bootstrap over samples (`n_boot = 1000`, seed 0);
   pixels are never sample counts.
5. **Accepted result.** EXP-004: oracle patch MAE -6.3% vs Base, oracle PSNR
   25.66 dB vs equal-capacity direct 22.60 dB, patch coverage 86.8%, edge
   displacement not detectably increased. Frozen as the Phase 4 headroom
   evidence.

## Implementation references

- Code: `src/evidence_net/evaluation/oracle.py`,
  `src/evidence_net/evaluation/oracle_report.py`
- Tests: `tests/unit/test_oracle.py`, `tests/unit/test_oracle_report.py`
- Experiments: EXP-004, `runs/oracle-gate3-20260815-205601/`

## Change procedure

Changing oracle decision rules, patch size, or report fields requires
`oracle-report-v2`, an ADR, and a rerun decision for EXP-004. The oracle
remains a study tool, never an inference component.
