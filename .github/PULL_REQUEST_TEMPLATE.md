<!--
PR rules (docs/four-developer-workflow.md): one lane objective per PR; name
every frozen contract version you consume; cross-lane changes need an ADR and
affected-owner review. Delete this comment block.
-->

## Lane and objective

- **Lane:** A (Benefit/Decision) | B (Diagnostics) | C (Product/Review) | D (Deploy/Ops) | Shared
- **Phase(s):**
- **Objective (one active implementation objective):**

## Contracts consumed (must name exact frozen versions)

- [ ] `dataset-v1`
- [ ] `tensor-v1`
- [ ] `metrics-v1`
- [ ] `artifacts-v1`
- [ ] `base-output-v1`
- [ ] `proposal-output-v1`
- [ ] `structural-summary-v1`
- [ ] `oracle-report-v1`
- [ ] `error-and-optional-fields-v1`
- New optional fields introduced: (name their new contract versions)

## Contract changes (if any)

- [ ] No frozen contract changed
- [ ] ADR filed in `DECISIONS.md` with migration impact
- [ ] Affected-owner review completed (list owners)

## Gate evidence

- Research gate this PR addresses (e.g. Gate 4/5/6/7/8/9/10, integration
  checkpoint I-V):
- Gate decision / expected outcome:
- Experiment registration (`EXPERIMENTS.md` EXP-XXX) if this PR reports
  results:
- Fixtures consumed (name schema + producer versions):

## Quality gates

- [ ] `ruff check .` passes
- [ ] `mypy src/evidence_net scripts` passes
- [ ] `pytest` passes
- [ ] `python scripts/verify_handoff.py` passes (handoff kill switch)
- [ ] `Test_NoisyLR/` isolation tests pass (no test-final data used)
