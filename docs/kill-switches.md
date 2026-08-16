# Kill Switches

Kill switches are the predeclared conditions under which a lane — or the
whole project — must stop, redesign, remove, or roll back. They are the
enforcement half of the research-gate system in
`EVIDENCE_NET_EXECUTION_WITH_DATASET_4_DEVELOPERS.md`. A kill switch is not a
silent deletion: it is a recorded decision (in `DECISIONS.md` /
`FAILURES.md`) followed by the predeclared action.

## How a gate works

Every research gate (Gates 4-10 for the four lanes, Gates 1-3 already
passed) declares acceptance criteria **before** evidence is collected.
Possible outcomes: **continue**, **redesign**, **remove**, or **stop**.
A lane whose gate fails does not merge further phase work until the decision
is recorded.

## Per-lane kill switches (scientific)

| Switch | Lane | Trigger condition | Action |
| --- | --- | --- | --- |
| Gate 4 | A | Benefit Predictor does not beat declared simple heuristics, or has no useful selective-risk ordering, or calibration is invalid within its stated domain | Simplify, redefine the event, redesign the proposal, or **stop the support-aware claim** |
| Gate 5 | A | Selective action does not improve a predeclared outcome, or abstention does not lower risk at useful coverage, or rejection certifies the Base output | Redesign the policy; do not disguise unresolved Base errors |
| Gate 6 | B | Measurement-consistency diagnostic adds no held-out value and no independently useful review information | **Remove the diagnostic** (it is compatibility, not truth) |
| Gate 7 | B | Stability adds no incremental value after controlling for simple features | Remove costly ensembles that fail incremental-value tests; never convert agreement into probability of truth |
| Gate 8 | B | Familiarity diagnostic fails to detect declared shifts, or systematically suppresses rare valid structures | Redesign the representation or threshold; bind calibration claims to the validated domain |
| Gate 9 | B | Structural claims lack separate candidate, ambiguity, acquisition, natural-failure, or downstream evidence | Do not claim hallucination resistance from candidate manipulation alone |
| Gate 10 | C (A+B review) | Users cannot distinguish Benefit, compatibility, stability, familiarity, rejection, and unresolved output, or treat them as physical proof | Rename, redesign, or **remove** the misunderstood layer |

## Global kill switches (process)

| Switch | Owner | Trigger condition | Action |
| --- | --- | --- | --- |
| Handoff gate | ALL | The nine frozen contracts are not all frozen, checkpoints not pinned, or lanes cannot reproduce the Phase 4 vertical slice | No parallel lane work starts; repair the handoff first |
| Dataset isolation | ALL | Any `Test_NoisyLR/` path enters a training, validation, calibration, hyperparameter-search, threshold-selection, or policy config | CI fails (`tests/unit/test_isolation.py`, `test_handoff.py`, `scripts/verify_handoff.py`); the offending change is blocked |
| Contract violation | ALL | A frozen contract version is changed without an ADR, version increment, migration note, and affected-owner review | CI fails (`scripts/verify_handoff.py`); change is blocked until the ADR passes |
| Secret / restricted data | D | A secret or restricted dataset lands in Git history | Halt the merge; rotate the secret; rewrite history per Phase 16 audit; add a failure entry |
| Gate 3 revocation | ALL | New evidence invalidates the oracle-headroom or handoff assumptions | Reopen the gate decision; all four lanes pause until a new decision is recorded |
| Integration failure | ALL | A promoted artifact fails its integration checkpoint (I-V) or end-to-end gate tests | The artifact is not promoted; the producing lane fixes or rolls back before promotion |
| Phase 18 freeze | ALL | Any post-freeze tuning, any missing provenance, any isolation or parity failure | Release is blocked; fix before tagging the validated release |

## Mechanical enforcement

- `scripts/verify_handoff.py` (run in CI and locally) exits non-zero when:
  - a frozen contract file is missing, unversioned, or marked `draft`;
  - `CODEOWNERS` or the fixture registry is missing/invalid;
  - a tracked file under `data/`, `configs/`, or `src/` references a
    `Test_NoisyLR` path (the isolation kill switch);
  - a registered checkpoint hash is missing.
- `tests/unit/test_handoff.py` covers the same invariants as unit tests.
- CI runs lint, format, mypy, pytest, environment check, smoke, and the
  handoff verification on every push and PR.

## Stopping a lane

1. Record the evidence and decision in `DECISIONS.md` (and `FAILURES.md` for
   negative results).
2. Freeze the lane's artifacts; mark the lane's phase incomplete.
3. Notify affected owners via the integration checkpoint; any consumed
   contract is handled by the contract-change procedure.
4. Redesign or remove per the gate outcome, or pause the lane entirely.
   A paused lane may resume only with a new recorded decision.

A kill switch is never a reason to hide a result. Negative results are
project assets (`FAILURES.md`).
