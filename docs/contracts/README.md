# Contract Registry (docs/contracts/)

Versioned, frozen scientific and software interfaces. A contract is the
shared interface that lets four developers work in parallel without silently
breaking one another's lanes.

## Frozen at the Phase 4 handoff (Research Gate 3: continue)

| Contract | Version | File | Owner |
| --- | --- | --- | --- |
| Dataset | v1 | [`dataset-v1.md`](dataset-v1.md) | ALL |
| Tensor | v1 | [`tensor-v1.md`](tensor-v1.md) | ALL |
| Metrics | v1 | [`metrics-v1.md`](metrics-v1.md) | ALL |
| Artifacts and run bundle | v1 | [`artifacts-v1.md`](artifacts-v1.md) | ALL |
| Base output | v1 | [`base-output-v1.md`](base-output-v1.md) | ALL |
| Proposal output | v1 | [`proposal-output-v1.md`](proposal-output-v1.md) | ALL |
| Structural summary | v1 | [`structural-summary-v1.md`](structural-summary-v1.md) | ALL |
| Oracle report | v1 | [`oracle-report-v1.md`](oracle-report-v1.md) | ALL |
| Error and optional fields | v1 | [`error-and-optional-fields-v1.md`](error-and-optional-fields-v1.md) | ALL |

These nine contracts are the freeze list of the four-developer handoff gate
(`EVIDENCE_NET_EXECUTION_WITH_DATASET_4_DEVELOPERS.md`, Part 3 — "Contracts to
freeze"). Every post-handoff lane (A, B, C, D) consumes exactly these
versions; a lane may only add optional fields, never change a frozen field.

## Post-handoff contracts (draft until their gate)

New contracts introduced after the handoff start as **draft** and promote to
frozen when their research gate passes. Lane-owned drafts:

| Contract | Version | Status | Owner | Gate to freeze |
| --- | --- | --- | --- | --- |
| Forward model (measurement consistency) — [`forward-model-v1.md`](forward-model-v1.md) | v1 | draft | B | Gate 6 (Phase 7) |
| Model stability — [`stability-v1.md`](stability-v1.md) | v1 | draft | B | Gate 7 (Phase 8) |
| Distribution familiarity — [`familiarity-v2.md`](familiarity-v2.md) | v2 | draft (v1 superseded) | B | Gate 8 (Phase 9 re-run) |
| Structural risk — [`structural-risk-v1.md`](structural-risk-v1.md) | v1 | draft | B | Gate 9 (Phase 10) |
| Proposal benefit event — [`support-definition-v1.md`](support-definition-v1.md) | v1 | draft | A | Gate 4 (Phase 5) |
| Benefit calibration — [`calibration-version-v1.md`](calibration-version-v1.md) | v1 | draft | A | Gate 4 (Phase 5) |
| Decision policy — [`decision-policy-v1.md`](decision-policy-v1.md) | v1 | draft | A | Gate 5 (Phase 6) |

## How to read a contract file

Each contract records:

- **Status** — `draft` (not yet frozen) or `frozen` (immutable at this
  version).
- **Frozen fields** — the exact interface other lanes may rely on.
- **Implementation references** — code, manifests, and detailed governing
  docs that realize the contract.
- **Change procedure** — the kill-switch rule for altering the contract.

## Contract-change procedure (kill switch)

Changing a **frozen** contract version is a cross-lane event:

1. File an ADR proposal in `DECISIONS.md` naming the affected contract
   version(s) and the migration impact.
2. List every lane (A, B, C, D) that consumes the contract; each affected
   owner must review the proposal.
3. Increment the version (`v1` -> `v2`), add a migration note, and record a
   rerun decision for affected experiments.
4. The old version remains valid until all consumers have migrated; code may
   not silently switch versions.
5. No scientific report may mix outputs from two versions of the same
   contract without stating both versions.

Mechanical enforcement: `scripts/verify_handoff.py` fails CI if any frozen
contract file is missing, unversioned, or marked `draft`. See
[`docs/kill-switches.md`](../kill-switches.md).
