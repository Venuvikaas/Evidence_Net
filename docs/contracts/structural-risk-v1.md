# Contract: structural-risk-v1

- **Name:** `structural-risk-v1`
- **Version:** v1
- **Status:** draft (promotes to frozen after Research Gate 9)
- **Owner:** Lane B (structural validation); reviewed by A (policy on the
  failure bank) and D (hidden-test integrity) at promotion
- **Governed by:** `docs/modality-contract.md`, `docs/evaluation-protocol.md`,
  `docs/failures` ledger

## Purpose

Fix the structural-risk test program (Phase 10) as **five separate threat
models**. Structural claims require separate candidate, ambiguity,
acquisition, natural-failure, and downstream evidence; no single suite may be
used to claim hallucination resistance (Gate 9).

## 1. Threat models (frozen labels)

| Threat model | Applies to | Frozen contents |
| --- | --- | --- |
| `candidate` | restored output | false-line insertion, real-line deletion, edge shift, merge, split, false periodicity, defect point |
| `ambiguity` | clean-candidate pairs | pairs of distinct clean candidates whose observations are near-identical (stripe and line cases from `forward-model-v1`) |
| `acquisition` | degraded input (pre-inference) | sensor noise, column striping, gain non-uniformity, dead pixels, local blur patch |
| `natural` | unedited frozen-model outputs | curated natural failure bank (`data/failures/natural-failures-v1.json`) |
| `downstream` | restored output | a frozen measurement task evaluated without co-training |

Reports must keep these categories separate; conflation is a contract
violation.

## 2. Frozen candidate manipulations (candidate threat model)

All manipulate the restored output on the output grid and are labeled
`candidate`:

- **False-line insertion** — a thin line that may not exist in the target
  (hallucination probe).
- **Real-line deletion** — removal of a dominant line (structure loss).
- **Edge shift** — a dominant edge moved by a fixed pixel amount.
- **Merge** — two nearby structures joined.
- **Split** — one structure broken into two.
- **False periodicity** — an added periodic pattern.
- **Defect point** — an isolated defect-like point added or removed.

Geometry and amplitudes are **frozen** in the hidden stress definitions
(`data/stress/hidden-stress-v1.json`), never ad-hoc in tests.

## 3. Frozen acquisition artifacts (acquisition threat model)

Applied to the degraded input before inference, bounded and clipped to
`[0, 1]`: additive sensor noise, column striping, gain non-uniformity, dead
pixels, and a local blur patch. Parameters are frozen in the hidden stress
definitions.

## 4. Hidden stress definitions and isolation

- Final stress perturbation/acquisition parameters are frozen, hash-verified,
  in `data/stress/hidden-stress-v1.json`.
- **Training code never reads stress definitions.** An automated test scans
  `src/evidence_net/training/` and fails on any reference to `stress_tests`
  or the hidden stress file (support training cannot tune against the final
  stress program).
- Changing the hidden definitions requires `hidden-stress-v2`, an ADR, and
  lane-D review of test integrity.

## 5. Frozen downstream task (downstream threat model)

- The frozen downstream task is **measurement fidelity** of a restored output
  against the target, over three frozen measurements: edge displacement (px),
  connected components of binary edges, and connected components of bright
  structures. The task is a pure function of outputs and targets; it is never
  co-trained and never reads hidden stress definitions.
- Downstream evaluation compares the frozen Base, the ungated candidate, and
  the oracle-patch output (a study proxy for selective restoration; the
  oracle is never used at inference). Grouped bootstrap CIs are reported per
  measurement (`metrics-v1` discipline).

## 6. Prohibited claims

- No hallucination-resistance claim from candidate manipulation alone.
- No conflation of candidate, ambiguity, acquisition, natural, and downstream
  evidence in any report.
- No claim that passing a stress suite proves the restored detail existed.

## 7. Implementation references

- Code: `src/evidence_net/stress_tests/structural.py`,
  `src/evidence_net/stress_tests/ambiguity.py`,
  `src/evidence_net/stress_tests/acquisition.py`,
  `src/evidence_net/stress_tests/downstream.py`,
  `src/evidence_net/stress_tests/hidden_stress.py`
- Hidden definitions: `data/stress/hidden-stress-v1.json`
- Failure bank: `data/failures/natural-failures-v1.json`
- Script: `scripts/measure_structural_risk.py`
- Tests: `tests/numerical/test_structural.py`,
  `tests/numerical/test_stress_isolation.py`
- Experiment: EXP-008 (structural-risk and downstream study, Research Gate 9)

## 8. Change procedure

Changing a manipulation, artifact, measurement, or the hidden definitions
requires `structural-risk-v2` (and `hidden-stress-v2` where applicable), an
ADR, and review by lanes A (policy on the failure bank) and D (hidden-test
integrity). Frozen failure-bank entries are immutable.
