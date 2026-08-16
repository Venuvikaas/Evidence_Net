# Downstream Validation Task (Phase 10, structural-risk-v1 section 5)

## The frozen downstream task

With no external downstream system available to the project yet, the frozen
downstream task is **measurement fidelity**: a restored output's support for
three structural measurements, compared with the target-derived values.

| Measurement | Definition | Unit |
| --- | --- | --- |
| `edge_displacement_px` | `edge_displacement(target, output)` (frozen in `metrics-v1`) | px |
| `edge_components` | 4-connected components of `binary_edges(output, 0.5)` | count |
| `bright_components` | 4-connected components of `output > 0.8` (lines / points) | count |

Downstream error per measurement is the absolute deviation of the output's
value from the target's value:
`|measurement(output) - measurement(target)|`.

## No co-training

The task is a **pure function** of outputs and targets
(`src/evidence_net/stress_tests/downstream.py`). It has no learned
parameters, is never trained, never reads the hidden stress definitions
(`data/stress/hidden-stress-v1.json`), and never touches `Test_NoisyLR/`.

## What is evaluated

For a paired sample, downstream error is aggregated per output type with the
grouped bootstrap discipline of `metrics-v1` (images / source groups are the
statistical units; pixels are never sample counts):

- `base` — the frozen Base Reconstruction (`base-output-v1`).
- `candidate` — the ungated candidate `b + d` (`proposal-output-v1`).
- `oracle-patch` — the oracle-patch output as a **study proxy** for
  selective restoration (the oracle sees ground truth; it is never used at
  inference). The gap between `candidate` and `oracle-patch` downstream
  error is the selective-restoration headroom measured downstream.

This answers "does selective restoration change downstream measurements"
without co-training the downstream task. When Lane A's Decision Policy is
promoted (Integration III), the policy actions replace the oracle proxy.

## Why this task

- It is reproducible with only frozen components already in the repository.
- It measures the consequence structural errors have on measurements that a
  reviewer would actually take from a restored image.
- It keeps downstream evidence separate from candidate, ambiguity,
  acquisition, and natural-failure evidence (Gate 9).

## Governed by

- `docs/contracts/structural-risk-v1.md` (threat models, section 5)
- `docs/evaluation-protocol.md` (grouped statistics)
- `docs/failures` ledger (natural failure bank feeds the case selection)
