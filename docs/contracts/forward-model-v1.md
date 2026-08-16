# Contract: forward-model-v1

- **Name:** `forward-model-v1`
- **Version:** v1
- **Status:** draft (promotes to frozen after Research Gate 6)
- **Owner:** Lane B (diagnostics); reviewed by A and C at promotion
- **Governed by:** `docs/modality-contract.md` sections 2 and 10,
  `docs/tensor-contract.md`, `docs/evaluation-protocol.md`

## Purpose

Fix the bounded, modality-specific forward operator family used by the
measurement-consistency diagnostic (Phase 7). The diagnostic re-degrades a
restored output through this family and reports how consistently the output
reproduces the observed degraded input. This is **compatibility, not truth**:
the family is a declared set of plausible degradation operators with bounded
parameters; it does not identify the true hidden degradation, and it never
proves that restored detail physically existed.

## 1. Scope and grid

- Clean / restored images live on the output grid (256x256 for the official
  dataset). Degraded observations live on the input grid (128x128). The
  forward family maps output grid -> input grid with a fixed scale factor of
  `2` (matching the official 128 -> 256 relationship).
- Tensors follow `tensor-v1`: channel-first `(1, H, W)` or plain `(H, W)`
  float32/float64 working arrays; the operator family operates on the
  `(H, W)` plane.

## 2. Operator family (frozen definitions)

| Operator | Kind | Operation order | Parameters (bounded) |
| --- | --- | --- | --- |
| `bilinear` | deterministic | bilinear 2x down-sample (align_corners=False) | scale = 2 |
| `area` | deterministic | 2x2 mean-pool down-sample | scale = 2 |
| `blur` | deterministic | Gaussian blur -> bilinear 2x down-sample | `blur_sigma` in `[0.0, 2.0]` px |
| `noisy-blur` | stochastic | Gaussian blur -> bilinear 2x down-sample -> additive Gaussian noise | `blur_sigma` in `[0.0, 2.0]` px, `noise_sigma` in `[0.0, 0.1]` (images in `[0, 1]`) |

- The **operation order** is part of the operator definition (blur before
  down-sampling before noise). The true hidden degradation order of the
  official data is unknown and is **not claimed**; the family only declares
  these as the probed candidates (modality contract section 2).
- **Stochastic treatment:** stochastic operators sample additive noise from
  `N(0, noise_sigma^2)` with an explicit, seeded `numpy.random.Generator`.
  Every stochastic evaluation reports the seed used and the observed noise
  variance; results are reproducible under the same seed.
- **Parameter bounds** are validated at operator construction: out-of-bounds
  values raise `ForwardError` (misspecification is detected, never silently
  clamped for evaluation).

## 3. Compatibility report (frozen semantics)

- For each operator `op` in the family and each image `i` with observation
  `y_i` and restored output `b_i` (frozen Base output or candidate):
  - `re_degrade_i = op(b_i)` on the input grid.
  - `residual_i = re_degrade_i - y_i`.
  - Per-image MAE and mean signed error (bias) are computed.
- Aggregation follows `metrics-v1`: images / source groups are the
  statistical units; per-operator MAE is aggregated with the seeded group
  bootstrap CI (`evaluation/statistics.py`). Pixels are never sample counts.
- The report shows the **residual distribution across the operator family**
  (min / median / max of per-operator means and the arg-min operator), never
  the minimum alone, so the diagnostic cannot be gamed by picking the
  friendliest operator.
- The report always carries the interpretation: the measurement-consistency
  diagnostic measures compatibility with the declared bounded family; it is
  not evidence that the true degradation was identified, and it does not
  certify restored detail (Gate 6 keeps it only if it adds held-out value or
  independently useful review information).

## 4. Non-identifiability (frozen threat model)

- Different clean candidates can re-degrade to nearly indistinguishable
  observations under members of this family; the family cannot distinguish
  them. Canonical examples: high-frequency stripe patterns (period 2) under
  `blur`/`noisy-blur`, and a thin line present vs absent under strong blur.
- Degradation order and parameter composition are not identifiable from a
  single observation.
- These cases are the Phase 7 stress suite and feed Phase 10's
  observation-ambiguity program. The diagnostic reports them as limitations,
  never as resolvable by the family.

## 5. Prohibited claims

- No claim that the family identifies the true degradation order or kernel.
- No claim that consistency implies correctness or physical existence of
  restored detail.
- No use of a single operator's minimum residual as a standalone
  trust score.

## 6. Implementation references

- Code: `src/evidence_net/stress_tests/forward.py`,
  `src/evidence_net/stress_tests/consistency.py`
- Config: `configs/modality/forward-v1.yaml`
- Script: `scripts/measure_consistency.py`
- Tests: `tests/numerical/test_forward.py`, `tests/numerical/test_consistency.py`
- Experiment: EXP-005 (incremental-value question, Research Gate 6)

## 7. Change procedure

Changing the operator family, bounds, operation order, or report semantics
requires `forward-model-v2`, an ADR, and review by lanes A (benefit features
consume consistency residuals) and C (review UI renders the diagnostic).
Unproven operators stay disabled by default (Integration II).
