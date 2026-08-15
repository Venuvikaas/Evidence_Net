# Modality Contract v0 (Initial)

- **Status:** Draft — expected to be revised after Phase 1 data intake.
- **Version:** v0 (initial, pre-dataset)
- **Supersedes:** —
- **Governed by:** `EXECUTION.md` Phase 0 (contract) and Phase 1 (data
  registration). Per ADR-001, this contract is revised once the official local
  datasets are audited.

## Purpose

This document records the current positioning, imaging assumptions, known
unknowns, and claim boundaries of EVIDENCE-Net. It is a versioned contract:
before code or experiments rely on a modality assumption, the assumption must
appear here.

## 1. Current positioning

The first validated release targets **semiconductor-like structural imagery**
containing fine boundaries, repeated patterns, narrow lines, isolated points,
and defect-like structures. The product is a **research platform** for
evidence-aware selective restoration; it is not marketed as an industrial
semiconductor solution until one concrete imaging modality, acquisition
process, and real validation dataset are established (ADR-001).

Inputs are expected to be degraded "Noisy LR" images: low-resolution captures
affected by mixed degradations. The exact degradation family is configured and
validated for the selected modality rather than assumed to be universal.

## 2. Image-formation assumptions

The following assumptions are **candidate** until Phase 1/2 evidence confirms
them from the official `train/` data:

- The degraded observation `y` is related to an underlying clean image `x`
  through a bounded forward family that may include additive Gaussian noise,
  signal-dependent or multiplicative noise, blur, down-sampling, and clipping.
- The operation order of the degradation steps is **uncertain**; the system
  must not claim to identify the exact hidden degradation order.
- The clean target `x` is a fidelity-oriented reference produced by the
  official dataset pipeline; its construction, alignment method, and residual
  uncertainty must be established in Phase 1 before it is used as a training
  target.

## 3. Sensor and acquisition characteristics

**Unknown.** No sensor model, pixel size, physical scale, bit depth, or
acquisition process is assumed before the official data is inspected. Phase 1
records whatever provenance information is available in `docs/data-card.md`.

## 4. Expected degradation sources

Candidate sources to be confirmed from data:

- Additive Gaussian noise.
- Signal-dependent / multiplicative noise.
- Blur (kernel unknown).
- Down-sampling (low-resolution input).
- Clipping / saturation.
- Unknown composition order of the above.

## 5. Sampling and reconstruction pipeline

- Inputs are read from the official local directories; the raw tensor and its
  metadata are preserved (see `docs/tensor-contract.md`).
- Derived views (robust normalization, gradients, local statistics, frequency
  components) are transformations of the same measurement, never independent
  evidence.

## 6. Pixel size and physical scale

**Unknown.** Recorded in the data card once the official data supplies it.
Structural metrics must be defined in spatial units (pixels or patches) that
do not depend on unverified physical calibration.

## 7. Typical structure and defect classes

Candidate classes for the structural-risk program (Phase 10):

- Narrow lines and boundaries.
- Repeated / periodic patterns.
- Isolated points and defect-like structures.
- Holes, bridges, merges, and splits.
- Regions where degradation removes information that the prior may hallucinate.

## 8. Target construction and alignment

**Unknown until Phase 1.** The data card must state what the clean target
actually represents (real, synthetic, simulated, averaged, or expert-selected),
how it was aligned to the input, and the residual alignment uncertainty. No
pairing rule is invented; it is discovered and documented from the official
`train/` structure.

## 9. Permitted decision uses

- Technical review of restoration outputs by imaging and validation engineers.
- Selective gating: accept, attenuate, reject, or abstain per region.
- Downstream inspection / measurement workflows **only** where the applicable
  calibration domain and policy are documented.
- EVIDENCE-Net never presents learned confidence or forward compatibility as
  physical proof.

## 10. Known non-identifiability cases

- Different clean candidates can re-degrade to nearly indistinguishable
  observations; the forward model cannot distinguish them.
- Degradation order and parameter composition are generally not identifiable
  from a single observation.
- Rare structures may be indistinguishable from noise at low resolution.

These cases are probed by the observation-ambiguity and forward-misspecification
suites (Phases 7 and 10).

## 11. Acceptable error and abstention policies

- Acceptable error bounds, critical regions, and abstention behavior are
  **not yet defined**; they are pre-declared per experiment in
  `EXPERIMENTS.md` before final testing.
- Abstention marks a region unresolved; falling back to the Base output never
  implies the Base output is verified.

## 12. Claim boundaries (non-goals)

From the product definition (Section 3.3) and ADR-004:

- No perfect recovery of destroyed information.
- No zero-hallucination claim.
- No exact identification of hidden degradation order.
- No physical proof that every accepted detail existed.
- No universally calibrated score across sensors or datasets.
- No replacement of expert review for high-impact decisions.

## 13. Known unknowns

- Exact target semantics and pairing rules of the official `train/` dataset.
- Alignment uncertainty between noisy input and clean target.
- Whether the degradation family is homogeneous across the dataset or
  structured by source group.
- Whether `Test_NoisyLR/` targets exist, are absent, or are separately
  governed (recorded, never used for development decisions).
- Whether calibration survives held-out source and degradation groups.
- Whether any diagnostic (measurement consistency, stability, familiarity)
  adds incremental held-out value.

## 14. Version history

| Version | Date | Change |
| --- | --- | --- |
| v0 | 2026-08-15 | Initial contract before official dataset intake. |
