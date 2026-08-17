# EVIDENCE-Net Release Report (Phase 18, v1)

- **Version:** v1
- **Status:** frozen release report
- **Owner:** all developers (joint release gate)
- **Date:** 2026-08-16

---

## 1. Frozen candidate

| Component | Version / reference | Hash |
| --- | --- | --- |
| Base Reconstruction | `base-model-v1` | `checkpoints/train-base-gate2/best.pt` sha256 `3e5d2f94…` |
| Detail Proposal | `proposal-model-v1` | `checkpoints/train-proposal-gate3v2/best.pt` sha256 `524156ed…` |
| Support definition | `support-definition-v1` | contract in `docs/contracts/` |
| Calibration | `calibration-v1` | fit on calibration split only |
| Decision policy | `decision-policy-v1` | default-accept + unresolved abstention (ADR-010) |
| Pipeline | `unified-inference-v1` | `src/evidence_net/inference/pipeline.py` |
| Source manifest | `official-test-noisylr-source-v1.json` | `aab75186…` (400 inputs, isolated) |

## 2. Final evaluation protocol

- Inputs: every supported `Test_NoisyLR/` input exactly **once**
  (400 files, 128x128 float32, per `NoisyLR/000000.npy … 000399.npy`).
- Pipeline: frozen Base -> bounded proposal -> default-accept gated output
  (the promoted simplified policy from ADR-010; unresolved mask reported,
  never certifying the Base).
- Output mapping: original relative input names are preserved in the
  output manifest (`NoisyLR/<id>.npy`).
- No post-run model, calibration, policy, or threshold changes.

## 3. Results

- **Coverage:** 400 / 400 outputs, one per supported input, no extras —
  verified programmatically.
- **Output contract:** all outputs are `(256, 256)` `float32` in `[0, 1]`,
  matching `artifacts-v1` — verified programmatically.
- **Provenance:** source manifest hash, checkpoint hashes, semantic
  versions, and per-output SHA256 hashes are recorded in
  `runs/release-final-inference/release-report.json`.

## 4. Governance and gate decisions (Phases 5–10)

| Gate | Decision | Evidence |
| --- | --- | --- |
| 4 (benefit) | **Simplify** — predictor at chance; event is the norm (79.4% beneficial) | EXP-009, ADR-009, `runs/benefit-eval-gate4-real-v4/` |
| 5 (policy) | **Continue, simplified** — default-accept + unresolved abstention beats Base | EXP-010, ADR-010, `runs/policy-eval-gate5-real/` |
| 6 (consistency) | **Keep** | EXP-005, ADR-011, `runs/consistency-gate6-real/` |
| 7 (stability) | **Keep** | EXP-006, ADR-012, `runs/stability-gate7-real/` |
| 8 (familiarity) | **NOT promoted** (0% shift detection; rare-valid false warnings exceed cap) | EXP-007, ADR-013, `runs/familiarity-gate8-real/` |
| 9 (structural) | **Continue** | EXP-008, ADR-014, `runs/structural-gate9-real/` |
| 10 (human interpretation) | **Pending participants** — protocol + capture ready; no participants available | EXP-011, ADR-015 |

## 5. Phase 15 deployment parity (ONNX export)

- **Exported components:** the two promoted model heads only — Base
  Reconstruction (`base.onnx`) and the bounded Detail Proposal head
  (`proposal.onnx`), via `deploy/export_onnx.py`, on the frozen
  128x128 -> 256x256 grid with dynamic batch/spatial axes.
- **Decision parity validated** (`tests/decision_parity/test_onnx_parity.py`)
  between PyTorch and ONNX Runtime within 1e-5:
  - **tensor parity**: `b`, `d`, candidate, and final outputs match;
  - **spatial parity**: all outputs are 256x256 (frozen 2x up-scale);
  - **ranking parity**: the residual-magnitude benefit score map computed
    from ONNX tensors matches the PyTorch one;
  - **action parity**: the promoted decision gate map (default-accept with
    unresolved abstention, ADR-010) matches;
  - **abstention parity**: the unresolved-region mask matches.
- **Calibration parity**: honestly **not-defined** — the promoted pipeline
  records `calibration-v1` as a version but never serves a calibration
  tensor, so there is nothing to compare at inference time (documented in
  the parity test).
- **Integrity:** export fails loudly instead of writing a placeholder
  asset; the exported graphs are re-loaded and verified in the test.
- **TensorRT:** not added — deployment requirements do not justify it; the
  parity gate runs on CPU ONNX Runtime in CI (`.[dev,deploy]` extra).

## 6. Published failures, negative results, and limitations

- **Benefit prediction is not discriminative** on the frozen event: all
  predictors at chance (pooled AUC 0.48–0.59). The support-aware
  per-patch claim is not promoted (ADR-009).
- **Familiarity diagnostic fails Gate 8** on real data and is disabled by
  default (ADR-013).
- **Human interpretation is untested** — participants not available; the
  protocol and capture machinery are ready (ADR-015).
- **No local targets for `Test_NoisyLR/`**: final evaluation measures
  outputs only; no ground-truth metrics are reported for the final inputs.
- **No industrial modality validation, downstream labels, or expert user
  study** was available; the release is a research platform for
  semiconductor-like structural imagery with those gaps explicit.
- A harness bug (proposal checkpoint `forward` returning the candidate,
  double-adding `b + d`) was found and fixed during the governed runs;
  documented in FAILURES.md.

## 7. Integrity statement

- No post-evaluation tuning was performed after the frozen candidate was
  set.
- `Test_NoisyLR/` never entered training, validation, calibration, model
  selection, or policy tuning; it is referenced only by the isolated
  source manifest and this final inference run.
- All hashes recorded in `runs/release-final-inference/release-report.json`.
