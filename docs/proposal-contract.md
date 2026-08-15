# Detail Proposal Contract v1 (Phase 4)

- **Status:** Draft — established before the proposal is trained (Phase 4).
- **Version:** v1
- **Governed by:** `EXECUTION.md` Phase 4 (Bounded Detail Proposal and Oracle
  Study) and `docs/product-definition.md` sections 8.1, 8.3, 10.3.

This document fixes the proposal tensor, the fusion rule, the oracle study
units, and the structural summaries. Code that produces or consumes proposals
must conform; changes require a decision-log entry and version increment.

## 1. Tensors and grids

- The Base Reconstruction `b`, the detail proposal `d`, the ungated candidate
  `c`, and the final gated reconstruction `x̂` all live on the **output grid**
  (256×256 for the official dataset), single channel, `float32`, values in
  `[0, 1]` for image tensors.
- The proposal is a **residual** on that same grid. Its signed values may
  leave `[0, 1]`; only the composed image tensors are clamped.
- Masks and gate maps are `uint8` (values `{0, 1}`) or `float32` (values in
  `[0, 1]`) on the output grid.

## 2. Bounded proposal parameterization

The proposal is amplitude-bounded by construction:

```
d = α · tanh(h_d(f(y), b))
```

where `y` is the degraded input, `b` is the frozen Base output, `f(y)` is a
feature map of the input, `h_d` is the proposer head, and `α` is the
configured amplitude bound. Consequently:

```
|d| ≤ α          elementwise
```

- `α` is a model configuration value (`model.amplitude`), validated `> 0`.
- The base model is **frozen** for proposal training: the proposer never
  receives gradients from the Base parameters (stop-gradient on `b`).

## 3. Fusion rule

The ungated candidate and the gated reconstruction are:

```
c  = b + d                  (ungated candidate, gate g = 1)
x̂  = b + g · d             (gated, gate g ∈ [0, 1])
```

- `g = 0` returns exactly the Base (`x̂ = b`); `g = 1` returns exactly the
  candidate (`x̂ = c`). These are the **fusion identities** and are tested.
- For the Phase 4 oracle study, `g` is a binary per-pixel or per-patch
  decision computed from ground truth. A continuous policy `π` replaces it
  from Phase 5 onward.

## 4. Oracle study units

- **Pixel decision:** accept the proposal at pixel `p` when the proposal
  strictly reduces the pixel absolute error vs the target:
  `|c_p - x_p| < |b_p - x_p|`; ties and increases are rejected.
- **Patch decision:** the output grid is partitioned into fixed
  `PATCH_SIZE × PATCH_SIZE` patches (`PATCH_SIZE = 16`, frozen for the
  official 256×256 grid). A patch `r` is accepted when the patch-level MAE of
  the candidate is strictly lower than that of the Base:
  `MAE_r(c, x) < MAE_r(b, x)`.
- **Coverage:** fraction of pixels/patches accepted.
- **Risk:** fraction of pixels/patches where accepting the proposal increases
  error (the oracle rejects these; an ungated system would take the harm).
- Both the patch grid and the pixel grid are evaluated; the patch grid is the
  primary region unit because it matches the Phase 5 benefit event.

## 5. Structural summaries

For a proposal `d`, Base `b`, and candidate `c`, the following summaries are
computed per image:

- **Magnitude:** mean and max of `|d|`, and relative mean magnitude
  `mean(|d|) / (range(b) + ε)`.
- **Edge:** mean and max of the normalized edge magnitude of `d`
  (Sobel gradient magnitude, same normalization as the evaluation metrics).
- **Multi-scale energy:** relative power of `d` per frequency band
  (same bands as `evaluation-protocol.md`), i.e. where the proposal adds or
  removes energy.
- **Structural change:** `edge_displacement(b, c)` (px, capped), `ssim(b, c)`,
  and the edge-magnitude difference `mean(|∇b| - |∇c|)` between Base and
  candidate.

## 6. What the contract does not claim

- The proposal does **not** claim that proposed detail physically existed in
  the target; it is the learned prior's best residual under the training
  objective.
- Oracle decisions are a **study tool** (they see ground truth); they are
  never used at inference. Phase 5 builds a predictor that estimates the
  probability that the oracle would accept.
- Benefit is measured on declared outcomes (PSNR / SSIM / MAE / structural
  error), not on visual appeal.
