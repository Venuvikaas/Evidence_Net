# Base Reconstruction Failure Catalogue (Phase 3)

- **Status:** Accepted (Phase 3).
- **Model:** Base Reconstruction (`configs/model/base-gate2.yaml`, 12 epochs).
- **Data:** 12 validation groups, seed 0 (`runs/catalogue-gate2/`).
- **Purpose:** Research Gate 2 requires failures to be exposed rather than
  hidden behind averages. This catalogue decomposes the per-pixel error of
  the Base Reconstruction by structural region type.

## Method

For each sample in the evaluation set:

1. The trained Base Reconstruction is applied to the 128x128 NoisyLR input.
2. The per-pixel absolute error `|prediction - target|` is computed on the
   256x256 grid.
3. Structural region masks are derived from the **target** edges (Sobel
   magnitude, normalized; binary threshold 0.5 per `docs/evaluation-protocol.md`):
   - `edge_band` — a 2-px dilation of the binary edge map.
   - `periodic` — high edge-density regions (magnitude >= 0.5).
   - `flat` — low-magnitude regions outside the edge band.
   - `all` — the whole image.
4. Mean absolute error is reported per region and per worst sample.

## Findings (12 validation groups, seed 0)

| Region | Mean MAE |
| --- | --- |
| Edge band | 0.084 |
| Periodic (high edge density) | 0.096 |
| Flat | 0.030 |
| All | 0.040 |

**Natural failure modes, in decreasing severity:**

1. **Periodic / textured regions** (MAE 0.096): dense, repeating structure
   is where the learned refinement helps least. Consistent with Phase 1's
   finding that alignment uncertainty is largest where local structure is
   rich — the model cannot know the exact phase to restore.
2. **Edge bands** (MAE 0.084): edges are the second-hardest region. The
   deterministic anchor already places edges close, but residual localization
   error concentrates exactly at boundaries (worst sample `002672` has
   edge-band MAE 0.100 vs overall 0.061).
3. **Flat regions** (MAE 0.030): smooth areas are restored well — the
   deterministic anchor plus a small learned correction is nearly sufficient.

**Worst samples** (overall MAE, all reported individually):

| Sample | Overall MAE | Edge-band MAE | Flat MAE |
| --- | --- | --- | --- |
| 001977 | 0.070 | 0.080 | 0.062 |
| 002672 | 0.061 | 0.100 | 0.036 |
| 002634 | 0.059 | 0.076 | 0.051 |

## Interpretation for later phases

- The failure signature (periodic > edge > flat) is the natural-error
  baseline that the **Bounded Detail Proposal** (Phase 4) must improve on in
  a targeted way; a proposal that merely repeats these errors adds no value.
- The catalogue is re-run per model version so later phases can attribute
  error changes to specific regions rather than to aggregate movement.
- These are *natural* failures of the trained model, not injected
  corruption; they are archived here for later stress testing
  (`docs/failures.md` conventions).

## Version history

| Version | Date | Change |
| --- | --- | --- |
| v1 | 2026-08-16 | Initial failure catalogue (Phase 3, Research Gate 2). |
