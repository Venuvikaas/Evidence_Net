# Evaluation Protocol — Metric Contracts (v1)

- **Status:** Accepted (Phase 2).
- **Version:** v1
- **Governed by:** `EXECUTION.md` Phase 2 and Part 1 Sections 3, 8-9.

This document fixes the metric implementations, spatial units, and
statistical discipline of the evaluation harness. Every later model (Base,
proposal, gated) is judged through the same trusted harness.

## 1. Tensors and domain

- Evaluations compare a **predicted** restoration to the **target** (GT)
  reference on the same grid (256×256 after restoration of the 128×128
  inputs).
- Metrics are computed on `float64` copies; inputs are clipped to `[0, 1]`
  before comparison (targets are already in `[0, 1]`).
- PSNR uses `data_range = 1.0` (the targets are exactly within `[0, 1]`).

## 2. Metric contracts

Spatial unit is **per image** (single pair) unless stated otherwise; all
aggregation is grouped by image/source group (Section 4).

### Primary metrics

| Metric | Definition | Range | Notes |
| --- | --- | --- | --- |
| PSNR | `10 * log10(data_range**2 / MSE)` | `[0, ∞]`, ∞ if MSE = 0 | reported as `inf` for identical images |
| SSIM | Wang et al. 2004, 7×7 Gaussian window (σ=1.5), K1=0.01, K2=0.03, `data_range=1.0`, zero-padded boundaries | `[-1, 1]`, 1 = identical | implemented in `evaluation/metrics.py` |
| MAE | mean absolute difference | `[0, ∞)`, 0 = identical | |

### Secondary structural and frequency diagnostics

| Metric | Definition | Spatial unit |
| --- | --- | --- |
| Edge displacement | Sobel gradient magnitude, normalized to [0,1]; binary edges at threshold 0.5; mean Chamfer distance (4-neighborhood, bounded at 16 px) from target edges to predicted edges | pixels |
| Structural error | mean over scales {1, 2} of the mean absolute difference of normalized edge-magnitude maps at that scale | per image (multi-scale) |
| Frequency diagnostics | radial power-spectrum (|FFT|² via `rfft2`) relative difference per band: low `[0, 1/8)`, mid `[1/8, 1/2)`, high `[1/2, 1]` of Nyquist | per image |

All secondary diagnostics are **reported, not optimized against**, until
Phase 10/14 evidence decides their role.

## 3. Edge cases

- Identical images: PSNR = `inf`, MAE = 0, SSIM = 1, edge displacement = 0,
  structural error = 0, frequency differences = 0.
- Both edge maps empty: edge displacement = 0.
- Target edges exist but predicted has none: edge displacement = bounded
  radius (16 px) — a documented cap, not a silent failure.
- Frequency bands with zero target power: relative difference is reported as
  `0` to avoid division blow-ups (documented in code).

## 4. Statistical discipline

- **Images / source groups are the statistical units.** Pixels are never
  reported as independent sample counts.
- `evaluation/statistics.py` provides:
  - `grouped_bootstrap_ci`: percentile bootstrap (default 1000 resamples,
    seeded) resampling **groups** with replacement → `{mean, ci_lo, ci_hi,
    n_groups, n_boot}`.
  - `aggregate_by_group`: requires exactly one metric value per group id;
    raises `GroupingError` on pixel-level inputs (values outnumber groups) or
    on shape mismatches.
- Paired comparisons (same samples through different restorers) are always
  kept on identical sample sets.

## 5. Evaluation procedure (Phase 2+)

1. Load the frozen train manifest and the approved splits.
2. Run each restorer over the same evaluation sample (fixed seed sampling
   from the validation split at development time).
3. Compute per-image metrics, then grouped aggregates with CIs.
4. Write a run bundle: config, manifest, environment, per-group metrics,
   comparison artifacts, summary — per `docs/run-and-artifact-contract.md`.
5. Report primary endpoints with CIs and worst-group values; never report
   pixel counts as sample sizes.

## 6. Version history

| Version | Date | Change |
| --- | --- | --- |
| v1 | 2026-08-15 | Initial metric contracts (Phase 2). |
