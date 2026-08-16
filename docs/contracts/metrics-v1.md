# Contract: metrics-v1

- **Name:** `metrics-v1`
- **Version:** v1
- **Status:** frozen
- **Frozen at:** Phase 4 handoff (Research Gate 3: continue, ADR-007)
- **Owner:** ALL lanes
- **Governed by:** `docs/evaluation-protocol.md`

## Purpose

Fix the metric definitions and statistical discipline so every lane reports
comparable numbers through the same trusted harness.

## Frozen fields

1. **Primary metrics.** PSNR (`data_range = 1.0`), SSIM (Wang 2004, 7x7
   Gaussian window, sigma 1.5, K1 0.01, K2 0.03, zero-padded), MAE — computed
   on `float64` copies, inputs clipped to `[0, 1]`.
2. **Secondary diagnostics.** Edge displacement (Sobel magnitude, threshold
   0.5, mean Chamfer distance bounded at 16 px), structural error
   (multi-scale edge-magnitude MAE over scales {1, 2}), frequency-band
   relative power differences (low `[0, 1/8)`, mid `[1/8, 1/2)`,
   high `[1/2, 1]` of Nyquist). Reported, not optimized against, until
   Phase 10/14 evidence decides their role.
3. **Statistical units.** Images / source groups are the statistical units.
   Pixels are never reported as independent sample counts. Aggregation uses
   the seeded group bootstrap in `evaluation/statistics.py`
   (`grouped_bootstrap_ci`, `aggregate_by_group` raising `GroupingError` on
   pixel-level inputs).
4. **Paired comparisons.** All models in a comparison run on identical paired
   sample sets; CIs are reported with `{mean, ci_lo, ci_hi, n_groups,
   n_boot}`.
5. **Edge cases.** Identical images: PSNR `inf`, MAE 0, SSIM 1, displacement
   0. Empty edge maps: displacement 0. Missing predicted edges: displacement
   = bounded radius (16 px) — a documented cap, not a silent failure.

## Implementation references

- Code: `src/evidence_net/evaluation/metrics.py`,
  `src/evidence_net/evaluation/statistics.py`
- Tests: `tests/unit/test_metrics.py`, `tests/unit/test_statistics.py`

## Change procedure

Changing a metric definition or the statistical discipline requires
`metrics-v2`, an ADR, a migration note, and a rerun decision for every
experiment that reports the metric. Reports must state the metric version.
