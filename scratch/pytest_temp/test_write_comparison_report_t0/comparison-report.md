# Baseline comparison report (validation split)

- Samples: 2 paired groups (`000000, 000001`)
- Statistical unit: source group (image); CIs are 95% group bootstraps, never pixel counts.

| restorer | PSNR (dB) | SSIM | MAE | edge displacement (px) | structural error |
| --- | --- | --- | --- | --- | --- |
| deterministic | inf [nan, nan] | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |

Frequency-band relative power differences (per restorer):

- `deterministic`: {'frequency_bands.[0.000,0.125)': '+0.0000', 'frequency_bands.[0.125,0.500)': '+0.0000', 'frequency_bands.[0.500,1.000)': '+0.0000'}

## Panel order in comparison sheets
Each `artifacts/comparison-<index>.png` shows five panels left to right: input, output, target, error, edges.
