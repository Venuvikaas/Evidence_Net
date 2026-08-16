# EVIDENCE-Net Technical Review Package — Run `golden-loop-001`

## 1. Provenance Record

| Component / Contract | Version Identifier |
| --- | --- |
| `dataset_manifest_hash` | `sha256:1111222233334444` |
| `base_model_version` | `base-v1-golden` |
| `proposal_model_version` | `proposal-v1-golden` |
| `support_definition_version` | `not-defined` |
| `calibration_version` | `not-defined` |
| `forward_model_version` | `not-defined` |
| `decision_policy_version` | `not-defined` |
| `pipeline_version` | `unified-inference-v1` |

## 2. Artifact Registry

| Artifact Name | Data Type | Shape | Value Range | SHA-256 Hash |
| --- | --- | --- | --- | --- |
| `input.npy` | `float32` | `[1, 32, 32]` | `[0.0004370868264231831, 0.9992098212242126]` | `sha256:d222d353732fc5d911c19fe3857b93fc385d0c2e518afd6522393e9377af0b95` |
| `base.npy` | `float32` | `[1, 32, 32]` | `[0.0004370868264231831, 0.9992098212242126]` | `sha256:d222d353732fc5d911c19fe3857b93fc385d0c2e518afd6522393e9377af0b95` |
| `proposal.npy` | `float32` | `[1, 32, 32]` | `[0.0, 0.0]` | `sha256:ad7facb2586fc6e966c004d7d1d16b024f5805ff7cb47c7a85dabd8b48892ca7` |
| `candidate.npy` | `float32` | `[1, 32, 32]` | `[0.0004370868264231831, 0.9992098212242126]` | `sha256:d222d353732fc5d911c19fe3857b93fc385d0c2e518afd6522393e9377af0b95` |
| `final.npy` | `float32` | `[1, 32, 32]` | `[0.0004370868264231831, 0.9992098212242126]` | `sha256:d222d353732fc5d911c19fe3857b93fc385d0c2e518afd6522393e9377af0b95` |

## 3. Quantitative Performance Metrics

### Stage: `base`

| Metric | Value | Unit |
| --- | --- | --- |
| `psnr` | `7.995787786270502` | `-` |
| `ssim` | `0.1432201468549359` | `-` |
| `mae` | `0.3233883269469118` | `-` |
| `edge_displacement_px` | `1.2629310344827587` | `-` |
| `structural_error` | `0.1486380755064889` | `-` |
| `frequency_bands` | `{'[0.000,0.125)': -0.06877909854090462, '[0.125,0.500)': -0.08644760968444644, '[0.500,1.000)': -0.034988250934805266}` | `-` |

### Stage: `candidate`

| Metric | Value | Unit |
| --- | --- | --- |
| `psnr` | `7.995787786270502` | `-` |
| `ssim` | `0.1432201468549359` | `-` |
| `mae` | `0.3233883269469118` | `-` |
| `edge_displacement_px` | `1.2629310344827587` | `-` |
| `structural_error` | `0.1486380755064889` | `-` |
| `frequency_bands` | `{'[0.000,0.125)': -0.06877909854090462, '[0.125,0.500)': -0.08644760968444644, '[0.500,1.000)': -0.034988250934805266}` | `-` |

### Stage: `final`

| Metric | Value | Unit |
| --- | --- | --- |
| `psnr` | `7.995787786270502` | `-` |
| `ssim` | `0.1432201468549359` | `-` |
| `mae` | `0.3233883269469118` | `-` |
| `edge_displacement_px` | `1.2629310344827587` | `-` |
| `structural_error` | `0.1486380755064889` | `-` |
| `frequency_bands` | `{'[0.000,0.125)': -0.06877909854090462, '[0.125,0.500)': -0.08644760968444644, '[0.500,1.000)': -0.034988250934805266}` | `-` |

