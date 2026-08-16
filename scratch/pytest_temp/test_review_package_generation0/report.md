# EVIDENCE-Net Technical Review Package — Run `run-review-01`

## 1. Provenance Record

| Component / Contract | Version Identifier |
| --- | --- |
| `dataset_manifest_hash` | `not-defined` |
| `base_model_version` | `base-model-v1` |
| `proposal_model_version` | `proposal-model-v1` |
| `support_definition_version` | `not-defined` |
| `calibration_version` | `not-defined` |
| `forward_model_version` | `not-defined` |
| `decision_policy_version` | `not-defined` |
| `pipeline_version` | `unified-inference-v1` |

## 2. Artifact Registry

| Artifact Name | Data Type | Shape | Value Range | SHA-256 Hash |
| --- | --- | --- | --- | --- |
| `input.npy` | `float32` | `[1, 64, 64]` | `[0.0, 1.0]` | `sha256:0` |

## 3. Quantitative Performance Metrics

### Stage: `base`

| Metric | Value | Unit |
| --- | --- | --- |
| `mae` | `0.05` | `pixel` |

