/**
 * Typed API Client for EVIDENCE-Net (Phase 13).
 */

export interface Provenance {
  dataset_manifest_hash: string;
  base_model_version: string;
  proposal_model_version: string;
  support_definition_version: string;
  calibration_version: string;
  forward_model_version: string;
  decision_policy_version: string;
  pipeline_version: string;
}

export interface ArtifactMeta {
  dtype: string;
  shape: number[];
  range: number[];
  hash: string;
}

export interface RestorationResponse {
  run_id: string;
  status: string;
  provenance: Provenance;
  metrics: Record<string, any>;
  artifacts: Record<string, ArtifactMeta>;
  run_dir: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
}

const API_BASE = "/api/v1";

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function runRestorationInference(
  inputValues?: number[][],
  hasTarget = false
): Promise<RestorationResponse> {
  const res = await fetch(`${API_BASE}/restoration`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input_values: inputValues,
      has_target: hasTarget,
    }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.message || "Restoration inference failed");
  }
  return res.json();
}

export async function recordReviewEvent(
  runId: string,
  actionType: string,
  decisionReason: string
): Promise<any> {
  const res = await fetch(`${API_BASE}/review/events`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      run_id: runId,
      action_type: actionType,
      decision_reason: decisionReason,
    }),
  });
  if (!res.ok) throw new Error("Failed to record review event");
  return res.json();
}
