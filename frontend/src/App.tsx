import React, { useState } from "react";
import {
  Layers,
  Sliders,
  ShieldCheck,
  Play,
  Crosshair,
  CheckCircle,
  AlertCircle,
} from "lucide-react";
import { RestorationWorkspace } from "./components/RestorationWorkspace";
import { InterventionInspector } from "./components/InterventionInspector";
import { ReliabilityWorkspace } from "./components/ReliabilityWorkspace";
import { RestorationResponse, runRestorationInference } from "./api/client";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"workspace" | "intervention" | "reliability">("workspace");
  const [restorationData, setRestorationData] = useState<RestorationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [lastRunId, setLastRunId] = useState<string | null>(null);
  const [hoverPixel, setHoverPixel] = useState<{ x: number; y: number; values: Record<string, number> } | null>(null);

  const [activeLayers, setActiveLayers] = useState<Record<string, boolean>>({
    benefit: true,
    consistency: false,
    stability: false,
    familiarity: false,
    decision: true,
    unresolved: false,
  });

  const handleRunInference = async () => {
    setLoading(true);
    setRunError(null);
    try {
      const data = await runRestorationInference();
      setRestorationData(data);
      setLastRunId(data.run_id);
    } catch (e) {
      // Surface the failure instead of silently keeping the synthetic demo:
      // a silent catch is exactly what made the button appear to do nothing.
      console.warn("API offline, using synthetic inference state", e);
      setRestorationData(null);
      setLastRunId(null);
      setRunError(
        "The EVIDENCE-Net API is not reachable, so the workspace keeps the synthetic demo. " +
          "Start the backend from the repository root (e.g. `python -m uvicorn evidence_net.api.app:app " +
          "--port 8000` or `bash scripts/demo_run.sh`), then click Run again."
      );
    } finally {
      setLoading(false);
    }
  };

  const toggleLayer = (layerKey: string) => {
    setActiveLayers((prev) => ({ ...prev, [layerKey]: !prev[layerKey] }));
  };

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="app-header">
        <div className="brand-logo">
          <div style={{ width: 12, height: 12, borderRadius: "50%", background: "var(--primary)", boxShadow: "var(--shadow-glow)" }} />
          EVIDENCE-Net
        </div>

        <nav className="header-tabs">
          <button
            className={`tab-btn ${activeTab === "workspace" ? "active" : ""}`}
            onClick={() => setActiveTab("workspace")}
          >
            <Layers size={16} />
            Restoration Workspace
          </button>

          <button
            className={`tab-btn ${activeTab === "intervention" ? "active" : ""}`}
            onClick={() => setActiveTab("intervention")}
          >
            <Sliders size={16} />
            Intervention Inspector
          </button>

          <button
            className={`tab-btn ${activeTab === "reliability" ? "active" : ""}`}
            onClick={() => setActiveTab("reliability")}
          >
            <ShieldCheck size={16} />
            Reliability & Provenance
          </button>
        </nav>

        <button className="btn-primary" onClick={handleRunInference} disabled={loading}>
          <Play size={16} />
          {loading ? "Executing Pipeline..." : "Run Unified Inference"}
        </button>
      </header>

      {/* Run status / error banner */}
      {runError && (
        <div className="status-banner status-error">
          <AlertCircle size={16} />
          <span>{runError}</span>
        </div>
      )}
      {!runError && lastRunId && (
        <div className="status-banner status-success">
          <CheckCircle size={16} />
          <span>
            Run <code>{lastRunId}</code> completed — the workspace now shows the real restored
            artifacts from the frozen Base + Proposal checkpoints.
          </span>
        </div>
      )}

      {/* Body Area */}
      <div className="app-body">
        {/* Left Sidebar Control Panel */}
        <aside className="sidebar-panel">
          <div>
            <h3 className="section-title">Reliability Layers</h3>
            <div className="control-card">
              <div className="toggle-group">
                {[
                  { key: "benefit", label: "Proposal Benefit Layer" },
                  { key: "consistency", label: "Measurement Consistency" },
                  { key: "stability", label: "Model Stability" },
                  { key: "familiarity", label: "Distribution Familiarity" },
                  { key: "decision", label: "Gated Decision Map" },
                  { key: "unresolved", label: "Unresolved-Region Mask" },
                ].map((item) => (
                  <label key={item.key} className="toggle-item">
                    <span>{item.label}</span>
                    <input
                      type="checkbox"
                      className="toggle-checkbox"
                      checked={activeLayers[item.key] || false}
                      onChange={() => toggleLayer(item.key)}
                    />
                  </label>
                ))}
              </div>
            </div>
          </div>

          <div>
            <h3 className="section-title">Pixel & Patch Inspector</h3>
            <div className="control-card">
              {hoverPixel ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 6, fontSize: "0.82rem", fontFamily: "var(--font-mono)" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--accent-cyan)" }}>
                    <Crosshair size={14} />
                    Coordinates: ({hoverPixel.x}, {hoverPixel.y})
                  </div>
                  <div>Input (y): {hoverPixel.values.input}</div>
                  <div>Base (b): {hoverPixel.values.base}</div>
                  <div>Proposal (d): +{hoverPixel.values.proposal}</div>
                  <div style={{ color: "var(--accent-emerald)", fontWeight: 600 }}>Final (o): {hoverPixel.values.final}</div>
                </div>
              ) : (
                <div style={{ color: "var(--text-dim)", fontSize: "0.8rem", textAlign: "center", padding: "10px 0" }}>
                  Hover any workspace canvas to inspect pixel intensities
                </div>
              )}
            </div>
          </div>

          <div>
            <h3 className="section-title">Contract Compliance</h3>
            <div className="control-card" style={{ fontSize: "0.8rem", color: "var(--text-muted)", gap: 8 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--accent-emerald)" }}>
                <CheckCircle size={14} />
                <code>artifacts-v1</code> frozen
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--accent-emerald)" }}>
                <CheckCircle size={14} />
                <code>error-and-optional-fields-v1</code> frozen
              </div>
            </div>
          </div>
        </aside>

        {/* Main Panel View Switch */}
        {activeTab === "workspace" && (
          <RestorationWorkspace
            restorationData={restorationData}
            activeLayers={activeLayers}
            onPixelHover={(x, y, values) => setHoverPixel({ x, y, values })}
          />
        )}

        {activeTab === "intervention" && <InterventionInspector />}

        {activeTab === "reliability" && (
          <ReliabilityWorkspace restorationData={restorationData} />
        )}
      </div>
    </div>
  );
};

export default App;
