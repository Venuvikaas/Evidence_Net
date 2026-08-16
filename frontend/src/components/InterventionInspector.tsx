import React from "react";
import { Sliders, PlusCircle, MinusCircle, CheckCircle2 } from "lucide-react";

export const InterventionInspector: React.FC = () => {
  return (
    <div style={{ padding: 24, display: "flex", flexDirection: "column", gap: 24, flex: 1, overflowY: "auto" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <h2 style={{ fontSize: "1.2rem", fontWeight: 700 }}>Intervention Inspector (Bounded Detail Proposal)</h2>
          <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginTop: 4 }}>
            Analyze positive (+d) and negative (-d) proposal interventions against the Base Reconstruction.
          </p>
        </div>

        <button className="btn-primary">
          <Sliders size={16} />
          Explore Policy Thresholds
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        {/* Positive Intervention */}
        <div className="pane-card">
          <div className="pane-header">
            <div className="pane-title" style={{ color: "var(--accent-emerald)" }}>
              <PlusCircle size={18} />
              Positive Proposals (d⁺ &gt; 0)
            </div>
            <span className="badge-tag">Constructive Evidence</span>
          </div>

          <div style={{ height: 180, background: "rgba(16, 185, 129, 0.05)", border: "1px dashed rgba(16, 185, 129, 0.2)", borderRadius: 8, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8 }}>
            <span style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--accent-emerald)", fontFamily: "var(--font-mono)" }}>+0.042 max</span>
            <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Fills missing line structural continuity</span>
          </div>

          <div className="metric-row">
            <span>Spatial Coverage: 12.4%</span>
            <span>Energy: 0.018</span>
          </div>
        </div>

        {/* Negative Intervention */}
        <div className="pane-card">
          <div className="pane-header">
            <div className="pane-title" style={{ color: "var(--accent-rose)" }}>
              <MinusCircle size={18} />
              Negative Proposals (d⁻ &lt; 0)
            </div>
            <span className="badge-tag">Suppressive Evidence</span>
          </div>

          <div style={{ height: 180, background: "rgba(244, 63, 94, 0.05)", border: "1px dashed rgba(244, 63, 94, 0.2)", borderRadius: 8, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8 }}>
            <span style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--accent-rose)", fontFamily: "var(--font-mono)" }}>-0.038 max</span>
            <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Suppresses high-frequency speckle artifact</span>
          </div>

          <div className="metric-row">
            <span>Spatial Coverage: 8.1%</span>
            <span>Energy: 0.012</span>
          </div>
        </div>
      </div>

      {/* Safety Gate Rule Verification */}
      <div className="control-card" style={{ borderLeft: "4px solid var(--accent-cyan)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <CheckCircle2 size={20} color="var(--accent-cyan)" />
          <h4 style={{ fontSize: "0.95rem", fontWeight: 600 }}>Proposal Boundedness & Headroom Audit</h4>
        </div>
        <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
          Detail proposal $d$ is bounded within contract bounds $\left[-0.10, +0.10\right]$. Oracle gating verifies positive headroom gain over Base Reconstruction.
        </p>
      </div>
    </div>
  );
};
