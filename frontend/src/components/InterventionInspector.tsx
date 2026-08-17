import React, { useEffect, useState } from "react";
import { Sliders, PlusCircle, MinusCircle, CheckCircle2, AlertCircle } from "lucide-react";
import { RestorationResponse, fetchArtifact } from "../api/client";

interface InterventionInspectorProps {
  restorationData: RestorationResponse | null;
  onExplorePolicy?: () => void;
}

interface ProposalStats {
  maxPositive: number;
  maxNegative: number;
  positiveCoverage: number; // fraction of pixels with d > +threshold
  negativeCoverage: number; // fraction of pixels with d < -threshold
  energy: number; // mean |d|
}

const AMPLITUDE = 0.1; // proposal contract bound |d| <= alpha (proposal-contract-v1)
const COVERAGE_THRESHOLD = 0.005;

export const InterventionInspector: React.FC<InterventionInspectorProps> = ({
  restorationData,
  onExplorePolicy,
}) => {
  const [stats, setStats] = useState<ProposalStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!restorationData) {
      setStats(null);
      setError(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const payload = await fetchArtifact(restorationData.run_id, "proposal.npy");
        const values = payload.values;
        const flat = values.flat();
        const maxPositive = Math.max(...flat);
        const maxNegative = Math.min(...flat);
        let positiveCount = 0;
        let negativeCount = 0;
        let energySum = 0;
        for (const v of flat) {
          if (v > COVERAGE_THRESHOLD) positiveCount += 1;
          else if (v < -COVERAGE_THRESHOLD) negativeCount += 1;
          energySum += Math.abs(v);
        }
        const total = flat.length || 1;
        if (!cancelled) {
          setStats({
            maxPositive,
            maxNegative,
            positiveCoverage: positiveCount / total,
            negativeCoverage: negativeCount / total,
            energy: energySum / total,
          });
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [restorationData]);

  return (
    <div style={{ padding: 24, display: "flex", flexDirection: "column", gap: 24, flex: 1, overflowY: "auto" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <h2 style={{ fontSize: "1.2rem", fontWeight: 700 }}>Intervention Inspector (Bounded Detail Proposal)</h2>
          <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginTop: 4 }}>
            Positive (+d) and negative (−d) proposal interventions against the Base Reconstruction, computed from
            the current run's <code>proposal.npy</code> artifact.
          </p>
        </div>

        {onExplorePolicy && (
          <button className="btn-primary" onClick={onExplorePolicy}>
            <Sliders size={16} />
            Explore Policy Thresholds
          </button>
        )}
      </div>

      {error && (
        <div className="control-card" style={{ borderLeft: "4px solid var(--accent-rose)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, color: "var(--accent-rose)", fontSize: "0.85rem" }}>
            <AlertCircle size={18} />
            Could not load proposal artifact: {error}
          </div>
        </div>
      )}

      {!restorationData ? (
        <div className="pane-card">
          <div className="unavailable-state" style={{ minHeight: 160 }}>
            <AlertCircle size={24} color="var(--text-dim)" />
            <span>Run an inference first to inspect the real proposal interventions.</span>
          </div>
        </div>
      ) : !stats ? (
        <div className="pane-card">
          <div className="unavailable-state" style={{ minHeight: 160 }}>
            <span>Loading proposal artifact…</span>
          </div>
        </div>
      ) : (
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
              <span style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--accent-emerald)", fontFamily: "var(--font-mono)" }}>
                +{stats.maxPositive.toFixed(4)}
              </span>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                Largest positive intervention in this run (contract bound +{AMPLITUDE.toFixed(2)})
              </span>
            </div>

            <div className="metric-row">
              <span>Spatial Coverage: {(stats.positiveCoverage * 100).toFixed(1)}%</span>
              <span>Energy (mean |d|): {stats.energy.toFixed(4)}</span>
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
              <span style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--accent-rose)", fontFamily: "var(--font-mono)" }}>
                {stats.maxNegative.toFixed(4)}
              </span>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                Largest negative intervention in this run (contract bound −{AMPLITUDE.toFixed(2)})
              </span>
            </div>

            <div className="metric-row">
              <span>Spatial Coverage: {(stats.negativeCoverage * 100).toFixed(1)}%</span>
              <span>Bound check: {Math.max(Math.abs(stats.maxPositive), Math.abs(stats.maxNegative)) <= AMPLITUDE ? "within ±0.10" : "EXCEEDS ±0.10"}</span>
            </div>
          </div>
        </div>
      )}

      {/* Safety Gate Rule Verification */}
      <div className="control-card" style={{ borderLeft: "4px solid var(--accent-cyan)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <CheckCircle2 size={20} color="var(--accent-cyan)" />
          <h4 style={{ fontSize: "0.95rem", fontWeight: 600 }}>Proposal Boundedness &amp; Headroom Audit</h4>
        </div>
        <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
          Detail proposal <code>d</code> is bounded within contract bounds [−0.10, +0.10] (proposal-contract-v1).
          Oracle gating verified positive headroom over Base in EXP-004; the promoted policy default-accepts the
          candidate and reports unresolved regions separately (ADR-010).
        </p>
      </div>
    </div>
  );
};
