import React, { useEffect, useMemo, useState } from "react";
import { Sliders, ShieldCheck, AlertCircle } from "lucide-react";
import { RestorationResponse, fetchArtifact } from "../api/client";

interface PolicyExplorerProps {
  restorationData: RestorationResponse | null;
}

/** Frozen decision-policy-v1 parameters (docs/contracts/decision-policy-v1.md). */
const FROZEN_POLICY = {
  version: "policy-v1",
  accept_threshold: 0.75,
  reject_threshold: 0.35,
  unresolved_edge_density: 0.35,
  note: "Promoted configuration is default-accept + unresolved abstention (ADR-010); benefit scores are a ranking signal only.",
};

export const PolicyExplorer: React.FC<PolicyExplorerProps> = ({ restorationData }) => {
  const [benefitScores, setBenefitScores] = useState<number[] | null>(null);
  const [unresolvedFraction, setUnresolvedFraction] = useState<number | null>(null);
  const [acceptFraction, setAcceptFraction] = useState<number | null>(null);
  const [threshold, setThreshold] = useState(0.005);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!restorationData) {
      setBenefitScores(null);
      setUnresolvedFraction(null);
      setAcceptFraction(null);
      setError(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const scorePayload = await fetchArtifact(restorationData.run_id, "proposal_benefit.npy");
        const scores = scorePayload.values.flat();
        let unresolved: number | null = null;
        if (restorationData.artifacts["unresolved.npy"]) {
          const u = await fetchArtifact(restorationData.run_id, "unresolved.npy");
          const flatU = u.values.flat();
          unresolved = flatU.filter((v) => v >= 0.5).length / (flatU.length || 1);
        }
        let accept: number | null = null;
        if (restorationData.artifacts["decision_map.npy"]) {
          const d = await fetchArtifact(restorationData.run_id, "decision_map.npy");
          const flatD = d.values.flat();
          accept = flatD.filter((v) => v >= 0.5).length / (flatD.length || 1);
        }
        if (!cancelled) {
          setBenefitScores(scores);
          setUnresolvedFraction(unresolved);
          setAcceptFraction(accept);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
          setBenefitScores([]);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [restorationData]);

  const maxScore = benefitScores ? Math.max(...benefitScores) : 0;

  const coverageAt = (t: number) => {
    if (!benefitScores || benefitScores.length === 0) return null;
    return benefitScores.filter((v) => v >= t).length / benefitScores.length;
  };

  const curve = useMemo(() => {
    if (!benefitScores) return [];
    const points: Array<{ t: number; coverage: number }> = [];
    const steps = 40;
    const hi = maxScore || 0.01;
    for (let i = 0; i <= steps; i++) {
      const t = (i / steps) * hi;
      points.push({ t, coverage: coverageAt(t) ?? 0 });
    }
    return points;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [benefitScores, maxScore]);

  const currentCoverage = coverageAt(threshold);

  return (
    <div style={{ padding: 24, display: "flex", flexDirection: "column", gap: 24, flex: 1, overflowY: "auto" }}>
      <div>
        <h2 style={{ fontSize: "1.2rem", fontWeight: 700 }}>Policy Threshold Explorer</h2>
        <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginTop: 4 }}>
          Coverage/risk trade-off over the current run's backend-computed benefit scores, against the frozen
          <code> decision-policy-v1 </code> configuration.
        </p>
      </div>

      {error && (
        <div className="control-card" style={{ borderLeft: "4px solid var(--accent-rose)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, color: "var(--accent-rose)", fontSize: "0.85rem" }}>
            <AlertCircle size={18} />
            {error}
          </div>
        </div>
      )}

      {!restorationData ? (
        <div className="pane-card">
          <div className="unavailable-state" style={{ minHeight: 160 }}>
            <Sliders size={24} color="var(--text-dim)" />
            <span>Run an inference first — the explorer operates on the run's real benefit scores.</span>
          </div>
        </div>
      ) : !benefitScores && !error ? (
        <div className="pane-card">
          <div className="unavailable-state" style={{ minHeight: 160 }}>
            <span>Loading run scores…</span>
          </div>
        </div>
      ) : !benefitScores || benefitScores.length === 0 ? (
        <div className="pane-card">
          <div className="unavailable-state" style={{ minHeight: 160 }}>
            <Sliders size={24} color="var(--text-dim)" />
            <span>
              <code>proposal_benefit.npy</code> not-defined for this run. Benefit scores are computed on the
              256×256 output grid (frozen checkpoints); this run's grid does not qualify.
            </span>
          </div>
        </div>
      ) : (
        <>
          <div className="control-card" style={{ borderLeft: "4px solid var(--accent-cyan)" }}>
            <h3 className="section-title">Frozen policy configuration (decision-policy-v1)</h3>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 20, fontSize: "0.85rem" }}>
              <div>Accept threshold: <code>{FROZEN_POLICY.accept_threshold}</code></div>
              <div>Reject threshold: <code>{FROZEN_POLICY.reject_threshold}</code></div>
              <div>Unresolved edge density: <code>{FROZEN_POLICY.unresolved_edge_density}</code></div>
              <div>Accept fraction (this run): <code>{(acceptFraction ?? 0) * 100}%</code></div>
              <div>Unresolved fraction (this run): <code>{(unresolvedFraction ?? 0) * 100}%</code></div>
            </div>
          </div>

          <div className="control-card">
            <h3 className="section-title">Coverage vs benefit-score threshold (this run)</h3>
            <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 12 }}>
              <Sliders size={16} color="var(--accent-cyan)" />
              <input
                type="range"
                min={0}
                max={Math.max(maxScore, 0.01)}
                step={Math.max(maxScore / 100, 0.0001)}
                value={Math.min(threshold, maxScore)}
                onChange={(e) => setThreshold(Number(e.target.value))}
                style={{ flex: 1 }}
              />
              <code style={{ fontSize: "0.8rem" }}>
                t = {threshold.toFixed(4)} → coverage {(currentCoverage ?? 0) * 100}%
              </code>
            </div>
            <svg viewBox="0 0 400 160" width="100%" height={160} role="img" aria-label="coverage curve">
              {curve.map((p, i) => {
                const x = (i / (curve.length - 1)) * 400;
                const y = 150 - p.coverage * 140;
                return i === 0 ? (
                  <path key="line" d={`M ${x} ${y} ${curve
                    .slice(1)
                    .map((q, j) => `L ${((j + 1) / (curve.length - 1)) * 400} ${150 - q.coverage * 140}`)
                    .join(" ")}`} fill="none" stroke="var(--accent-cyan)" strokeWidth={2} />
                ) : null;
              })}
              <circle
                cx={(threshold / Math.max(maxScore, 0.01)) * 400}
                cy={150 - (currentCoverage ?? 0) * 140}
                r={5}
                fill="var(--accent-emerald)"
              />
            </svg>
            <div style={{ fontSize: "0.75rem", color: "var(--text-dim)" }}>
              X = benefit ranking score threshold (residual-magnitude, labels-v2 margin event) · Y = patch coverage
              accepting scores ≥ threshold. The promoted policy does not gate on this score (ADR-010); the curve is
              exploratory only.
            </div>
          </div>

          <div className="control-card" style={{ borderLeft: "4px solid var(--accent-amber)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <ShieldCheck size={18} color="var(--accent-amber)" />
              <span style={{ fontSize: "0.85rem", fontWeight: 600 }}>Honest limitation</span>
            </div>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 6 }}>
              {FROZEN_POLICY.note} Risk curves per action require labeled evaluation data (recorded in governed run
              bundles, not served by this service); this explorer shows coverage only.
            </p>
          </div>
        </>
      )}
    </div>
  );
};
