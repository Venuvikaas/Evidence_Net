import React, { useEffect, useState } from "react";
import { AlertTriangle, ShieldCheck, Lock, AlertCircle } from "lucide-react";
import { fetchFrozenData } from "../api/client";

interface FailureCase {
  case_id: string;
  kind: string;
  region: string;
  effect: string;
  source_run: string;
  source_manifest?: string;
  review_note?: string;
}

export const FailureBrowser: React.FC = () => {
  const [cases, setCases] = useState<FailureCase[] | null>(null);
  const [bankMeta, setBankMeta] = useState<{ schema?: string; note?: string } | null>(null);
  const [stress, setStress] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const bank = await fetchFrozenData("failures");
        const stressDef = await fetchFrozenData("stress-definitions");
        if (!cancelled) {
          setCases(bank.cases ?? []);
          setBankMeta({ schema: bank.schema, note: bank.note });
          setStress(stressDef);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div style={{ padding: 24, display: "flex", flexDirection: "column", gap: 24, flex: 1, overflowY: "auto" }}>
      <div>
        <h2 style={{ fontSize: "1.2rem", fontWeight: 700 }}>Failure &amp; Stress Browser</h2>
        <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginTop: 4 }}>
          Frozen Gate 9 evidence: the natural-failure bank and hidden stress definitions (Phase 10). These are
          immutable, backend-served records — not computed from the current run.
        </p>
      </div>

      {error && (
        <div className="control-card" style={{ borderLeft: "4px solid var(--accent-rose)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, color: "var(--accent-rose)", fontSize: "0.85rem" }}>
            <AlertCircle size={18} />
            Could not load frozen records: {error}
          </div>
        </div>
      )}

      {/* Natural failure bank */}
      <div className="control-card">
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
          <AlertTriangle size={18} color="var(--accent-amber)" />
          <h3 className="section-title" style={{ marginBottom: 0 }}>
            Natural Failure Bank {bankMeta?.schema ? <code>({bankMeta.schema})</code> : ""}
          </h3>
          {bankMeta?.schema && <Lock size={12} color="var(--text-dim)" />}
        </div>
        <p style={{ fontSize: "0.78rem", color: "var(--text-dim)", marginBottom: 12 }}>{bankMeta?.note}</p>
        {cases && cases.length > 0 ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {cases.map((c) => (
              <div key={c.case_id} style={{ background: "rgba(0,0,0,0.3)", padding: "10px 14px", borderRadius: 6, border: "1px solid var(--border-color)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                  <code style={{ color: "var(--accent-rose)", fontWeight: 700 }}>{c.case_id}</code>
                  <span className="badge-tag">{c.kind}</span>
                  <span className="badge-tag">{c.region}</span>
                  {c.source_manifest && <span className="badge-tag">{c.source_manifest}</span>}
                </div>
                <div style={{ fontSize: "0.8rem", marginTop: 6 }}>{c.effect}</div>
                <div style={{ fontSize: "0.72rem", color: "var(--text-dim)", marginTop: 4 }}>
                  source: <code>{c.source_run}</code>
                  {c.review_note ? ` · ${c.review_note}` : ""}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="unavailable-state" style={{ minHeight: 100 }}>
            <span>No cases recorded.</span>
          </div>
        )}
      </div>

      {/* Hidden stress definitions */}
      <div className="control-card">
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
          <ShieldCheck size={18} color="var(--accent-cyan)" />
          <h3 className="section-title" style={{ marginBottom: 0 }}>
            Hidden Stress Definitions {stress?.schema ? <code>({stress.schema})</code> : ""}
          </h3>
          <Lock size={12} color="var(--text-dim)" />
        </div>
        {stress ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div>
              <div style={{ fontSize: "0.72rem", color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>hash (frozen)</div>
              <code style={{ fontSize: "0.75rem" }}>{stress.hash}</code>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 12 }}>
              {["perturbation", "acquisition"].map((group) => (
                <div key={group} style={{ background: "rgba(0,0,0,0.3)", padding: "10px 14px", borderRadius: 6, border: "1px solid var(--border-color)" }}>
                  <div style={{ fontSize: "0.75rem", fontWeight: 600, textTransform: "capitalize", marginBottom: 6 }}>{group}</div>
                  <div style={{ fontSize: "0.75rem", fontFamily: "var(--font-mono)", color: "var(--text-muted)", display: "flex", flexDirection: "column", gap: 3 }}>
                    {Object.entries(stress[group] ?? {}).map(([k, v]) => (
                      <div key={k} style={{ display: "flex", justifyContent: "space-between" }}>
                        <span>{k}</span>
                        <span style={{ color: "var(--accent-cyan)" }}>{String(v)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-dim)" }}>
              Seed: <code>{stress.seed}</code> — hidden definitions are frozen and isolated from training (Gate 9 test
              isolation).
            </div>
          </div>
        ) : (
          <div className="unavailable-state" style={{ minHeight: 100 }}>
            <span>No stress definitions served.</span>
          </div>
        )}
      </div>
    </div>
  );
};
