import React, { useEffect, useRef, useState } from "react";
import { Eye, Layers, AlertCircle, ShieldCheck, RotateCcw } from "lucide-react";
import { RestorationResponse, ArtifactPayload, fetchArtifact } from "../api/client";

interface RestorationWorkspaceProps {
  restorationData: RestorationResponse | null;
  activeLayers: Record<string, boolean>;
  onPixelHover?: (x: number, y: number, values: Record<string, number>) => void;
}

const ARTIFACT_NAMES = ["input.npy", "base.npy", "proposal.npy", "candidate.npy", "final.npy"] as const;

/** Optional artifacts the backend may serve for a run (256x256 output grid). */
const OPTIONAL_ARTIFACTS = ["proposal_benefit.npy", "decision_map.npy", "unresolved.npy"] as const;

interface LayerSpec {
  title: string;
  badge: string;
  legend: string;
  artifact?: (typeof OPTIONAL_ARTIFACTS)[number];
}

const LAYER_SPECS: Record<string, LayerSpec> = {
  benefit: {
    title: "Proposal Benefit Layer",
    badge: "proposal_benefit.npy",
    artifact: "proposal_benefit.npy",
    legend:
      "Backend-computed residual-magnitude benefit ranking score on the labels-v2 margin event " +
      "(MAE(candidate) + 0.005 < MAE(base), per 16x16 patch, ADR-016). A ranking signal only — it never gates outputs.",
  },
  consistency: {
    title: "Measurement Consistency",
    badge: "measurement_consistency.npy",
    legend:
      "Not computed by this service. Consistency requires re-degrading outputs with the bounded operator family " +
      "(measure_consistency.py); it is labeled compatibility, never truth.",
  },
  stability: {
    title: "Model Stability",
    badge: "model_stability.npy",
    legend:
      "Not computed by this service. Stability requires perturbation and checkpoint comparisons " +
      "(measure_stability.py); agreement is stability, never probability of truth.",
  },
  familiarity: {
    title: "Distribution Familiarity",
    badge: "distribution_familiarity.npy",
    legend:
      "Not computed by this service. Familiarity-v2 (ADR-017) requires a calibrated reference population; " +
      "without one the layer is not-defined and must not gate actions.",
  },
  decision: {
    title: "Gated Decision Map",
    badge: "decision_map.npy",
    artifact: "decision_map.npy",
    legend:
      "Gate map of the promoted policy (decision-policy-v1, ADR-010): 1 = candidate emitted (default accept), " +
      "0 = Base fallback in unresolved patches. Reported as a layer; the released final output stays default-accept.",
  },
  unresolved: {
    title: "Unresolved-Region Mask",
    badge: "unresolved.npy",
    artifact: "unresolved.npy",
    legend:
      "Unresolved regions (patch edge density >= 0.35, EXP-004 periodic-region evidence). " +
      "Fallback to the Base never certifies it (kill-switch rule).",
  },
};

interface Viewport {
  scale: number;
  x: number;
  y: number;
}

export const RestorationWorkspace: React.FC<RestorationWorkspaceProps> = ({
  restorationData,
  activeLayers,
  onPixelHover,
}) => {
  const canvasRefs = {
    input: useRef<HTMLCanvasElement>(null),
    base: useRef<HTMLCanvasElement>(null),
    proposal: useRef<HTMLCanvasElement>(null),
    candidate: useRef<HTMLCanvasElement>(null),
    final: useRef<HTMLCanvasElement>(null),
  };
  const layerRefs = {
    benefit: useRef<HTMLCanvasElement>(null),
    decision: useRef<HTMLCanvasElement>(null),
    unresolved: useRef<HTMLCanvasElement>(null),
  };
  const gridRef = useRef<HTMLDivElement>(null);

  // Real artifacts fetched from the API run bundle (maps pane -> 2D grid).
  const [artifacts, setArtifacts] = useState<Record<string, number[][]> | null>(null);
  const [artifactRanges, setArtifactRanges] = useState<Record<string, [number, number]>>({});
  const [fetchError, setFetchError] = useState<string | null>(null);

  // Linked pan/zoom viewport applied to every canvas.
  const [viewport, setViewport] = useState<Viewport>({ scale: 1, x: 0, y: 0 });
  const dragRef = useRef<{ startX: number; startY: number; vx: number; vy: number } | null>(null);

  // Fetch the actual artifact tensors for the completed run.
  useEffect(() => {
    if (!restorationData) {
      setArtifacts(null);
      setFetchError(null);
      setViewport({ scale: 1, x: 0, y: 0 });
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const results: Record<string, number[][]> = {};
        const ranges: Record<string, [number, number]> = {};
        const names: string[] = [
          ...ARTIFACT_NAMES,
          ...OPTIONAL_ARTIFACTS.filter((name) => restorationData.artifacts[name]),
        ];
        for (const name of names) {
          const payload: ArtifactPayload = await fetchArtifact(restorationData.run_id, name);
          results[name] = payload.values;
          const flat = payload.values.flat();
          ranges[name] = [Math.min(...flat), Math.max(...flat)];
        }
        if (!cancelled) {
          setArtifacts(results);
          setArtifactRanges(ranges);
          setFetchError(null);
        }
      } catch (e) {
        if (!cancelled) setFetchError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [restorationData]);

  useEffect(() => {
    const width = 64;
    const height = 64;

    const drawPattern = (
      canvas: HTMLCanvasElement | null,
      type: "input" | "base" | "proposal" | "candidate" | "final",
      realValues?: number[][],
      realRange?: [number, number]
    ) => {
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      if (realValues && realValues.length > 0) {
        const h = realValues.length;
        const w = realValues[0].length;
        canvas.width = w;
        canvas.height = h;
        const imgData = ctx.createImageData(w, h);
        const data = imgData.data;
        const [lo, hi] = realRange ?? [0, 1];
        const span = hi - lo || 1.0;

        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const idx = (y * w + x) * 4;
            const val = realValues[y][x];
            if (type === "proposal") {
              // Divergent visualization: positive (red), negative (blue), zero (dark).
              const norm = (val - lo) / span;
              data[idx] = val > 0 ? Math.floor(norm * 255) : 0;
              data[idx + 1] = 120;
              data[idx + 2] = val < 0 ? Math.floor((1 - norm) * 255) : 200;
            } else {
              const byteVal = Math.floor(((val - lo) / span) * 255);
              data[idx] = byteVal;
              data[idx + 1] = byteVal;
              data[idx + 2] = byteVal;
            }
            data[idx + 3] = 255;
          }
        }
        ctx.putImageData(imgData, 0, 0);
        return;
      }

      // Synthetic fallback (offline demo).
      canvas.width = width;
      canvas.height = height;
      const imgData = ctx.createImageData(width, height);
      const data = imgData.data;

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = (y * width + x) * 4;
          const cx = x - 32;
          const cy = y - 32;
          const dist = Math.sqrt(cx * cx + cy * cy);

          let val = 0;
          if (type === "input") {
            val = dist < 20 ? (x % 4 === 0 ? 0.9 : 0.2) : 0.1;
            val += (Math.random() - 0.5) * 0.2;
          } else if (type === "base") {
            val = dist < 20 ? 0.6 : 0.1;
          } else if (type === "proposal") {
            val = dist < 20 && x % 4 === 0 ? 0.3 : 0.0;
          } else if (type === "candidate") {
            val = dist < 20 ? (x % 4 === 0 ? 0.9 : 0.6) : 0.1;
          } else if (type === "final") {
            val = dist < 20 ? (x % 4 === 0 ? 0.88 : 0.6) : 0.1;
          }

          val = Math.max(0, Math.min(1, val));
          const byteVal = Math.floor(val * 255);

          if (type === "proposal") {
            data[idx] = val > 0 ? byteVal : 0;
            data[idx + 1] = 120;
            data[idx + 2] = val < 0 ? byteVal : 200;
          } else {
            data[idx] = byteVal;
            data[idx + 1] = byteVal;
            data[idx + 2] = byteVal;
          }
          data[idx + 3] = 255;
        }
      }
      ctx.putImageData(imgData, 0, 0);
    };

    drawPattern(canvasRefs.input.current, "input", artifacts?.["input.npy"], artifactRanges["input.npy"]);
    drawPattern(canvasRefs.base.current, "base", artifacts?.["base.npy"], artifactRanges["base.npy"]);
    drawPattern(canvasRefs.proposal.current, "proposal", artifacts?.["proposal.npy"], artifactRanges["proposal.npy"]);
    drawPattern(canvasRefs.candidate.current, "candidate", artifacts?.["candidate.npy"], artifactRanges["candidate.npy"]);
    drawPattern(canvasRefs.final.current, "final", artifacts?.["final.npy"], artifactRanges["final.npy"]);

    // Diagnostic layers (real backend values when present).
    drawLayerMap(layerRefs.benefit.current, artifacts?.["proposal_benefit.npy"], artifactRanges["proposal_benefit.npy"], "divergent");
    drawLayerMap(layerRefs.decision.current, artifacts?.["decision_map.npy"], artifactRanges["decision_map.npy"], "binary");
    drawLayerMap(layerRefs.unresolved.current, artifacts?.["unresolved.npy"], artifactRanges["unresolved.npy"], "mask");
  }, [artifacts, artifactRanges, restorationData]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    // Map the visible position to canvas pixels; correct under pan/zoom.
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * canvas.width);
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * canvas.height);

    if (onPixelHover) {
      onPixelHover(x, y, {
        input: artifacts?.["input.npy"]?.[y % 128]?.[x % 128] ?? 0.24,
        base: artifacts?.["base.npy"]?.[y % 128]?.[x % 128] ?? 0.6,
        proposal: artifacts?.["proposal.npy"]?.[y % 128]?.[x % 128] ?? 0.3,
        final: artifacts?.["final.npy"]?.[y % 128]?.[x % 128] ?? 0.88,
      });
    }
  };

  // Linked pan/zoom across all canvases.
  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    setViewport((vp) => {
      const scale = Math.min(8, Math.max(1, vp.scale * factor));
      const wx = (mx - vp.x) / vp.scale;
      const wy = (my - vp.y) / vp.scale;
      return { scale, x: mx - wx * scale, y: my - wy * scale };
    });
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    // Never start a pan on interactive controls: pointer capture would steal
    // the subsequent click (e.g. the Reset view button would never fire).
    if ((e.target as HTMLElement).closest("button")) return;
    dragRef.current = { startX: e.clientX, startY: e.clientY, vx: viewport.x, vy: viewport.y };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const scale = viewport.scale;
    const maxX = scale > 1 ? (scale - 1) * rect.width : 0;
    const maxY = scale > 1 ? (scale - 1) * rect.height : 0;
    const x = Math.min(0, Math.max(-maxX, drag.vx + (e.clientX - drag.startX)));
    const y = Math.min(0, Math.max(-maxY, drag.vy + (e.clientY - drag.startY)));
    setViewport({ scale, x, y });
  };

  const endDrag = () => {
    dragRef.current = null;
  };

  const resetView = () => setViewport({ scale: 1, x: 0, y: 0 });

  const panes = [
    { key: "input", title: "Input Corrupted Tensor", badge: "y (raw)", ref: canvasRefs.input },
    { key: "base", title: "Base Reconstruction", badge: "b = U(y)+h(y)", ref: canvasRefs.base },
    { key: "proposal", title: "Bounded Detail Proposal", badge: "d", ref: canvasRefs.proposal },
    { key: "candidate", title: "Ungated Candidate", badge: "c = clip(b+d)", ref: canvasRefs.candidate },
    { key: "final", title: "Final Gated Reconstruction", badge: "o = G(b,d)", ref: canvasRefs.final },
  ];

  const gridLabel = artifacts?.["base.npy"]
    ? `${artifacts["base.npy"].length}x${artifacts["base.npy"][0].length}`
    : "64x64";

  const activeLayerEntries = Object.entries(LAYER_SPECS).filter(
    ([key]) => activeLayers[key]
  );

  return (
    <div
      ref={gridRef}
      className="workspace-grid"
      onWheel={handleWheel}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={endDrag}
      onPointerLeave={endDrag}
    >
      {fetchError && (
        <div className="pane-card" style={{ gridColumn: "1 / -1" }}>
          <div className="unavailable-state">
            <AlertCircle size={24} color="var(--text-dim)" />
            <span>Could not load artifacts: {fetchError}</span>
          </div>
        </div>
      )}

      <div className="pane-card" style={{ gridColumn: "1 / -1", flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
          Scroll to zoom · drag to pan · applied to all panes
        </span>
        <button className="btn-secondary" onClick={resetView} style={{ padding: "4px 10px", fontSize: "0.75rem" }}>
          <RotateCcw size={12} />
          Reset view
        </button>
      </div>

      {panes.map((pane) => {
        const realRange = artifactRanges[`${pane.key}.npy`];
        return (
          <div className="pane-card" key={pane.key}>
            <div className="pane-header">
              <div className="pane-title">
                <Eye size={16} color="var(--primary)" />
                {pane.title}
              </div>
              <span className="badge-tag">{pane.badge}</span>
            </div>

            <div className="canvas-wrapper">
              <canvas
                ref={pane.ref}
                className="pane-canvas"
                style={canvasTransform(viewport)}
                onMouseMove={handleMouseMove}
              />
            </div>

            <div className="metric-row">
              <span>Grid: {gridLabel}</span>
              <span>
                Range:{" "}
                {realRange
                  ? `[${realRange[0].toFixed(3)}, ${realRange[1].toFixed(3)}]`
                  : "[0.00, 1.00]"}
              </span>
            </div>
          </div>
        );
      })}

      {/* Optional diagnostic layers — real backend values or honest not-defined */}
      {activeLayerEntries.map(([key, spec]) => {
        const hasArtifact = spec.artifact && artifacts?.[spec.artifact];
        return (
          <div className="pane-card" key={key}>
            <div className="pane-header">
              <div className="pane-title">
                <Layers size={16} color="var(--accent-amber)" />
                {spec.title}
              </div>
              <span className="badge-tag">{spec.badge}</span>
            </div>

            {hasArtifact && spec.artifact ? (
              <div className="canvas-wrapper">
                <canvas
                  ref={layerRefs[key as keyof typeof layerRefs]}
                  className="pane-canvas"
                  style={canvasTransform(viewport)}
                />
              </div>
            ) : (
              <div className="unavailable-state" style={{ minHeight: 140 }}>
                <ShieldCheck size={24} color="var(--text-dim)" />
                <span>
                  <code>{spec.artifact ?? spec.badge}</code> not-defined
                  {restorationData ? "" : " — run an inference first"}
                </span>
              </div>
            )}

            <div className="metric-row">
              <span>Legend:</span>
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-dim)", lineHeight: 1.5 }}>
              {spec.legend}
            </div>
          </div>
        );
      })}
    </div>
  );
};

function canvasTransform(viewport: Viewport): React.CSSProperties {
  return {
    transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.scale})`,
    transformOrigin: "0 0",
    cursor: "grab",
  };
}

function drawLayerMap(
  canvas: HTMLCanvasElement | null,
  realValues: number[][] | undefined,
  realRange: [number, number] | undefined,
  kind: "divergent" | "binary" | "mask"
) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  if (!realValues || realValues.length === 0) {
    canvas.width = 64;
    canvas.height = 64;
    ctx.clearRect(0, 0, 64, 64);
    return;
  }
  const h = realValues.length;
  const w = realValues[0].length;
  canvas.width = w;
  canvas.height = h;
  const imgData = ctx.createImageData(w, h);
  const data = imgData.data;
  const [lo, hi] = realRange ?? [0, 1];
  const span = hi - lo || 1.0;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      const val = realValues[y][x];
      const norm = (val - lo) / span;
      if (kind === "binary") {
        // Gate map: 1 = accept (green), 0 = Base fallback (red).
        data[idx] = val >= 0.5 ? 16 : 220;
        data[idx + 1] = val >= 0.5 ? 185 : 30;
        data[idx + 2] = val >= 0.5 ? 129 : 30;
      } else if (kind === "mask") {
        // Unresolved: 1 = amber overlay, 0 = dark.
        data[idx] = 245;
        data[idx + 1] = val >= 0.5 ? 158 : 11;
        data[idx + 2] = val >= 0.5 ? 11 : 15;
      } else {
        // Divergent heat (benefit score): dark -> cyan -> amber.
        data[idx] = Math.floor(norm * 255);
        data[idx + 1] = Math.floor(120 + norm * 100);
        data[idx + 2] = Math.floor((1 - norm) * 180);
      }
      data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}
