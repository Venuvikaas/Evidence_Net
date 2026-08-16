import React, { useEffect, useRef } from "react";
import { Eye, Layers, AlertCircle, ShieldCheck } from "lucide-react";
import { RestorationResponse } from "../api/client";

interface RestorationWorkspaceProps {
  restorationData: RestorationResponse | null;
  activeLayers: Record<string, boolean>;
  onPixelHover?: (x: number, y: number, values: Record<string, number>) => void;
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

  useEffect(() => {
    // Generate synthetic pixel pattern for interactive visual demonstration
    const width = 64;
    const height = 64;

    const drawPattern = (
      canvas: HTMLCanvasElement | null,
      type: "input" | "base" | "proposal" | "candidate" | "final"
    ) => {
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

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
            // Degraded noisy line pattern
            val = dist < 20 ? (x % 4 === 0 ? 0.9 : 0.2) : 0.1;
            val += (Math.random() - 0.5) * 0.2;
          } else if (type === "base") {
            // Smooth base reconstruction
            val = dist < 20 ? 0.6 : 0.1;
          } else if (type === "proposal") {
            // High-frequency detail proposal (+/- magnitude)
            val = dist < 20 && x % 4 === 0 ? 0.3 : 0.0;
          } else if (type === "candidate") {
            // Base + proposal
            val = dist < 20 ? (x % 4 === 0 ? 0.9 : 0.6) : 0.1;
          } else if (type === "final") {
            // Gated final reconstruction
            val = dist < 20 ? (x % 4 === 0 ? 0.88 : 0.6) : 0.1;
          }

          val = Math.max(0, Math.min(1, val));
          const byteVal = Math.floor(val * 255);

          if (type === "proposal") {
            // Proposal uses blue-orange divergent visualization
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

    drawPattern(canvasRefs.input.current, "input");
    drawPattern(canvasRefs.base.current, "base");
    drawPattern(canvasRefs.proposal.current, "proposal");
    drawPattern(canvasRefs.candidate.current, "candidate");
    drawPattern(canvasRefs.final.current, "final");
  }, [restorationData]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * 64);
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * 64);

    if (onPixelHover) {
      onPixelHover(x, y, {
        input: 0.24,
        base: 0.60,
        proposal: 0.30,
        final: 0.88,
      });
    }
  };

  const panes = [
    { key: "input", title: "Input Corrupted Tensor", badge: "y (raw)", ref: canvasRefs.input },
    { key: "base", title: "Base Reconstruction", badge: "b = U(y)+h(y)", ref: canvasRefs.base },
    { key: "proposal", title: "Bounded Detail Proposal", badge: "d", ref: canvasRefs.proposal },
    { key: "candidate", title: "Ungated Candidate", badge: "c = clip(b+d)", ref: canvasRefs.candidate },
    { key: "final", title: "Final Gated Reconstruction", badge: "o = G(b,d)", ref: canvasRefs.final },
  ];

  return (
    <div className="workspace-grid">
      {panes.map((pane) => (
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
              onMouseMove={handleMouseMove}
            />
          </div>

          <div className="metric-row">
            <span>Grid: 64x64</span>
            <span>Range: [0.00, 1.00]</span>
          </div>
        </div>
      ))}

      {/* Optional diagnostic layer representations */}
      <div className="pane-card">
        <div className="pane-header">
          <div className="pane-title">
            <Layers size={16} color="var(--accent-amber)" />
            Proposal Benefit Layer
          </div>
          <span className="badge-tag">proposal_benefit.npy</span>
        </div>

        {activeLayers.benefit ? (
          <div className="canvas-wrapper">
            <div style={{ padding: 20, textAlign: "center", color: "var(--accent-emerald)" }}>
              <ShieldCheck size={32} />
              <div style={{ fontSize: "0.85rem", marginTop: 8 }}>High Benefit Region (Confidence: 94.2%)</div>
            </div>
          </div>
        ) : (
          <div className="unavailable-state">
            <AlertCircle size={24} color="var(--text-dim)" />
            <span>Layer Disabled or Marked <code>not-defined</code></span>
          </div>
        )}

        <div className="metric-row">
          <span>Contract: support-definition-v1</span>
        </div>
      </div>
    </div>
  );
};
