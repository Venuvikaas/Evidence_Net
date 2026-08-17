import React, { useRef, useState } from "react";
import {
  Layers,
  Sliders,
  ShieldCheck,
  Play,
  Crosshair,
  CheckCircle,
  AlertCircle,
  AlertTriangle,
} from "lucide-react";
import { RestorationWorkspace } from "./components/RestorationWorkspace";
import { InterventionInspector } from "./components/InterventionInspector";
import { ReliabilityWorkspace } from "./components/ReliabilityWorkspace";
import { PolicyExplorer } from "./components/PolicyExplorer";
import { FailureBrowser } from "./components/FailureBrowser";
import { RestorationResponse, runRestorationInference } from "./api/client";
import { UploadedImage, imageFileToGrayscale } from "./api/imageUpload";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<
    "workspace" | "intervention" | "reliability" | "policy" | "failures"
  >("workspace");
  const [restorationData, setRestorationData] = useState<RestorationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [lastRunId, setLastRunId] = useState<string | null>(null);
  const [lastRunInputName, setLastRunInputName] = useState<string | null>(null);
  const [lastRunTargetName, setLastRunTargetName] = useState<string | null>(null);
  const [hoverPixel, setHoverPixel] = useState<{ x: number; y: number; values: Record<string, number> } | null>(null);
  const [uploadedImage, setUploadedImage] = useState<UploadedImage | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [targetFile, setTargetFile] = useState<File | null>(null);
  const [targetError, setTargetError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const targetInputRef = useRef<HTMLInputElement>(null);
  const dragCounter = useRef(0);

  const [activeLayers, setActiveLayers] = useState<Record<string, boolean>>({
    benefit: true,
    consistency: false,
    stability: false,
    familiarity: false,
    decision: true,
    unresolved: false,
  });

  const applyImageFile = async (file: File) => {
    setUploadError(null);
    try {
      const image = await imageFileToGrayscale(file);
      setUploadedImage(image);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleFileSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    // Reset the input so selecting the same file again re-triggers onChange.
    e.target.value = "";
    if (file) void applyImageFile(file);
  };

  const handleTargetSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    // Reset the input so selecting the same file again re-triggers onChange.
    e.target.value = "";
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setTargetError(
        `Unsupported file type "${file.type || "unknown"}". Upload an image (PNG/JPG/BMP/...).`
      );
      return;
    }
    setTargetError(null);
    setTargetFile(file);
  };

  // Drag-and-drop upload: the whole app body accepts dropped images.
  const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    dragCounter.current += 1;
    setIsDragging(true);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    // preventDefault is required for the drop to be accepted.
    e.preventDefault();
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    dragCounter.current -= 1;
    if (dragCounter.current <= 0) {
      dragCounter.current = 0;
      setIsDragging(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    dragCounter.current = 0;
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setUploadError(
        `Unsupported file type "${file.type || "unknown"}". Drop an image (PNG/JPG/BMP/...).`
      );
      return;
    }
    void applyImageFile(file);
  };

  const handleRunInference = async () => {
    setLoading(true);
    setRunError(null);
    try {
      let inputValues: number[][] | undefined;
      let hasTarget = false;
      let targetValues: number[][] | undefined;
      let shape: number[] | undefined;

      if (uploadedImage) {
        inputValues = uploadedImage.values;
        shape = [uploadedImage.height, uploadedImage.width];
        if (targetFile) {
          // Convert the target to the input's exact grid so pixel metrics
          // (PSNR/SSIM/MAE) are computed on aligned arrays.
          const target = await imageFileToGrayscale(targetFile, {
            width: uploadedImage.width,
            height: uploadedImage.height,
          });
          targetValues = target.values;
          hasTarget = true;
        }
      } else if (targetFile) {
        setRunError(
          "Upload an input image first — the target is compared against the restored input."
        );
        return;
      }

      const data = await runRestorationInference(
        inputValues,
        hasTarget,
        targetValues,
        shape
      );
      setRestorationData(data);
      setLastRunId(data.run_id);
      // Snapshot the filenames so the banner reflects what this run restored,
      // even if the user changes the uploads afterwards.
      setLastRunInputName(uploadedImage?.name ?? null);
      setLastRunTargetName(hasTarget && targetFile ? targetFile.name : null);
    } catch (e) {
      // Surface the failure instead of silently keeping the synthetic demo:
      // a silent catch is exactly what made the button appear to do nothing.
      console.warn("API offline, using synthetic inference state", e);
      setRestorationData(null);
      setLastRunId(null);
      setLastRunInputName(null);
      setLastRunTargetName(null);
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

          <button
            className={`tab-btn ${activeTab === "policy" ? "active" : ""}`}
            onClick={() => setActiveTab("policy")}
          >
            <Sliders size={16} />
            Policy Explorer
          </button>

          <button
            className={`tab-btn ${activeTab === "failures" ? "active" : ""}`}
            onClick={() => setActiveTab("failures")}
          >
            <AlertTriangle size={16} />
            Failure Browser
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
            Run <code>{lastRunId}</code> completed — restored{" "}
            {lastRunInputName ? (
              <code className="banner-filename">“{lastRunInputName}”</code>
            ) : (
              "the built-in demo input"
            )}
            {lastRunTargetName && (
              <>
                {" "}
                (metrics computed against{" "}
                <code className="banner-filename">“{lastRunTargetName}”</code>)
              </>
            )}
            . The workspace now shows the real restored artifacts from the frozen
            Base + Proposal checkpoints.
          </span>
        </div>
      )}

      {/* Body Area */}
      <div
        className={`app-body${isDragging ? " drag-active" : ""}`}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {isDragging && (
          <div className="drop-overlay">
            <Crosshair size={28} />
            <span>Drop image to restore it</span>
          </div>
        )}
        {/* Left Sidebar Control Panel */}
        <aside className="sidebar-panel">
          <div>
            <h3 className="section-title">Input Image</h3>
            <div className="control-card">
              {uploadedImage ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.82rem", color: "var(--accent-cyan)" }}>
                    <CheckCircle size={14} />
                    <span className="upload-file-name">{uploadedImage.name}</span>
                  </div>
                  <div style={{ fontSize: "0.78rem", color: "var(--text-dim)" }}>
                    {uploadedImage.width}×{uploadedImage.height} grayscale — the next run
                    will restore this image.
                  </div>
                  <button className="btn-secondary upload-btn" onClick={() => setUploadedImage(null)}>
                    Reset to demo input
                  </button>
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                    Drop an image anywhere in the workspace, or use the button below. Images
                    are converted to grayscale and downscaled to ≤512px client-side.
                  </div>
                  <button className="btn-secondary upload-btn" onClick={() => fileInputRef.current?.click()}>
                    <Crosshair size={14} />
                    Upload Image
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    aria-label="Upload input image"
                    style={{ display: "none" }}
                    onChange={handleFileSelected}
                  />
                  {uploadError && (
                    <div style={{ fontSize: "0.78rem", color: "var(--accent-rose)" }}>
                      Could not read image: {uploadError}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          <div>
            <h3 className="section-title">Target Image (optional)</h3>
            <div className="control-card">
              {targetFile ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.82rem", color: "var(--accent-emerald)" }}>
                    <CheckCircle size={14} />
                    <span className="upload-file-name">{targetFile.name}</span>
                  </div>
                  <div style={{ fontSize: "0.78rem", color: "var(--text-dim)" }}>
                    Ground truth — the next run will compute PSNR / SSIM / MAE of the
                    restored outputs against it (resized to match the input grid).
                  </div>
                  <button className="btn-secondary upload-btn" onClick={() => setTargetFile(null)}>
                    Remove target
                  </button>
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                    Upload a ground-truth image to compute restoration metrics. Requires an
                    input image; the target is resized to the input's grid.
                  </div>
                  <button className="btn-secondary upload-btn" onClick={() => targetInputRef.current?.click()}>
                    <ShieldCheck size={14} />
                    Upload Target
                  </button>
                  <input
                    ref={targetInputRef}
                    type="file"
                    accept="image/*"
                    aria-label="Upload target image"
                    style={{ display: "none" }}
                    onChange={handleTargetSelected}
                  />
                  {targetError && (
                    <div style={{ fontSize: "0.78rem", color: "var(--accent-rose)" }}>
                      {targetError}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

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

        {activeTab === "intervention" && (
          <InterventionInspector
            restorationData={restorationData}
            onExplorePolicy={() => setActiveTab("policy")}
          />
        )}

        {activeTab === "reliability" && (
          <ReliabilityWorkspace restorationData={restorationData} />
        )}

        {activeTab === "policy" && <PolicyExplorer restorationData={restorationData} />}

        {activeTab === "failures" && <FailureBrowser />}
      </div>
    </div>
  );
};

export default App;
