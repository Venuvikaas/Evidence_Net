#!/usr/bin/env bash
# Demo launcher for EVIDENCE-Net.
#
# Starts the FastAPI backend (real frozen checkpoints, port 8000) and the
# Vite dev server (port 3000). Open http://localhost:3000 and click
# "Run Unified Inference" — the workspace then draws the REAL artifacts
# produced by the frozen Base + Proposal checkpoints.
#
# Usage: bash scripts/demo_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# --- Locate the Python interpreter (prefer the project venv) ---------------
PY=""
for candidate in ".venv/Scripts/python.exe" ".venv/bin/python" "python"; do
  if command -v "$candidate" > /dev/null 2>&1; then
    PY="$candidate"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "ERROR: no Python interpreter found. Create the venv first:"
  echo "  python -m venv .venv && .venv/Scripts/pip install -e \".[dev]\""
  exit 1
fi

if ! command -v npm > /dev/null 2>&1; then
  echo "ERROR: npm not found. Install Node.js first (frontend/ requires it)."
  exit 1
fi

# --- Refuse to start if the ports are already taken -------------------------
if netstat -ano 2>/dev/null | grep -qE "LISTENING.*:(8000|3000)"; then
  echo "ERROR: port 8000 or 3000 is already in use."
  echo "       Stop the existing API/frontend processes and re-run this script."
  exit 1
fi

API_PID=""
VITE_PID=""
cleanup() {
  echo ""
  echo "==> Stopping EVIDENCE-Net demo..."
  [ -n "$VITE_PID" ] && kill "$VITE_PID" 2>/dev/null || true
  [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

echo "==> Starting EVIDENCE-Net API on :8000 (frozen checkpoints)..."
"$PY" -m uvicorn evidence_net.api.app:app \
  --host 127.0.0.1 --port 8000 \
  > /tmp/evidence_api.log 2>&1 &
API_PID=$!

echo "==> Starting frontend on :3000 (Vite)..."
(cd frontend && npm run dev -- --port 3000 > /tmp/evidence_vite.log 2>&1) &
VITE_PID=$!

# Wait for both to come up (give the API extra time to load the checkpoints).
API_READY=0
for _ in $(seq 1 60); do
  if curl -s -m 2 http://127.0.0.1:8000/api/v1/health > /dev/null 2>&1; then
    API_READY=1
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "ERROR: the API process exited early. Check /tmp/evidence_api.log"
    exit 1
  fi
  sleep 1
done

VITE_READY=0
for _ in $(seq 1 30); do
  if curl -s -m 2 -o /dev/null http://localhost:3000/ 2>/dev/null; then
    VITE_READY=1
    break
  fi
  sleep 1
done

if [ "$API_READY" != "1" ] || [ "$VITE_READY" != "1" ]; then
  echo "ERROR: servers did not come up in time."
  echo "  API log:   /tmp/evidence_api.log"
  echo "  Vite log:  /tmp/evidence_vite.log"
  exit 1
fi

echo ""
echo "============================================================"
echo "  EVIDENCE-Net demo is ready:"
echo "    UI:    http://localhost:3000"
echo "    API:   http://127.0.0.1:8000/api/v1/health"
echo "    Docs:  http://127.0.0.1:8000/docs"
echo ""
echo "  In the browser: click \"Run Unified Inference\", then hover"
echo "  the canvases to inspect real per-pixel values."
echo "============================================================"
echo ""
echo "PIDs: API=$API_PID Vite=$VITE_PID (logs: /tmp/evidence_api.log, /tmp/evidence_vite.log)"
echo "Press Ctrl+C to stop both servers."

# Keep the script alive so Ctrl+C cleans up both processes.
while kill -0 "$API_PID" 2>/dev/null || kill -0 "$VITE_PID" 2>/dev/null; do
  sleep 2
done

cleanup
