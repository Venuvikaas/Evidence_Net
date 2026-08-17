import { defineConfig, devices } from "@playwright/test";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const frontendDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(frontendDir, "..");
// Prefer the project venv's Python on Windows then POSIX, else system python.
const winPython = join(repoRoot, ".venv", "Scripts", "python.exe");
const posixPython = join(repoRoot, ".venv", "bin", "python");
const python = existsSync(winPython)
  ? winPython
  : existsSync(posixPython)
    ? posixPython
    : "python";

/**
 * End-to-end review-UI tests. Each run starts (or reuses) the real FastAPI
 * backend on :8000 and the Vite dev server on :3000, then drives the browser.
 *
 * The backend anchors run bundles to <repo>/runs regardless of its working
 * directory, so the API may be launched from the repo root here.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  expect: { timeout: 60_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:3000",
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      // Use the locally installed Google Chrome so no browser download is
      // needed on developer machines; on machines without Chrome, run
      // `npx playwright install chromium` and drop the channel below.
      use: { ...devices["Desktop Chrome"], channel: "chrome" },
    },
  ],
  webServer: [
    {
      command: `"${python}" -m uvicorn evidence_net.api.app:app --host 127.0.0.1 --port 8000`,
      cwd: repoRoot,
      url: "http://127.0.0.1:8000/api/v1/health",
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
    {
      command: "npm run dev -- --port 3000",
      cwd: frontendDir,
      url: "http://localhost:3000",
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
});
