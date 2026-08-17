import { test, expect, type Page } from "@playwright/test";

/**
 * End-to-end review-UI test (Phase 13 "full review workflow").
 *
 * Starts the real API + frontend, clicks "Run Unified Inference", and
 * verifies the workspace stops showing the synthetic demo and draws the
 * real run artifacts from the backend (success banner, grid label, and a
 * canvas redraw at the real artifact resolution).
 */
test("Run Unified Inference executes and the canvases update with real artifacts", async ({
  page,
}) => {
  await page.goto("/");

  const runButton = page.getByRole("button", { name: "Run Unified Inference" });
  await expect(runButton).toBeVisible();

  // Before the run the workspace shows the synthetic 64x64 demo pattern.
  await expect(page.getByText("Grid: 64x64").first()).toBeVisible();
  const before = await sampleFirstCanvas(page);

  await runButton.click();

  // The API round-trip completes: a success banner with the run id appears.
  await expect(page.getByText(/Run eval-\d+.* completed/)).toBeVisible({
    timeout: 90_000,
  });

  // Real artifacts are fetched and drawn: every pane reports the artifact
  // grid (128x128 after the API's downsampling) instead of the demo 64x64.
  await expect(page.getByText("Grid: 128x128").first()).toBeVisible({
    timeout: 60_000,
  });

  // The first canvas was actually redrawn (resolution and/or pixel data
  // changed from the synthetic demo).
  await expect
    .poll(async () => sampleFirstCanvas(page), { timeout: 60_000 })
    .not.toEqual(before);
});

async function sampleFirstCanvas(page: Page) {
  return page.locator("canvas").first().evaluate((canvas) => {
    const c = canvas as HTMLCanvasElement;
    const ctx = c.getContext("2d");
    if (!ctx) return { width: c.width, height: c.height, block: [] };
    const w = Math.min(16, c.width);
    const h = Math.min(16, c.height);
    const block = ctx.getImageData(0, 0, w, h).data;
    return { width: c.width, height: c.height, block: Array.from(block) };
  });
}
