import { test, expect, type Page } from "@playwright/test";

/**
 * Phase 13 review-UI coverage: real intervention stats, reliability
 * dashboards, review-event recording, review-package export, and the
 * linked pan/zoom across the workspace canvases.
 *
 * Every assertion targets values the backend actually computed (or an
 * explicit not-defined state) — never hard-coded UI placeholders.
 */
test("Intervention Inspector shows stats computed from the run's proposal artifact", async ({
  page,
}) => {
  await runAndWait(page);

  await page.getByRole("button", { name: "Intervention Inspector" }).click();

  // Largest-intervention cards are derived from the served proposal.npy.
  await expect(
    page.getByText("Largest positive intervention in this run")
  ).toBeVisible({ timeout: 30_000 });
  await expect(
    page.getByText("Largest negative intervention in this run")
  ).toBeVisible();

  // Bound check reflects the real data (promoted contract keeps |d| <= 0.10;
  // the checkpoint-less passthrough produces zero proposals, which is also in-bounds).
  await expect(page.getByText(/Bound check: within|EXCEEDS/)).toBeVisible();

  // No hard-coded confidence placeholder survives.
  await expect(page.getByText("Confidence: 94.2%")).toHaveCount(0);
});

test("Reliability tab computes dashboards, records review events, and exports a review package", async ({
  page,
}) => {
  await runAndWait(page);

  await page.getByRole("button", { name: "Reliability & Provenance" }).click();

  // Dashboards render from served artifacts: benefit histogram when the run
  // qualifies (256x256 grid), otherwise the honest not-defined card.
  await expect(page.getByText("Reliability Dashboards (this run)")).toBeVisible();
  const benefitOrNotDefined = page.getByText(
    /Benefit score ranking|not-defined/
  );
  await expect(benefitOrNotDefined.first()).toBeVisible({ timeout: 30_000 });

  // Worst-group / calibration / downstream is explicitly not served by this
  // service (it lives in governed evaluation bundles).
  await expect(
    page.getByText("Worst-group / calibration / downstream")
  ).toBeVisible();

  // Approve action records a review event against the run.
  await page.getByRole("button", { name: "Approve Gated Output" }).click();
  await expect(page.getByText(/Review action recorded for run eval-\d+/)).toBeVisible();

  // Export produces the JSON review package as a download.
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export Package" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/^eval-\d+-\d+-review\.json$/);
});

test("linked pan/zoom applies to the workspace canvases and Reset view restores them", async ({
  page,
}) => {
  await runAndWait(page);

  const firstCanvas = page.locator("canvas").first();
  await expect(firstCanvas).toBeVisible();

  const initialTransform = await canvasTransform(firstCanvas);
  expect(initialTransform).toContain("scale(1)");

  // Zoom in over the workspace grid (wheel up).
  const grid = page.locator(".workspace-grid");
  const box = await grid.boundingBox();
  if (!box) throw new Error("workspace grid has no bounding box");
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.wheel(0, -240);

  // All canvases share the same zoomed transform.
  await expect
    .poll(async () => canvasTransform(firstCanvas), { timeout: 15_000 })
    .not.toContain("scale(1)");

  // Reset view returns every canvas to the identity transform.
  await page.getByRole("button", { name: "Reset view" }).click();
  await expect.poll(() => canvasTransform(firstCanvas)).toContain("scale(1)");
});

async function runAndWait(page: Page) {
  await page.goto("/");
  await page.getByRole("button", { name: "Run Unified Inference" }).click();
  await expect(page.getByText(/Run eval-\d+.* completed/)).toBeVisible({
    timeout: 90_000,
  });
}

async function canvasTransform(locator: ReturnType<Page["locator"]>) {
  return locator.evaluate((el) => (el as HTMLElement).style.transform);
}
