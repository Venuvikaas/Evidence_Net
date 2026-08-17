import { test, expect } from "@playwright/test";

/**
 * Regression test for the API-offline state (the fix that made the run
 * button show a visible error instead of silently doing nothing).
 *
 * The backend is intentionally left running; the restoration request is
 * intercepted and aborted in the browser, which exercises the exact same
 * fetch-failure path a real unreachable API would trigger.
 */
test("Run Unified Inference shows the API-offline error banner instead of doing nothing", async ({
  page,
}) => {
  // Simulate the backend being unreachable: abort the restoration POST.
  await page.route("**/api/v1/restoration", (route) => route.abort());

  await page.goto("/");

  const runButton = page.getByRole("button", { name: "Run Unified Inference" });
  await expect(runButton).toBeVisible();
  await expect(page.getByText("Grid: 64x64").first()).toBeVisible();

  await runButton.click();

  // The failure is surfaced: a red banner explains the API is unreachable
  // and how to start it...
  await expect(page.getByText(/The EVIDENCE-Net API is not reachable/)).toBeVisible({
    timeout: 30_000,
  });

  // ...no success banner is shown...
  await expect(page.getByText(/Run eval-\d+.* completed/)).toHaveCount(0);

  // ...and the synthetic demo is retained, but now with a visible reason.
  await expect(page.getByText("Grid: 64x64").first()).toBeVisible();

  // The button returns to its idle state after the failed attempt.
  await expect(runButton).toBeEnabled();
});
