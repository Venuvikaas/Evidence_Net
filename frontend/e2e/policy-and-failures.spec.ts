import { test, expect } from "@playwright/test";

/**
 * Phase 13 review-UI coverage for the Policy Explorer and Failure Browser.
 *
 * Policy Explorer must show either the real backend-computed benefit scores
 * (checkpoint runs on the 256x256 grid) or an explicit "not-defined" state
 * (checkpoint-less pipelines) — never hard-coded values. The Failure Browser
 * serves the frozen Gate 9 evidence banks, which exist in every environment.
 */
test("Policy Explorer shows real backend-computed scores or an explicit not-defined state", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Run Unified Inference" }).click();

  // Wait for the run to complete before switching tabs.
  await expect(page.getByText(/Run eval-\d+.* completed/)).toBeVisible({
    timeout: 90_000,
  });

  await page.getByRole("button", { name: "Policy Explorer" }).click();

  // The frozen policy configuration is always shown.
  await expect(
    page.getByText("Frozen policy configuration (decision-policy-v1)")
  ).toBeVisible();

  // Either the run qualified for benefit scores (coverage explorer) or the
  // backend honestly says not-defined. Both are correct; a hard-coded value
  // is not.
  const eitherState = page.getByText(
    /Coverage vs benefit-score threshold \(this run\)|not-defined for this run/
  );
  await expect(eitherState.first()).toBeVisible({ timeout: 30_000 });

  // Hard-coded fake values from the pre-fix UI must never appear.
  await expect(page.getByText("94.2%")).toHaveCount(0);
  await expect(page.getByText("+0.042 max")).toHaveCount(0);
});

test("Failure Browser serves the frozen Gate 9 evidence banks", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Failure Browser" }).click();

  // Natural failure bank from the committed frozen record.
  await expect(page.getByText(/Natural Failure Bank/)).toBeVisible();
  await expect(page.getByText(/natural-failures-v1/)).toBeVisible();
  // The periodic-region stress cases archived in FAIL-001.
  await expect(page.getByText("000893")).toBeVisible();
  await expect(page.getByText("proposal-harm").first()).toBeVisible();

  // Hidden stress definitions with their frozen hash.
  await expect(page.getByText(/Hidden Stress Definitions/)).toBeVisible();
  await expect(page.getByText(/087d6c1369de350a/)).toBeVisible();
});
