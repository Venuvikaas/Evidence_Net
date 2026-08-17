import { test, expect } from "@playwright/test";

/**
 * E2E coverage for the target-image flow: uploading a ground-truth target
 * alongside the input makes the run compute restoration metrics, which are
 * shown in the Reliability & Provenance tab.
 */
test("uploading a target computes and displays restoration metrics", async ({
  page,
}) => {
  // Input: 16x16 checkerboard. Target: 16x16 solid mid-gray. Distinct enough
  // that metrics are finite and meaningful.
  const input = makeBmp(16, 16, (x, y) => ((x + y) % 2 === 0 ? [255, 255, 255] : [0, 0, 0]));
  const target = makeBmp(16, 16, () => [128, 128, 128]);

  await page.goto("/");

  await page.setInputFiles('input[aria-label="Upload input image"]', {
    name: "input-checker.bmp",
    mimeType: "image/bmp",
    buffer: input,
  });
  await expect(page.getByText("input-checker.bmp")).toBeVisible();

  await page.setInputFiles('input[aria-label="Upload target image"]', {
    name: "target-gray.bmp",
    mimeType: "image/bmp",
    buffer: target,
  });
  await expect(page.getByText("target-gray.bmp")).toBeVisible();

  // Run with both images: the success banner appears and names both files.
  await page.getByRole("button", { name: "Run Unified Inference" }).click();
  await expect(page.getByText(/Run eval-\d+.* completed/)).toBeVisible({
    timeout: 90_000,
  });
  await expect(page.locator(".status-success")).toContainText("input-checker.bmp");
  await expect(page.locator(".status-success")).toContainText("target-gray.bmp");

  // Open the Reliability tab: metrics were computed against the target.
  await page.getByRole("button", { name: "Reliability & Provenance" }).click();
  await expect(page.getByText("Restoration Metrics (vs uploaded target)")).toBeVisible();
  await expect(page.getByText("base output")).toBeVisible();
  await expect(page.getByText("final output")).toBeVisible();
  // Primary metrics are shown for each output (base, candidate, final). The
  // target card also mentions PSNR/SSIM/MAE, so scope to the first metric row.
  await expect(page.getByText("PSNR").first()).toBeVisible();
  await expect(page.getByText("SSIM").first()).toBeVisible();
  await expect(page.getByText("MAE").first()).toBeVisible();
});

/**
 * Build a minimal uncompressed 24-bit BMP (BITMAPFILEHEADER + BITMAPINFOHEADER
 * + bottom-up BGR rows padded to 4 bytes). Returns a Buffer.
 */
function makeBmp(
  width: number,
  height: number,
  pixel: (x: number, y: number) => [number, number, number]
): Buffer {
  const rowSize = Math.ceil((width * 3) / 4) * 4;
  const pixelDataSize = rowSize * height;
  const fileSize = 54 + pixelDataSize;
  const buf = Buffer.alloc(fileSize);

  buf.write("BM", 0, "ascii");
  buf.writeUInt32LE(fileSize, 2);
  buf.writeUInt32LE(54, 10);

  buf.writeUInt32LE(40, 14);
  buf.writeInt32LE(width, 18);
  buf.writeInt32LE(height, 22);
  buf.writeUInt16LE(1, 26);
  buf.writeUInt16LE(24, 28);
  buf.writeUInt32LE(0, 30);
  buf.writeUInt32LE(pixelDataSize, 34);
  buf.writeInt32LE(2835, 38);
  buf.writeInt32LE(2835, 42);

  for (let y = 0; y < height; y++) {
    const rowStart = 54 + (height - 1 - y) * rowSize;
    for (let x = 0; x < width; x++) {
      const [r, g, b] = pixel(x, y);
      const off = rowStart + x * 3;
      buf[off] = b;
      buf[off + 1] = g;
      buf[off + 2] = r;
    }
  }
  return buf;
}
