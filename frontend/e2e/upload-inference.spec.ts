import { test, expect } from "@playwright/test";

/**
 * E2E coverage for the image-upload flow: the user uploads their own image
 * instead of the built-in demo input, and "Run Unified Inference" restores
 * that image.
 *
 * The fixture is a minimal 24-bit BMP built in-memory (no image libraries),
 * which Chrome decodes without any tooling.
 */
test("uploaded image is used for the run and the workspace updates", async ({
  page,
}) => {
  // 16x16 checkerboard: even cells bright (255), odd cells dark (0).
  const bmp = makeBmp(16, 16, (x, y) => ((x + y) % 2 === 0 ? [255, 255, 255] : [0, 0, 0]));

  await page.goto("/");

  // Upload through the hidden file input.
  await page.setInputFiles('input[type="file"]', {
    name: "checker-16.bmp",
    mimeType: "image/bmp",
    buffer: bmp,
  });

  // The uploaded file is reflected in the Input Image card.
  await expect(page.getByText("checker-16.bmp")).toBeVisible();

  await page.getByRole("button", { name: "Run Unified Inference" }).click();

  // The run completes against the uploaded input...
  await expect(page.getByText(/Run eval-\d+.* completed/)).toBeVisible({
    timeout: 90_000,
  });

  // ...and the workspace redraws with the uploaded image's artifacts instead
  // of the synthetic 64x64 demo grid. The grid size depends on whether the
  // frozen checkpoints are present: with models, a 16x16 input produces 32x32
  // outputs (2x upscale); in checkpoint-less CI the passthrough pipeline keeps
  // 16x16. Accept either, but never the demo grid.
  await expect(page.getByText("Grid: 64x64").first()).toHaveCount(0, {
    timeout: 60_000,
  });
  await expect(page.getByText(/Grid: (16x16|32x32)/).first()).toBeVisible();
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

  // BITMAPFILEHEADER (14 bytes)
  buf.write("BM", 0, "ascii");
  buf.writeUInt32LE(fileSize, 2);
  buf.writeUInt32LE(54, 10); // pixel data offset

  // BITMAPINFOHEADER (40 bytes)
  buf.writeUInt32LE(40, 14);
  buf.writeInt32LE(width, 18);
  buf.writeInt32LE(height, 22); // positive -> bottom-up rows
  buf.writeUInt16LE(1, 26); // planes
  buf.writeUInt16LE(24, 28); // bits per pixel
  buf.writeUInt32LE(0, 30); // compression: none
  buf.writeUInt32LE(pixelDataSize, 34);
  buf.writeInt32LE(2835, 38); // horizontal ppm
  buf.writeInt32LE(2835, 42); // vertical ppm

  // Pixel data (bottom-up, BGR order)
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
