import { test, expect } from "@playwright/test";

/**
 * E2E coverage for drag-and-drop upload: dropping an image anywhere on the
 * workspace registers it as the run input (same as the upload button), and
 * the run restores the dropped image.
 */
test("drag-and-drop uploads an image and the run uses it", async ({ page }) => {
  // 16x16 checkerboard BMP fixture (bright/even, dark/odd cells).
  const bmp = makeBmp(16, 16, (x, y) => ((x + y) % 2 === 0 ? [255, 255, 255] : [0, 0, 0]));
  const bytes = Array.from(bmp);

  await page.goto("/");

  // Drag over: the drop overlay appears.
  await dispatchDrag(page, bytes, "dragenter");
  await dispatchDrag(page, bytes, "dragover");
  await expect(page.getByText("Drop image to restore it")).toBeVisible();

  // Drop: the overlay disappears and the file is registered in the card.
  await dispatchDrag(page, bytes, "drop");
  await expect(page.getByText("Drop image to restore it")).toHaveCount(0);
  await expect(page.getByText("dropped-16.bmp")).toBeVisible();

  // The run uses the dropped image instead of the demo input.
  await page.getByRole("button", { name: "Run Unified Inference" }).click();
  await expect(page.getByText(/Run eval-\d+.* completed/)).toBeVisible({
    timeout: 90_000,
  });
  await expect(page.getByText("Grid: 64x64").first()).toHaveCount(0, {
    timeout: 60_000,
  });
  // Grid depends on whether frozen checkpoints exist (2x upscale) or the
  // passthrough pipeline runs in CI; accept either.
  await expect(page.getByText(/Grid: (16x16|32x32)/).first()).toBeVisible();
});

async function dispatchDrag(page: import("@playwright/test").Page, bytes: number[], type: string) {
  await page.evaluate(
    ({ b, t }) => {
      const el = document.querySelector(".app-body");
      if (!el) throw new Error("app-body not found");
      const dt = new DataTransfer();
      dt.items.add(new File([new Uint8Array(b)], "dropped-16.bmp", { type: "image/bmp" }));
      el.dispatchEvent(
        new DragEvent(t, { dataTransfer: dt, bubbles: true, cancelable: true })
      );
    },
    { b: bytes, t: type }
  );
}

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
