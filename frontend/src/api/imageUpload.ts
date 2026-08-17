/**
 * Client-side image upload helpers (Phase 13 review UI).
 *
 * Converts an uploaded image file into the grayscale float grid the
 * restoration API expects (values in [0, 1], row-major) and downscales it
 * client-side so the JSON payload stays small. The Base/Proposal models are
 * fully convolutional, so any input size is accepted and the output grid is
 * 2x the input.
 */

export const MAX_UPLOAD_DIM = 512;

export interface UploadedImage {
  name: string;
  width: number;
  height: number;
  values: number[][];
}

export interface GrayscaleOptions {
  maxDim?: number;
  /** Exact output size (e.g. convert a target image to the input's grid so
   * pixel metrics are computed on aligned arrays). */
  width?: number;
  height?: number;
}

export async function imageFileToGrayscale(
  file: File,
  options: GrayscaleOptions = {}
): Promise<UploadedImage> {
  const bitmap = await createImageBitmap(file);
  try {
    let width: number;
    let height: number;
    if (options.width && options.height) {
      width = options.width;
      height = options.height;
    } else {
      const maxDim = options.maxDim ?? MAX_UPLOAD_DIM;
      const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
      width = Math.max(1, Math.round(bitmap.width * scale));
      height = Math.max(1, Math.round(bitmap.height * scale));
    }

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D context unavailable");
    ctx.drawImage(bitmap, 0, 0, width, height);

    const imageData = ctx.getImageData(0, 0, width, height);
    const values: number[][] = [];
    for (let y = 0; y < height; y++) {
      const row = new Array<number>(width);
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        // Rec. 601 luma; rounded to keep the JSON payload compact.
        const luma =
          (0.299 * imageData.data[i] +
            0.587 * imageData.data[i + 1] +
            0.114 * imageData.data[i + 2]) /
          255;
        row[x] = Math.round(luma * 1e5) / 1e5;
      }
      values.push(row);
    }

    return { name: file.name, width, height, values };
  } finally {
    bitmap.close();
  }
}
