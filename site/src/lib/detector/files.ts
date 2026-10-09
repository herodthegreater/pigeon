/**
 * What the detector worker downloads: YOLOv8n's manifest and fp16 weights, and
 * the Coop's test set (testset.json and its sprite sheets). The photos are cut
 * from the sheets into one RGB buffer, 224 x 224 x 3 bytes per photo, with their
 * true bird boxes in pixels.
 */

import type { TestImage } from "../engine/testset";
import type { Box } from "./score";
import type { Manifest } from "./engine";

export interface Photos {
  images: TestImage[];
  grid: number;
  size: number;
  px: Uint8Array;
  gt: Box[][];
}

const base = () => (import.meta.env.BASE_URL ?? "/").replace(/\/?$/, "/");
export const MODEL_DIR = () => `${base()}coop/yolov8n/`;
export const TESTSET_URL = () => `${base()}testset/testset.json`;
export const sheetUrl = (s: number) => `${base()}testset/sheet-${s}.webp`;

async function open(url: string): Promise<Response> {
  const file = url.split("/").pop();
  let r: Response;
  try {
    r = await fetch(url);
  } catch {
    throw new Error(`couldn't download ${file} (network error)`);
  }
  if (!r.ok) throw new Error(`couldn't download ${file} (HTTP ${r.status})`);
  return r;
}

/**
 * Download several files at once, reporting bytes as they arrive against the total
 * the server announced (0 when a server doesn't say). Resolves to the raw bytes.
 */
export async function downloadAll(urls: string[], onBytes: (got: number, total: number) => void): Promise<ArrayBuffer[]> {
  const responses = await Promise.all(urls.map(open));
  const sizes = responses.map((r) => Number(r.headers.get("content-length")) || 0);
  const total = sizes.every((n) => n > 0) ? sizes.reduce((a, n) => a + n, 0) : 0;
  let got = 0;
  return Promise.all(
    responses.map(async (r, i) => {
      if (!r.body) { const b = await r.arrayBuffer(); got += b.byteLength; onBytes(got, total); return b; }
      const chunks: Uint8Array[] = [], reader = r.body.getReader();
      for (;;) {
        let step: ReadableStreamReadResult<Uint8Array>;
        try {
          step = await reader.read();
        } catch {
          throw new Error(`the download of ${urls[i].split("/").pop()} was cut off`);
        }
        if (step.done) break;
        chunks.push(step.value);
        got += step.value.byteLength;
        onBytes(got, total);
      }
      return new Blob(chunks as BlobPart[]).arrayBuffer();
    }),
  );
}

const parse = <T>(buf: ArrayBuffer, file: string): T => {
  try {
    return JSON.parse(new TextDecoder().decode(buf));
  } catch {
    throw new Error(`${file} is damaged (it isn't valid JSON)`);
  }
};

export const parseManifest = (buf: ArrayBuffer) => parse<Manifest>(buf, "model.json");
export const parseTestSet = (buf: ArrayBuffer) => parse<{ grid: number; size: number; images: TestImage[] }>(buf, "testset.json");

/** Cut every photo out of its decoded sheet. `onPhoto` reports progress per sheet. */
export async function cutPhotos(meta: { grid: number; size: number; images: TestImage[] }, sheets: Map<number, ArrayBuffer>, onSheet?: (done: number, total: number) => void): Promise<Photos> {
  const { grid, size, images } = meta, one = size * size * 3;
  const px = new Uint8Array(images.length * one);
  let done = 0;
  for (const [s, buf] of sheets) {
    let bmp: ImageBitmap;
    try {
      bmp = await createImageBitmap(new Blob([buf], { type: "image/webp" }), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
    } catch {
      throw new Error(`couldn't decode sheet-${s}.webp (this browser may not read WebP images)`);
    }
    const canvas = new OffscreenCanvas(bmp.width, bmp.height);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("this browser can't draw images off screen (no OffscreenCanvas 2D)");
    ctx.drawImage(bmp, 0, 0);
    bmp.close();
    const d = ctx.getImageData(0, 0, canvas.width, canvas.height);
    images.forEach((img, i) => {
      if (img.sheet !== s) return;
      const ox = (img.cell % grid) * size, oy = Math.floor(img.cell / grid) * size;
      let k = i * one;
      for (let y = 0; y < size; y++) {
        let src = ((oy + y) * d.width + ox) * 4;
        for (let x = 0; x < size; x++, src += 4) { px[k++] = d.data[src]; px[k++] = d.data[src + 1]; px[k++] = d.data[src + 2]; }
      }
    });
    onSheet?.(++done, sheets.size);
  }
  const gt = images.map((img) => img.boxes.map((b) => b.map((v) => v * size) as Box));
  return { images, grid, size, px, gt };
}
