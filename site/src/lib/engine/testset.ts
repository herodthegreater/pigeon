/** The fixed eval set: 500 images packed into 10 x 10 WebP sprite sheets. */

import { INPUT } from "./constants";
import { fetchOk } from "./fetch";

export interface TestImage {
  id: string;
  sheet: number;
  cell: number;
  label: 0 | 1;
  hard: boolean;
  boxes: [number, number, number, number][]; // normalized x0, y0, x1, y1
  ref_logit: number;
  credit: { author: string | null; title: string | null; url: string; license: string };
}

export interface TestSet {
  images: TestImage[];
  pixels: Float32Array[]; // per image, 3 x 224 x 224 in 0..1 (NCHW, one sample)
  sheetUrl: (s: number) => string;
  grid: number;
}

const base = () => (import.meta.env.BASE_URL ?? "/").replace(/\/?$/, "/");

async function decodeSheet(url: string, onBytes?: (n: number) => void): Promise<ImageData> {
  const blob = await fetchOk(url, onBytes).then((r) => r.blob());
  const bmp = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  return ctx.getImageData(0, 0, bmp.width, bmp.height);
}

/** Load the first `count` images (sheets are balanced, so any prefix of 100 is a fair sample). */
export async function loadTestSet(count = 500, onBytes?: (n: number) => void): Promise<TestSet> {
  const meta = await fetchOk(`${base()}testset/testset.json`, onBytes).then((r) => r.json());
  const images: TestImage[] = meta.images.slice(0, count);
  const sheets = [...new Set(images.map((i) => i.sheet))];
  const sheetUrl = (s: number) => `${base()}testset/sheet-${s}.webp`;
  const data = new Map(await Promise.all(sheets.map(async (s) => [s, await decodeSheet(sheetUrl(s), onBytes)] as const)));
  const plane = INPUT * INPUT;
  const pixels = images.map((img) => {
    const d = data.get(img.sheet)!;
    const ox = (img.cell % meta.grid) * INPUT, oy = Math.floor(img.cell / meta.grid) * INPUT;
    const out = new Float32Array(3 * plane);
    for (let y = 0; y < INPUT; y++) {
      for (let x = 0; x < INPUT; x++) {
        const src = ((oy + y) * d.width + ox + x) * 4, dst = y * INPUT + x;
        out[dst] = d.data[src] / 255;
        out[plane + dst] = d.data[src + 1] / 255;
        out[2 * plane + dst] = d.data[src + 2] / 255;
      }
    }
    return out;
  });
  return { images, pixels, sheetUrl, grid: meta.grid };
}

/** Stack images [from, to) into one batch buffer. */
export function batch(ts: TestSet, from: number, to: number): Float32Array {
  const one = 3 * INPUT * INPUT;
  const out = new Float32Array((to - from) * one);
  for (let i = from; i < to; i++) out.set(ts.pixels[i], (i - from) * one);
  return out;
}
