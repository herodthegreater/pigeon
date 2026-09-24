/** The scoreboard, computed exactly like train/metrics.py. */

import { CAM, INPUT } from "./constants";
import type { TestImage } from "./testset";

export interface Score {
  n: number;
  accuracy: number;
  precision: number;
  recall: number;
  hardNegAcc: number;
  pointing: number;
  agreement: number;
}

/** Bilinear upsample (PyTorch align_corners=False semantics) and return the peak's normalized (x, y). */
export function camPeak(cams: Float32Array, i: number): [number, number] {
  const off = i * CAM * CAM;
  const scale = CAM / INPUT;
  let best = -Infinity, bx = 0, by = 0;
  for (let y = 0; y < INPUT; y++) {
    const sy = Math.max(0, (y + 0.5) * scale - 0.5);
    const y0 = Math.min(Math.floor(sy), CAM - 1), y1 = Math.min(y0 + 1, CAM - 1), fy = sy - y0;
    for (let x = 0; x < INPUT; x++) {
      const sx = Math.max(0, (x + 0.5) * scale - 0.5);
      const x0 = Math.min(Math.floor(sx), CAM - 1), x1 = Math.min(x0 + 1, CAM - 1), fx = sx - x0;
      const v =
        (1 - fy) * ((1 - fx) * cams[off + y0 * CAM + x0] + fx * cams[off + y0 * CAM + x1]) +
        fy * ((1 - fx) * cams[off + y1 * CAM + x0] + fx * cams[off + y1 * CAM + x1]);
      if (v > best) { best = v; bx = x; by = y; }
    }
  }
  return [(bx + 0.5) / INPUT, (by + 0.5) / INPUT];
}

export function score(images: TestImage[], logits: Float32Array, cams: Float32Array, refPred?: Uint8Array): Score {
  let correct = 0, tp = 0, fp = 0, fn = 0, hard = 0, hardOk = 0, birds = 0, hits = 0, agree = 0;
  images.forEach((img, i) => {
    const pred = logits[i] > 0 ? 1 : 0;
    if (pred === img.label) correct++;
    if (pred === 1 && img.label === 1) tp++;
    if (pred === 1 && img.label === 0) fp++;
    if (pred === 0 && img.label === 1) fn++;
    if (img.hard) { hard++; if (pred === 0) hardOk++; }
    if (refPred && refPred[i] === pred) agree++;
    if (img.label === 1) {
      birds++;
      const [x, y] = camPeak(cams, i);
      if (img.boxes.some(([x0, y0, x1, y1]) => x0 <= x && x <= x1 && y0 <= y && y <= y1)) hits++;
    }
  });
  const n = images.length;
  return {
    n,
    accuracy: correct / n,
    precision: tp / Math.max(tp + fp, 1),
    recall: tp / Math.max(tp + fn, 1),
    hardNegAcc: hard ? hardOk / hard : NaN,
    pointing: birds ? hits / birds : NaN,
    agreement: refPred ? agree / n : 1,
  };
}

export const predictions = (logits: Float32Array) => Uint8Array.from(logits, (v) => (v > 0 ? 1 : 0));
