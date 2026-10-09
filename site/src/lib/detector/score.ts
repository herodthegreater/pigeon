/**
 * How the Coop scores YOLOv8n: Ultralytics-style NMS, the cockpit's single-class
 * AP@50 (harness/evaluate.py's BirdmAPEvaluator, VOC all-points interpolation),
 * and the bird's condition from how much of the native AP a recipe keeps.
 * Pure functions: the worker and the unit tests share them.
 */

import type { Condition } from "../ui/mascot";
import { keptCondition } from "../ui/birds";

/** [x0, y0, x1, y1] in pixels. */
export type Box = [number, number, number, number];
/** A detection: a box and its confidence. */
export type Det = [number, number, number, number, number];

export const IOU_NMS = 0.7;
export const MAX_DET = 300;
export const MAX_NMS = 30000;

export function iou(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const i = iw * ih;
  return i / Math.max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9);
}

/** Greedy NMS, highest confidence first. Sorts `cands` in place; returns the kept detections. */
export function nms(cands: Det[], iouThr = IOU_NMS): Det[] {
  cands.sort((a, z) => z[4] - a[4]);
  if (cands.length > MAX_NMS) cands.length = MAX_NMS;
  const keep: Det[] = [], dead = new Uint8Array(cands.length);
  for (let i = 0; i < cands.length && keep.length < MAX_DET; i++) {
    if (dead[i]) continue;
    const a = cands[i];
    keep.push(a);
    for (let j = i + 1; j < cands.length; j++) if (!dead[j] && iou(a, cands[j]) > iouThr) dead[j] = 1;
  }
  return keep;
}

/**
 * Bird AP@50 over a set of photos. `gt[i]` and `preds[i]` are photo i's true boxes
 * and detections; photos without birds still count, as false alarms.
 */
export function ap50(gt: Box[][], preds: Det[][]): number {
  const scores: number[] = [], tp: number[] = [];
  let nGt = 0;
  for (let i = 0; i < gt.length; i++) {
    const g = gt[i], matched = new Uint8Array(g.length);
    nGt += g.length;
    for (const b of [...(preds[i] ?? [])].sort((a, z) => z[4] - a[4])) {
      scores.push(b[4]);
      if (g.length) {
        let j = 0, best = -1;
        g.forEach((t, k) => { const v = iou(b, t); if (v > best) { best = v; j = k; } });
        if (best >= 0.5 && !matched[j]) { matched[j] = 1; tp.push(1); continue; }
      }
      tp.push(0);
    }
  }
  if (!nGt || !scores.length) return 0;
  const order = scores.map((_, i) => i).sort((a, z) => scores[z] - scores[a]);
  const mrec = [0], mpre = [0];
  let ctp = 0, cfp = 0;
  for (const i of order) {
    if (tp[i]) ctp++; else cfp++;
    mrec.push(ctp / nGt);
    mpre.push(ctp / Math.max(ctp + cfp, 1e-9));
  }
  mrec.push(1); mpre.push(0);
  for (let i = mpre.length - 1; i > 0; i--) mpre[i - 1] = Math.max(mpre[i - 1], mpre[i]);
  let s = 0;
  for (let i = 0; i < mrec.length - 1; i++) if (mrec[i + 1] !== mrec[i]) s += (mrec[i + 1] - mrec[i]) * mpre[i + 1];
  return s;
}

/** For one photo: true boxes found (IoU >= 0.5), missed, and detections that matched nothing. */
export function matchCounts(dets: Det[], gt: Box[]): { found: number; missed: number; falseAlarms: number } {
  const used = new Set<number>();
  let found = 0;
  for (const b of [...dets].sort((a, z) => z[4] - a[4])) {
    let j = -1, best = -1;
    gt.forEach((t, k) => { const v = iou(b, t); if (v > best) { best = v; j = k; } });
    if (best >= 0.5 && !used.has(j)) { used.add(j); found++; }
  }
  return { found, missed: gt.length - found, falseAlarms: dets.length - found };
}

/** The detector uses the site's one bird rule (lib/ui/birds.ts): healthy at 95% of native AP or more, bones at 60% or less. */
export { HEALTHY_KEEP, SKELETON_KEEP } from "../ui/birds";

/**
 * The detector's condition from AP kept (ap / nativeAp). `starve` runs from 0 at the
 * healthy line to 1 at the skeleton line; it drives how thin mascotSVG draws the bird.
 */
export function detectorCondition(ap: number, nativeAp: number): { condition: Condition; starve: number; kept: number } {
  const kept = nativeAp > 0 ? ap / nativeAp : 0;
  return { ...keptCondition(kept), kept };
}
