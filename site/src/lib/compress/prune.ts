/**
 * Pruning and weight sharing — the other two ways to make weights cheaper.
 * Same contract as quant.ts: flat Float32Array in, new Float32Array out.
 */

import type { Shape } from "./quant";

/** Magnitude threshold below which `fraction` of |values| fall (fraction in [0, 1]). */
export function magnitudeThreshold(values: Float32Array | number[], fraction: number): number {
  if (fraction <= 0) return -1;
  const abs = Float32Array.from(values, Math.abs).sort();
  const k = Math.min(abs.length - 1, Math.floor(fraction * abs.length));
  return abs[k];
}

/** Unstructured: zero the smallest-|w| `fraction` of weights (per tensor, or with a given global threshold). */
export function pruneMagnitude(w: Float32Array, fraction: number, threshold?: number): Float32Array {
  const t = threshold ?? magnitudeThreshold(w, fraction);
  const out = new Float32Array(w.length);
  let kept = 0;
  const budget = w.length - Math.round(fraction * w.length);
  for (let i = 0; i < w.length; i++) {
    // strict > keeps exactly the budget when there are ties at the threshold
    if (Math.abs(w[i]) > t && kept < budget) {
      out[i] = w[i];
      kept++;
    }
  }
  return out;
}

/**
 * N:M semi-structured sparsity: in every run of M consecutive weights along the
 * input dimension, keep the N largest. 2:4 is the pattern NVIDIA GPUs accelerate.
 */
export function pruneNM(w: Float32Array, shape: Shape, n = 2, m = 4): Float32Array {
  const out = new Float32Array(w.length);
  const perChannel = w.length / shape[0];
  for (let c = 0; c < shape[0]; c++) {
    const base = c * perChannel;
    for (let s = 0; s < perChannel; s += m) {
      const idx = [];
      for (let i = s; i < Math.min(s + m, perChannel); i++) idx.push(base + i);
      idx.sort((a, b) => Math.abs(w[b]) - Math.abs(w[a]));
      for (const i of idx.slice(0, n)) out[i] = w[i];
    }
  }
  return out;
}

/** L1 norm of each output channel (the classic filter-importance score). */
export function channelL1(w: Float32Array, shape: Shape): Float32Array {
  const per = w.length / shape[0];
  const norms = new Float32Array(shape[0]);
  for (let c = 0; c < shape[0]; c++) {
    let s = 0;
    for (let i = c * per; i < (c + 1) * per; i++) s += Math.abs(w[i]);
    norms[c] = s / per;
  }
  return norms;
}

/**
 * Structured: remove the lowest-L1 `fraction` of output channels. Returns the
 * channel mask so the caller can also zero the bias (and, for honest size
 * accounting, the matching input slice of the next layer).
 */
export function pruneChannels(w: Float32Array, shape: Shape, fraction: number): { w: Float32Array; keep: Uint8Array } {
  const norms = channelL1(w, shape);
  const order = Array.from(norms.keys()).sort((a, b) => norms[a] - norms[b]);
  const drop = Math.min(shape[0] - 1, Math.round(fraction * shape[0])); // never remove every channel
  const keep = new Uint8Array(shape[0]).fill(1);
  for (const c of order.slice(0, drop)) keep[c] = 0;
  const per = w.length / shape[0];
  const out = new Float32Array(w);
  for (let c = 0; c < shape[0]; c++) if (!keep[c]) out.fill(0, c * per, (c + 1) * per);
  return { w: out, keep };
}

/**
 * Weight sharing: 1-D k-means over the tensor's values (Deep Compression /
 * Core ML palettization). Each weight becomes one of k centroids, so it can be
 * stored as a log2(k)-bit index into a small codebook.
 * Initialised linearly over [min, max] (the Deep Compression paper's best init).
 */
export function kmeansCluster(w: Float32Array, k: number, iters = 12): { w: Float32Array; codebook: Float32Array } {
  let lo = Infinity, hi = -Infinity;
  for (const v of w) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const cb = Float32Array.from({ length: k }, (_, i) => (k === 1 ? (lo + hi) / 2 : lo + ((hi - lo) * i) / (k - 1)));
  const assign = new Uint16Array(w.length);
  const nearest = (v: number) => {
    // centroids stay sorted, so binary search
    let a = 0, b = k - 1;
    while (b - a > 1) { const m = (a + b) >> 1; if (cb[m] <= v) a = m; else b = m; }
    return Math.abs(v - cb[a]) <= Math.abs(v - cb[b]) ? a : b;
  };
  for (let it = 0; it < iters; it++) {
    const sum = new Float64Array(k), cnt = new Uint32Array(k);
    for (let i = 0; i < w.length; i++) { const c = nearest(w[i]); assign[i] = c; sum[c] += w[i]; cnt[c]++; }
    let moved = 0;
    for (let c = 0; c < k; c++) if (cnt[c]) { const n = sum[c] / cnt[c]; moved = Math.max(moved, Math.abs(n - cb[c])); cb[c] = n; }
    cb.sort();
    if (moved < 1e-7) break;
  }
  const out = new Float32Array(w.length);
  for (let i = 0; i < w.length; i++) out[i] = cb[nearest(w[i])];
  return { w: out, codebook: cb };
}

export function sparsity(w: Float32Array): number {
  let z = 0;
  for (const v of w) if (v === 0) z++;
  return z / w.length;
}
