/**
 * The precision lab, ported from cockpit/harness/precision.py by way of the
 * cockpit's browser engine (cockpit/web/engine.js): fake quantization with
 * tensor, channel or group scales, on an integer or a minifloat grid, plus the
 * storage those choices cost. Pure functions, so they run in the worker and in
 * unit tests alike. Every step rounds to float32 (Math.fround) so rounding ties
 * land exactly where PyTorch puts them.
 */

/** Each shared scale is stored as fp16: the metadata a finer granularity pays for. */
export const SCALE_BITS = 16;
/** The native model is the fp16 checkpoint Ultralytics ships. */
export const NATIVE_BITS = 16;

/** Who shares a scale: the whole tensor, each output channel, or groups of this many weights within a channel. */
export type Granularity = "tensor" | "channel" | number;
export type Levels = "int" | "fp";
export type Clip = "max" | "mse";

/** Minifloat layouts [exponent bits, mantissa bits] for the float grid, by total bits (sign included). */
export const FP_FORMATS: Record<number, [number, number]> = { 8: [4, 3], 7: [3, 3], 6: [3, 2], 5: [2, 2], 4: [2, 1], 3: [1, 1], 2: [1, 0] };
const ALPHAS = Array.from({ length: 29 }, (_, i) => Math.fround(0.3 + i * (0.7 / 28)));
const f = Math.fround;

/** Every non-negative magnitude a sign + E + M minifloat stores, scaled so the largest is 1 (4 bits: E2M1). */
export function fpGrid(bits: number): number[] {
  const [e, m] = FP_FORMATS[bits];
  const bias = e > 1 ? (1 << (e - 1)) - 1 : 0;
  const vals = new Set<number>();
  for (let ex = 0; ex < 1 << e; ex++)
    for (let man = 0; man < 1 << m; man++)
      vals.add(ex === 0 ? (man / (1 << m)) * 2 ** (1 - bias) : (1 + man / (1 << m)) * 2 ** (ex - bias));
  const g = Float32Array.from([...vals].sort((a, z) => a - z));
  const top = g[g.length - 1];
  return Array.from(g, (v) => f(v / top));
}

const gridCache = new Map<number, { sg: number[]; mids: number[] }>();
function signedGrid(bits: number) {
  let g = gridCache.get(bits);
  if (!g) {
    const pos = fpGrid(bits);
    const sg = [...pos.slice(1).reverse().map((v) => -v), ...pos];
    g = { sg, mids: sg.slice(1).map((v, i) => f(f(v + sg[i]) / 2)) };
    gridCache.set(bits, g);
  }
  return g;
}

/** Round half to even, like torch.round. */
const rhe = (x: number) => (Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : Math.round(x));

function snap(x: number, scale: number, bits: number, levels: Levels): number {
  if (levels === "fp") {
    const { sg, mids } = signedGrid(bits);
    const v = Math.min(Math.max(f(x / scale), -1), 1);
    let lo = 0, hi = mids.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (mids[mid] < v) lo = mid + 1; else hi = mid; }
    return f(sg[lo] * scale);
  }
  const q = (1 << (bits - 1)) - 1, step = f(scale / q);
  return f(Math.min(Math.max(rhe(f(x / step)), -q), q) * step);
}

/** One scale-sharing row: src[start, end), zero-padded to length `len` (only the 1-bit mean sees the padding). */
function quantRow(src: Float32Array, dst: Float32Array, start: number, end: number, len: number, bits: number, levels: Levels, clip: Clip) {
  if (bits === 1) { // sign times the mean magnitude: no zero, no range to clip
    let s = 0;
    for (let i = start; i < end; i++) s += Math.abs(src[i]);
    s = f(s / len);
    for (let i = start; i < end; i++) dst[i] = src[i] >= 0 ? s : -s;
    return;
  }
  let amax = 1e-12;
  for (let i = start; i < end; i++) amax = Math.max(amax, Math.abs(src[i]));
  let scale = amax;
  if (clip === "mse") {
    let best = Infinity;
    for (const a of ALPHAS) {
      const sc = f(amax * a);
      let err = 0;
      for (let i = start; i < end; i++) { const d = snap(src[i], sc, bits, levels) - src[i]; err += d * d; }
      if (err < best) { best = err; scale = sc; }
    }
  }
  for (let i = start; i < end; i++) dst[i] = snap(src[i], scale, bits, levels);
}

/** [start, end, padded length] of every scale-sharing row of an [O, cols] weight. */
export function rowsOf(O: number, cols: number, gran: Granularity): [number, number, number][] {
  if (gran === "tensor") return [[0, O * cols, O * cols]];
  const g = gran === "channel" ? cols : Math.min(gran, cols);
  const rows: [number, number, number][] = [];
  for (let o = 0; o < O; o++) for (let s = 0; s < cols; s += g) rows.push([o * cols + s, o * cols + Math.min(s + g, cols), g]);
  return rows;
}

/** Fake-quantize an [O, cols] weight (cols = I * kh * kw). At 16 bits or more it is returned untouched. */
export function quantize(w: Float32Array, O: number, cols: number, bits: number, gran: Granularity, levels: Levels, clip: Clip = "max"): Float32Array {
  if (bits >= NATIVE_BITS) return w;
  if (levels === "fp" && !FP_FORMATS[bits]) levels = "int"; // 1 bit has no float layout
  const out = new Float32Array(w.length);
  for (const [s, e, len] of rowsOf(O, cols, gran)) quantRow(w, out, s, e, len, bits, levels, clip);
  return out;
}

export const nScales = (O: number, cols: number, gran: Granularity) =>
  gran === "tensor" ? 1 : gran === "channel" ? O : O * Math.ceil(cols / gran);

/**
 * Storage of one conv in bits: its weights at `bits` plus their fp16 scales, and its
 * bias at fp16 (biases stay native, as in the harness). `outs` and `ins` are the
 * channels stored: all of them, or only the live ones once masked channels are cut out.
 */
export function convBits(outs: number, ins: number, k: number, bits: number, gran: Granularity): number {
  const cols = ins * k, weights = outs * cols;
  const w = bits >= NATIVE_BITS ? weights * NATIVE_BITS : weights * bits + nScales(outs, cols, gran) * SCALE_BITS;
  return w + outs * NATIVE_BITS;
}
