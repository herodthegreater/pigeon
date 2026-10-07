/**
 * Live, training-free weight compression — the browser half of Pigeon.
 *
 * Every function takes a weight tensor as a flat Float32Array plus its shape
 * ([out, in/groups, kh, kw] for convs, [out, in] for linear) and returns a NEW
 * array holding the values a real compressed model would compute with
 * ("fake quantization": rounded to the encodable values, stored as float).
 * Accuracy measured on these arrays transfers exactly to a real export that
 * uses the same scheme.
 */

export type Shape = number[];

export type Granularity =
  | { kind: "tensor" }
  | { kind: "channel" } // one scale per output channel (axis 0)
  | { kind: "group"; size: number }; // one scale per `size` consecutive weights within a channel

export interface IntQuantOptions {
  bits: number; // 1..16 (1 = binary sign, 2 with ternary=true = {-1, 0, +1})
  granularity: Granularity;
  symmetric: boolean; // false = asymmetric: scale + zero-point
  rounding?: "nearest" | "stochastic";
  ternary?: boolean;
  /** clip the range to this fraction of max|w| before quantizing (1 = no clipping) */
  clip?: number;
  rng?: () => number;
}

/** Split a tensor into the index ranges that share one scale. */
export function scaleGroups(n: number, shape: Shape, g: Granularity): Array<[number, number]> {
  if (g.kind === "tensor") return [[0, n]];
  const out = shape[0];
  const perChannel = n / out;
  const ranges: Array<[number, number]> = [];
  for (let c = 0; c < out; c++) {
    const start = c * perChannel;
    if (g.kind === "channel") {
      ranges.push([start, start + perChannel]);
    } else {
      for (let s = start; s < start + perChannel; s += g.size) ranges.push([s, Math.min(s + g.size, start + perChannel)]);
    }
  }
  return ranges;
}

/** Round half to even — what hardware and numpy do; avoids a systematic upward bias. */
export function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

export function quantizeInt(w: Float32Array, shape: Shape, o: IntQuantOptions): Float32Array {
  const q = new Float32Array(w.length);
  const rnd = o.rounding === "stochastic" ? (x: number) => Math.floor(x + (o.rng ?? Math.random)()) : roundHalfEven;
  const clip = o.clip ?? 1;
  for (const [a, b] of scaleGroups(w.length, shape, o.granularity)) {
    let lo = Infinity, hi = -Infinity, absMax = 0, absSum = 0;
    for (let i = a; i < b; i++) {
      const v = w[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
      const av = Math.abs(v);
      if (av > absMax) absMax = av;
      absSum += av;
    }
    if (o.bits >= 16 || !(absMax > 0)) {
      q.set(w.subarray(a, b), a);
      continue;
    }
    if (o.bits === 1) {
      // binary: sign(w) * mean|w| (XNOR-Net style)
      const s = absSum / (b - a);
      for (let i = a; i < b; i++) q[i] = w[i] >= 0 ? s : -s;
      continue;
    }
    if (o.ternary) {
      // ternary: threshold at 0.7 * mean|w|, scale = mean of surviving |w| (TWN)
      const t = (0.7 * absSum) / (b - a);
      let sum = 0, cnt = 0;
      for (let i = a; i < b; i++) if (Math.abs(w[i]) > t) { sum += Math.abs(w[i]); cnt++; }
      const s = cnt ? sum / cnt : 0;
      for (let i = a; i < b; i++) q[i] = Math.abs(w[i]) > t ? Math.sign(w[i]) * s : 0;
      continue;
    }
    if (o.symmetric) {
      const qmax = 2 ** (o.bits - 1) - 1;
      const scale = (absMax * clip) / qmax;
      for (let i = a; i < b; i++) q[i] = Math.max(-qmax, Math.min(qmax, rnd(w[i] / scale))) * scale;
    } else {
      const levels = 2 ** o.bits - 1;
      const mid = (lo + hi) / 2, half = ((hi - lo) / 2) * clip;
      const l = mid - half, h = mid + half;
      const scale = (h - l) / levels || 1;
      const zp = roundHalfEven(-l / scale);
      for (let i = a; i < b; i++) q[i] = (Math.max(0, Math.min(levels, rnd(w[i] / scale) + zp)) - zp) * scale;
    }
  }
  return q;
}

/* ---------------------------------------------------------------- floats */

export interface MiniFloat {
  name: string;
  exp: number; // exponent bits
  man: number; // mantissa bits
  bias: number;
  max: number; // largest finite value
}

export const FORMATS: Record<string, MiniFloat> = {
  fp16: { name: "FP16", exp: 5, man: 10, bias: 15, max: 65504 },
  bf16: { name: "BF16", exp: 8, man: 7, bias: 127, max: 3.3895313892515355e38 },
  fp8_e4m3: { name: "FP8 E4M3", exp: 4, man: 3, bias: 7, max: 448 },
  fp8_e5m2: { name: "FP8 E5M2", exp: 5, man: 2, bias: 15, max: 57344 },
  fp4_e2m1: { name: "FP4 E2M1", exp: 2, man: 1, bias: 1, max: 6 },
};

/** Round one value to the nearest representable value of a minifloat (saturating, with subnormals). */
export function roundMiniFloat(x: number, f: MiniFloat): number {
  if (x === 0 || !Number.isFinite(x)) return x === 0 ? 0 : Math.sign(x) * f.max;
  const s = Math.sign(x);
  const a = Math.min(Math.abs(x), f.max);
  const eMin = 1 - f.bias; // smallest normal exponent; below it, subnormal spacing is fixed
  const e = Math.max(Math.floor(Math.log2(a)), eMin);
  const step = 2 ** (e - f.man);
  return s * Math.min(roundHalfEven(a / step) * step, f.max);
}

export interface FloatQuantOptions {
  format: keyof typeof FORMATS;
  /** how values are scaled into the format's range before rounding */
  scaling:
    | { kind: "none" } // cast as-is (FP16 / BF16)
    | { kind: "tensor" } // one FP32 scale per tensor (typical FP8)
    | { kind: "mx"; block: number } // MXFP: power-of-two (E8M0) scale per block (MXFP4 block = 32)
    | { kind: "nv"; block: number }; // NVFP4: E4M3 scale per block (16) + FP32 scale per tensor
}

export function quantizeFloat(w: Float32Array, shape: Shape, o: FloatQuantOptions): Float32Array {
  const f = FORMATS[o.format];
  const q = new Float32Array(w.length);
  const sc = o.scaling;
  const rangeMax = (a: number, b: number) => {
    let m = 0;
    for (let i = a; i < b; i++) m = Math.max(m, Math.abs(w[i]));
    return m;
  };
  if (sc.kind === "none") {
    for (let i = 0; i < w.length; i++) q[i] = roundMiniFloat(w[i], f);
    return q;
  }
  if (sc.kind === "tensor") {
    const s = rangeMax(0, w.length) / f.max || 1;
    for (let i = 0; i < w.length; i++) q[i] = roundMiniFloat(w[i] / s, f) * s;
    return q;
  }
  const groups = scaleGroups(w.length, shape, { kind: "group", size: sc.block });
  const emaxElem = Math.floor(Math.log2(f.max));
  const tensorScale = sc.kind === "nv" ? rangeMax(0, w.length) / (f.max * FORMATS.fp8_e4m3.max) || 1 : 1;
  for (const [a, b] of groups) {
    const amax = rangeMax(a, b);
    if (amax === 0) continue;
    const s =
      sc.kind === "mx"
        ? 2 ** (Math.floor(Math.log2(amax)) - emaxElem) // OCP MX: shared E8M0 exponent
        : roundMiniFloat(amax / f.max / tensorScale, FORMATS.fp8_e4m3) * tensorScale || tensorScale;
    for (let i = a; i < b; i++) q[i] = roundMiniFloat(w[i] / s, f) * s;
  }
  return q;
}

/* ------------------------------------------------------------- analysis */

export function mse(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s / a.length;
}

export function distinctValues(a: Float32Array, groupRange?: [number, number]): number {
  const [s, e] = groupRange ?? [0, a.length];
  return new Set(a.subarray(s, e)).size;
}
