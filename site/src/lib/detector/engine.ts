/**
 * The real YOLOv8n, running in TF.js (WebGL). A TypeScript port of the cockpit's
 * browser engine (cockpit/web/engine.js), which is itself a port of the
 * cockpit's Python harness: the forward pass (fused convs, C2f, SPPF, the Detect
 * head and its DFL decode), Ultralytics-style NMS, channel masking, the
 * precision lab, and live-channel accounting of storage and compute.
 *
 * The Coop's photos are already 224 x 224, YOLOv8n's input here, so there is no
 * letterbox: boxes come out in the photo's own pixels.
 */

import * as tf from "@tensorflow/tfjs";
import { convBits, NATIVE_BITS, quantize } from "./quant";
import { nms, type Det } from "./score";
import { granularityOf, nativeRecipe, type DetectorRecipe } from "./recipe";

export interface ConvMeta {
  name: string;
  shape: [number, number, number, number]; // O, I, kh, kw
  stride: number;
  pad: number;
  w: number; // offset of the weights in weights.bin, in fp16 values
  b: number; // offset of the bias
}

export interface GraphNode {
  i: number;
  type: string;
  f: number | number[];
  n?: number;
  add?: boolean[];
  k?: number;
  nc?: number;
  reg_max?: number;
  strides?: number[];
}

export interface PruneStage { stage: string; channels: number; flop_frac: number; param_frac: number }

export interface Manifest {
  imgsz: number;
  bird_class: number;
  weights_file: string;
  weights_bytes: number;
  convs: ConvMeta[];
  graph: GraphNode[];
  head: string;
  prune_stages: PruneStage[];
  params: number;
  other_params: number;
  macs: number;
  preview: string[];
  recorded: { what: string; ultralytics: string; torch: string; date: string; ap: Record<string, { all: number; quick: number }> };
}

/** Per conv: the channels that still carry signal, and its output size. */
export type Live = Map<string, { liveIn: number; liveOut: number; oh: number; ow: number }>;

export interface Cost {
  bytes: number;
  macs: number;
  /** Average bits per stored conv weight, scales included. */
  effBits: number;
}

interface NatConv {
  w: Float32Array; b: Float32Array;
  O: number; I: number; cols: number; kh: number; kw: number;
  stride: number; pad: number; stage: string | null;
  order: Uint32Array; // output channels, weakest (smallest L1 norm) first
}

/** Confidence floor for scoring, as in the harness (Ultralytics' validation default). */
export const CONF_EVAL = 0.001;

const HALF = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    t[h] = e === 0 ? s * 2 ** -14 * (m / 1024) : e === 31 ? (m ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + m / 1024);
  }
  return t;
})();

const stageOf = (name: string) => {
  const p = name.split(".");
  return p[0] === "model" && /^\d+$/.test(p[1]) ? `model.${p[1]}` : null;
};

/** Start TF.js on WebGL. Returns false when this browser (or this worker) has no WebGL. */
export async function startWebGL(): Promise<{ ok: boolean; float32: boolean }> {
  let ok = false;
  // Shapes as uniforms: most ops then compile one GPU program that serves every batch
  // size, instead of one per shape (the probe's single photo, a full batch, a short one).
  tf.env().set("WEBGL_USE_SHAPES_UNIFORMS", true);
  try { ok = await tf.setBackend("webgl"); } catch { ok = false; }
  await tf.ready().catch(() => {});
  ok = ok && tf.getBackend() === "webgl";
  return { ok, float32: ok && tf.env().getBool("WEBGL_RENDER_FLOAT32_ENABLED") };
}

export class Detector {
  readonly S: number;
  private nat = new Map<string, NatConv>();
  private gpu = new Map<string, { w: tf.Tensor4D; b: tf.Tensor1D; key: string; liveOut: number }>();
  private anch: tf.Tensor2D;
  private strd: tf.Tensor2D;
  private proj: tf.Tensor1D;
  private probeLog: { name: string; liveIn: number; oh: number; ow: number }[] | null = null;

  constructor(readonly m: Manifest, weights: ArrayBuffer, private px: Uint8Array) {
    this.S = m.imgsz;
    const u16 = new Uint16Array(weights, 0, weights.byteLength >> 1);
    for (const c of m.convs) {
      const [O, I, kh, kw] = c.shape, cols = I * kh * kw;
      const w = new Float32Array(O * cols), b = new Float32Array(O);
      for (let i = 0; i < w.length; i++) w[i] = HALF[u16[c.w + i]];
      for (let i = 0; i < O; i++) b[i] = HALF[u16[c.b + i]];
      const l1 = new Float64Array(O);
      for (let o = 0; o < O; o++) { let s = 0; for (let j = 0; j < cols; j++) s += Math.abs(w[o * cols + j]); l1[o] = s; }
      const order = Uint32Array.from([...l1.keys()].sort((a, z) => l1[a] - l1[z]));
      this.nat.set(c.name, { w, b, O, I, cols, kh, kw, stride: c.stride, pad: c.pad, stage: stageOf(c.name), order });
    }
    const head = m.graph.find((n) => n.type === "Detect")!;
    const an: number[] = [], st: number[] = [];
    for (const s of head.strides!) {
      const h = this.S / s;
      for (let y = 0; y < h; y++) for (let x = 0; x < h; x++) { an.push(x + 0.5, y + 0.5); st.push(s); }
    }
    this.anch = tf.keep(tf.tensor2d(an, [st.length, 2]));
    this.strd = tf.keep(tf.tensor2d(st, [st.length, 1]));
    this.proj = tf.keep(tf.range(0, head.reg_max!, 1, "float32") as tf.Tensor1D);
    this.apply(nativeRecipe());
  }

  /* ------------------------------ recipes ------------------------------ */

  /** Mask, quantize and upload every conv whose setting changed. */
  apply(r: DetectorRecipe): void {
    const gran = granularityOf(r);
    for (const [name, c] of this.nat) {
      const frac = c.stage ? r.masks[c.stage] ?? 0 : 0;
      const key = r.bits >= NATIVE_BITS ? `${frac}|16` : `${frac}|${r.bits}|${gran}|${r.levels}`;
      const old = this.gpu.get(name);
      if (old?.key === key) continue;
      let w = c.w, b = c.b;
      const k = Math.min(Math.round(frac * c.O), c.O - 1);
      if (k > 0) {
        w = w.slice(); b = b.slice();
        for (let i = 0; i < k; i++) { const o = c.order[i]; w.fill(0, o * c.cols, (o + 1) * c.cols); b[o] = 0; }
      }
      w = quantize(w, c.O, c.cols, r.bits, gran, r.levels);
      let liveOut = 0; // output channels that still carry anything (weights or bias)
      for (let o = 0; o < c.O; o++) {
        let alive = b[o] !== 0;
        for (let j = o * c.cols, e = j + c.cols; !alive && j < e; j++) alive = w[j] !== 0;
        if (alive) liveOut++;
      }
      const tw = tf.tidy(() => tf.tensor4d(w, [c.O, c.I, c.kh, c.kw]).transpose([2, 3, 1, 0]) as tf.Tensor4D);
      if (old) tf.dispose([old.w, old.b]);
      this.gpu.set(name, { w: tf.keep(tw), b: tf.keep(tf.tensor1d(b)), key, liveOut });
    }
  }

  /**
   * One forward pass that records, for every conv, the input channels that carry
   * any signal: a masked producer (zero weights, zero bias, SiLU(0) = 0) leaves a
   * channel exactly zero everywhere. With each conv's live output rows, that is
   * what physically removing the masked channels would leave (harness/mask.py,
   * active_cost), residual adds and concats included.
   */
  probe(): Live {
    this.probeLog = [];
    try {
      tf.tidy(() => { this.forward(tf.randomUniform([1, this.S, this.S, 3], 0, 1, "float32", 14) as tf.Tensor4D); });
      const live: Live = new Map();
      for (const p of this.probeLog) live.set(p.name, { liveIn: p.liveIn, liveOut: this.gpu.get(p.name)!.liveOut, oh: p.oh, ow: p.ow });
      return live;
    } finally {
      this.probeLog = null;
    }
  }

  /**
   * Storage of every weight at the recipe's precision, and compute per photo. With
   * `live` (from a probe) only live channels count; without it every channel is
   * stored and compute is the native count from the export.
   */
  cost(r: DetectorRecipe, live: Live | null): Cost {
    const gran = granularityOf(r);
    let bits = this.m.other_params * NATIVE_BITS, macs = 0, weights = 0, biasBits = 0;
    for (const [name, c] of this.nat) {
      const k = c.kh * c.kw, l = live?.get(name);
      const outs = l ? l.liveOut : c.O, ins = l ? l.liveIn : c.I;
      bits += convBits(outs, ins, k, r.bits, gran);
      biasBits += outs * NATIVE_BITS;
      weights += outs * ins * k;
      if (l) macs += l.oh * l.ow * outs * ins * k;
    }
    return { bytes: bits / 8, macs: live ? macs : this.m.macs, effBits: (bits - biasBits - this.m.other_params * NATIVE_BITS) / Math.max(weights, 1) };
  }

  /* ------------------------------ forward ------------------------------ */

  private convRaw(x: tf.Tensor4D, name: string): tf.Tensor4D {
    const g = this.gpu.get(name)!, c = this.nat.get(name)!;
    const y = tf.add(tf.conv2d(x, g.w, c.stride, c.pad), g.b) as tf.Tensor4D;
    if (this.probeLog) {
      const sums = tf.tidy(() => tf.sum(tf.abs(x), [0, 1, 2]).dataSync());
      let liveIn = 0;
      for (const v of sums) if (v > 0) liveIn++;
      this.probeLog.push({ name, liveIn, oh: y.shape[1], ow: y.shape[2] });
    }
    return y;
  }

  private conv(x: tf.Tensor4D, p: string): tf.Tensor4D {
    const t = this.convRaw(x, `${p}.conv`);
    return tf.mul(t, tf.sigmoid(t)) as tf.Tensor4D; // SiLU
  }

  private c2f(x: tf.Tensor4D, p: string, n: GraphNode): tf.Tensor4D {
    const [a, b] = tf.split(this.conv(x, `${p}.cv1`), 2, 3) as tf.Tensor4D[];
    const ys = [a, b];
    let last = b;
    for (let k = 0; k < n.n!; k++) {
      let h = this.conv(this.conv(last, `${p}.m.${k}.cv1`), `${p}.m.${k}.cv2`);
      if (n.add![k]) h = tf.add(last, h) as tf.Tensor4D;
      ys.push(h);
      last = h;
    }
    return this.conv(tf.concat(ys, 3), `${p}.cv2`);
  }

  private sppf(x: tf.Tensor4D, p: string, k: number): tf.Tensor4D {
    const y0 = this.conv(x, `${p}.cv1`);
    const y1 = tf.maxPool(y0, k, 1, "same"), y2 = tf.maxPool(y1, k, 1, "same"), y3 = tf.maxPool(y2, k, 1, "same");
    return this.conv(tf.concat([y0, y1, y2, y3], 3), `${p}.cv2`);
  }

  /** Raw head outputs: box distributions [B, A, 4 * reg_max] and class logits [B, A, nc]. */
  private forward(x: tf.Tensor4D): [tf.Tensor3D, tf.Tensor3D] {
    const outs: tf.Tensor4D[] = [];
    let y = x;
    for (const n of this.m.graph) {
      const p = `model.${n.i}`;
      const from = (j: number) => (j === -1 ? y : outs[j]);
      if (n.type === "Detect") {
        const bx: tf.Tensor3D[] = [], cl: tf.Tensor3D[] = [];
        (n.f as number[]).map(from).forEach((t, j) => {
          const b = this.convRaw(this.conv(this.conv(t, `${p}.cv2.${j}.0`), `${p}.cv2.${j}.1`), `${p}.cv2.${j}.2`);
          const c = this.convRaw(this.conv(this.conv(t, `${p}.cv3.${j}.0`), `${p}.cv3.${j}.1`), `${p}.cv3.${j}.2`);
          const [B, h, w] = b.shape;
          bx.push(b.reshape([B, h * w, 4 * n.reg_max!]));
          cl.push(c.reshape([B, h * w, n.nc!]));
        });
        return [tf.concat(bx, 1), tf.concat(cl, 1)];
      }
      if (n.type === "Concat") y = tf.concat((n.f as number[]).map(from), 3);
      else {
        const inp = from(n.f as number);
        if (n.type === "Conv") y = this.conv(inp, p);
        else if (n.type === "C2f") y = this.c2f(inp, p, n);
        else if (n.type === "SPPF") y = this.sppf(inp, p, n.k!);
        else if (n.type === "Upsample") y = tf.image.resizeNearestNeighbor(inp, [inp.shape[1] * 2, inp.shape[2] * 2]);
        else throw new Error(`YOLOv8n has no ${n.type} layer this engine knows`);
      }
      outs.push(y);
    }
    throw new Error("the model graph has no Detect head");
  }

  /* --------------------- detect: decode + NMS --------------------- */

  /**
   * Bird detections for photos `idxs`, in batches. Returns null as soon as `stale()`
   * says a newer request has arrived. Yields between batches so messages get through.
   */
  async detect(idxs: number[], batch: number, onBatch?: (done: number) => void, stale?: () => boolean): Promise<Det[][] | null> {
    const S = this.S, one = S * S * 3, bird = this.m.bird_class, reg = this.m.graph.find((n) => n.type === "Detect")!.reg_max!;
    const out: Det[][] = [];
    for (let s = 0; s < idxs.length; s += batch) {
      if (stale?.()) return null;
      const ids = idxs.slice(s, s + batch), B = ids.length;
      const flat = new Uint8Array(B * one);
      ids.forEach((i, j) => flat.set(this.px.subarray(i * one, (i + 1) * one), j * one));
      const [xyxy, mx, am] = tf.tidy(() => {
        const x = tf.div(tf.tensor4d(flat, [B, S, S, 3], "int32").toFloat(), 255) as tf.Tensor4D;
        const [box, cls] = this.forward(x), A = box.shape[1];
        const d = tf.sum(tf.mul(tf.softmax(box.reshape([B, A, 4, reg])), this.proj), 3);
        const lt = d.slice([0, 0, 0], [B, A, 2]), rb = d.slice([0, 0, 2], [B, A, 2]);
        return [
          tf.concat([tf.mul(tf.sub(this.anch, lt), this.strd), tf.mul(tf.add(this.anch, rb), this.strd)], 2),
          tf.max(cls, 2),
          tf.argMax(cls, 2),
        ];
      });
      const [bx, mv, av] = await Promise.all([xyxy.data(), mx.data(), am.data()]);
      tf.dispose([xyxy, mx, am]);
      const A = mv.length / B;
      const clip = (v: number) => Math.min(Math.max(v, 0), S);
      for (let j = 0; j < B; j++) {
        const cand: Det[] = [];
        for (let a = 0; a < A; a++) {
          const k = j * A + a;
          if (av[k] !== bird) continue; // a box counts only when bird is its top class
          const cf = 1 / (1 + Math.exp(-mv[k]));
          if (cf <= CONF_EVAL) continue;
          cand.push([bx[k * 4], bx[k * 4 + 1], bx[k * 4 + 2], bx[k * 4 + 3], cf]);
        }
        // NMS on the raw boxes, then clip to the photo, in Ultralytics' order
        out.push(nms(cand).map((b) => [clip(b[0]), clip(b[1]), clip(b[2]), clip(b[3]), b[4]]));
      }
      onBatch?.(Math.min(s + B, idxs.length));
      await new Promise((r) => setTimeout(r, 0));
    }
    return out;
  }
}
