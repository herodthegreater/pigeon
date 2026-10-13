/**
 * A compression recipe: an ordered list of steps, each applied to the weight
 * tensors of the chosen layer kinds. This is what the Coop lets people stack.
 */

import { kmeansCluster, pruneChannels, pruneMagnitude, pruneNM, sparsity } from "../compress/prune";
import { quantizeFloat, quantizeInt, FORMATS, type FloatQuantOptions, type Granularity } from "../compress/quant";
import { tensorBits, type Encoding, type Sparsity } from "../compress/size";
import type { Kind, TensorMeta, Weights } from "./model";

export const ALL_KINDS: Kind[] = ["conv", "depthwise", "pointwise", "squeeze_excite", "head"];

export type Step =
  | { op: "int"; parts: Kind[]; bits: number; granularity: Granularity; symmetric: boolean; ternary?: boolean; clip?: number }
  | { op: "float"; parts: Kind[]; format: FloatQuantOptions["format"]; scaling: FloatQuantOptions["scaling"] }
  | { op: "prune"; parts: Kind[]; method: "magnitude" | "nm" | "channels"; amount: number }
  | { op: "cluster"; parts: Kind[]; k: number };

export interface Recipe {
  steps: Step[];
}

export interface TensorReport {
  name: string;
  kind: Kind;
  numel: number;
  bits: number;
  sparsity: number;
}

export interface Applied {
  weights: Weights;
  totalBytes: number;
  originalBytes: number;
  tensors: TensorReport[];
}

function encodingOf(step: Step): Encoding | null {
  switch (step.op) {
    case "int":
      if (step.bits >= 16) return null;
      return step.ternary
        ? { kind: "ternary", granularity: step.granularity }
        : { kind: "int", bits: step.bits, granularity: step.granularity, symmetric: step.symmetric };
    case "float": {
      const f = FORMATS[step.format];
      const bits = 1 + f.exp + f.man;
      const s = step.scaling;
      if (s.kind === "mx") return { kind: "float", bits, block: s.block, scaleBitsPerBlock: 8 };
      if (s.kind === "nv") return { kind: "float", bits, block: s.block, scaleBitsPerBlock: 8, tensorScaleBits: 32 };
      return { kind: "float", bits, tensorScaleBits: s.kind === "tensor" ? 32 : 0 };
    }
    case "cluster":
      return { kind: "codebook", k: step.k };
    default:
      return null;
  }
}

export function applyRecipe(tensors: TensorMeta[], original: Weights, recipe: Recipe): Applied {
  const weights: Weights = new Map();
  const reports: TensorReport[] = [];
  const keptChannels = new Map<string, Uint8Array>(); // layer -> channel mask, to zero biases
  let totalBits = 0, originalBits = 0;

  for (const t of tensors.filter((t) => t.role === "weight")) {
    let w = original.get(t.name)!;
    let enc: Encoding = { kind: "fp32" };
    let sp: Sparsity = { kind: "dense" };
    for (const step of recipe.steps) {
      if (!step.parts.includes(t.kind)) continue;
      switch (step.op) {
        case "int":
          if (step.bits < 16) w = quantizeInt(w, t.shape, step);
          break;
        case "float":
          w = quantizeFloat(w, t.shape, step);
          break;
        case "cluster":
          w = kmeansCluster(w, step.k).w;
          break;
        case "prune":
          // 2:4 always removes half, whatever amount the step last had (its label names no amount).
          if (step.method !== "nm" && step.amount <= 0) break;
          if (step.method === "magnitude") {
            w = pruneMagnitude(w, step.amount);
            sp = { kind: "unstructured", nonzero: 1 - step.amount };
          } else if (step.method === "nm") {
            w = pruneNM(w, t.shape, 2, 4);
            sp = { kind: "nm", n: 2, m: 4 };
          } else if (t.shape[0] > 8) {
            const r = pruneChannels(w, t.shape, step.amount);
            w = r.w;
            keptChannels.set(t.layer, r.keep);
            sp = { kind: "channels", keptFraction: r.keep.reduce((a, b) => a + b, 0) / t.shape[0] };
          }
          break;
      }
      enc = encodingOf(step) ?? enc;
    }
    if (sp.kind === "unstructured") sp = { kind: "unstructured", nonzero: 1 - sparsity(w) };
    weights.set(t.name, w);
    const bits = tensorBits({ numel: t.numel, outChannels: t.shape[0] }, enc, sp);
    totalBits += bits;
    originalBits += t.numel * 32;
    reports.push({ name: t.name, kind: t.kind, numel: t.numel, bits, sparsity: sparsity(w) });
  }

  // biases stay FP32 (int32 in real integer kernels) — but vanish with their channel
  for (const t of tensors.filter((t) => t.role === "bias")) {
    const keep = keptChannels.get(t.layer);
    let b = original.get(t.name)!;
    let n = t.numel;
    if (keep) {
      b = b.map((v, i) => (keep[i] ? v : 0));
      n = keep.reduce((a, c) => a + c, 0);
    }
    weights.set(t.name, b);
    totalBits += n * 32;
    originalBits += t.numel * 32;
  }
  return { weights, totalBytes: Math.ceil(totalBits / 8), originalBytes: originalBits / 8, tensors: reports };
}
