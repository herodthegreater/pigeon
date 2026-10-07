/**
 * Honest size accounting: the bytes a real file would take, including every
 * scale, zero-point, codebook and sparsity index — not just "bits x weights".
 */

import type { Granularity } from "./quant";

export type Encoding =
  | { kind: "fp32" }
  | { kind: "float"; bits: number; scaleBitsPerBlock?: number; block?: number; tensorScaleBits?: number }
  | { kind: "int"; bits: number; granularity: Granularity; symmetric: boolean; scaleBits?: number }
  | { kind: "codebook"; k: number; entryBits?: number }
  | { kind: "ternary"; granularity: Granularity; scaleBits?: number };

export type Sparsity =
  | { kind: "dense" }
  | { kind: "unstructured"; nonzero: number } // stored as bitmap + packed nonzeros (or dense, whichever is smaller)
  | { kind: "nm"; n: number; m: number } // N of every M kept, log2(C(M,N))-ish index bits per group
  | { kind: "channels"; keptFraction: number }; // channels physically removed

export interface TensorInfo {
  numel: number;
  outChannels: number;
}

function scaleCount(t: TensorInfo, g: Granularity): number {
  if (g.kind === "tensor") return 1;
  if (g.kind === "channel") return t.outChannels;
  const per = t.numel / t.outChannels;
  return t.outChannels * Math.ceil(per / g.size);
}

/** Bits to store one tensor's weights under an encoding and sparsity pattern. */
export function tensorBits(t: TensorInfo, enc: Encoding, sp: Sparsity = { kind: "dense" }): number {
  let numel = t.numel;
  let outCh = t.outChannels;
  if (sp.kind === "channels") {
    outCh = Math.max(1, Math.round(outCh * sp.keptFraction));
    numel = Math.round((numel / t.outChannels) * outCh);
  }
  const info = { numel, outChannels: outCh };

  let perWeight: number;
  let overhead: number;
  switch (enc.kind) {
    case "fp32":
      perWeight = 32; overhead = 0; break;
    case "float":
      perWeight = enc.bits;
      overhead = (enc.block ? Math.ceil(numel / enc.block) * (enc.scaleBitsPerBlock ?? 8) : 0) + (enc.tensorScaleBits ?? 0);
      break;
    case "int":
      perWeight = enc.bits;
      overhead = scaleCount(info, enc.granularity) * ((enc.scaleBits ?? 16) + (enc.symmetric ? 0 : enc.bits));
      break;
    case "ternary":
      perWeight = Math.log2(3); // 1.58 bits: BitNet b1.58's namesake
      overhead = scaleCount(info, enc.granularity) * (enc.scaleBits ?? 16);
      break;
    case "codebook":
      perWeight = Math.ceil(Math.log2(Math.max(2, enc.k)));
      overhead = enc.k * (enc.entryBits ?? 16);
      break;
  }

  if (sp.kind === "unstructured") {
    const nnz = Math.round(sp.nonzero * numel);
    const sparse = numel /* bitmap */ + nnz * perWeight;
    return Math.min(sparse, numel * perWeight) + overhead;
  }
  if (sp.kind === "nm") {
    const kept = (numel / sp.m) * sp.n;
    const indexBits = Math.ceil(Math.log2(sp.m)); // per kept weight, as in NVIDIA 2:4 (2 bits)
    return kept * (perWeight + indexBits) + overhead;
  }
  return numel * perWeight + overhead;
}

export const bytes = (bits: number) => Math.ceil(bits / 8);

export function formatBytes(b: number): string {
  if (b >= 1e6) return `${(b / 1e6).toFixed(2)} MB`;
  if (b >= 1e3) return `${(b / 1e3).toFixed(1)} KB`;
  return `${b} B`;
}
