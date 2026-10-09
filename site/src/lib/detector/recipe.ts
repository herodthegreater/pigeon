/**
 * A YOLOv8n recipe: one precision setting for every conv, plus a fraction of
 * channels masked in each backbone and neck stage. Plain data, so it crosses
 * to the worker and back as JSON.
 */

import type { Granularity, Levels } from "./quant";

export interface DetectorRecipe {
  /** Bits per weight: 1 to 8, or 16 for the native fp16 weights. */
  bits: number;
  granularity: "tensor" | "channel" | "group";
  /** Weights per scale when granularity is "group". */
  group: number;
  levels: Levels;
  /** Stage ("model.4") -> fraction of its channels masked, 0 to 0.9. */
  masks: Record<string, number>;
}

/** The quick estimate scores the first 100 photos; the test set's sheets are balanced, so any 100 is a fair sample. */
export const QUICK = 100;

export const GROUP_SIZES = [16, 32, 64, 128] as const;
export const MAX_MASK = 0.9;

export const nativeRecipe = (): DetectorRecipe => ({ bits: 16, granularity: "channel", group: 32, levels: "int", masks: {} });

export const isNative = (r: DetectorRecipe) => r.bits >= 16 && !Object.values(r.masks).some((v) => v > 0);

export const granularityOf = (r: DetectorRecipe): Granularity => (r.granularity === "group" ? r.group : r.granularity);

/** Normalized, so equal recipes share a cache entry whatever order their keys came in. */
export function keyOf(r: DetectorRecipe): string {
  const masks = Object.entries(r.masks).filter(([, v]) => v > 0).sort(([a], [b]) => a.localeCompare(b));
  return r.bits >= 16 ? JSON.stringify([16, masks]) : JSON.stringify([r.bits, granularityOf(r), r.levels, masks]);
}

/** "L4" for "model.4": Ultralytics numbers the layers of the model in order. */
export const layerName = (stage: string) => stage.replace("model.", "L");

/** A short label for the runs table: "INT4 per channel · cut L4 20%, L6 10%". */
export function describe(r: DetectorRecipe): string {
  const g = r.granularity === "group" ? `groups of ${r.group}` : r.granularity === "tensor" ? "one scale per tensor" : "per channel";
  const precision = r.bits >= 16 ? "Native fp16" : r.bits === 1 ? `1-bit, ${g}` : `${r.levels === "fp" ? "FP" : "INT"}${r.bits} ${g}`;
  const cuts = Object.entries(r.masks)
    .filter(([, v]) => v > 0)
    .sort(([a], [b]) => Number(a.split(".")[1]) - Number(b.split(".")[1]))
    .map(([s, v]) => `${layerName(s)} ${Math.round(v * 100)}%`);
  if (!cuts.length) return precision;
  const shown = cuts.length > 3 ? `${cuts.slice(0, 3).join(", ")} and ${cuts.length - 3} more` : cuts.join(", ");
  return `${precision} · cut ${shown}`;
}

const NECK = ["model.12", "model.15", "model.16", "model.18", "model.19", "model.21"];

export const PRESETS: { id: string; name: string; hint: string; recipe: DetectorRecipe }[] = [
  { id: "native", name: "Native fp16", hint: "Ultralytics' weights, with BatchNorm folded into each convolution.", recipe: nativeRecipe() },
  { id: "int8-channel", name: "INT8 per channel", hint: "8-bit integers with one scale for every output channel.", recipe: { ...nativeRecipe(), bits: 8 } },
  { id: "int8-tensor", name: "INT8, one scale", hint: "8-bit integers, one scale shared by a whole tensor.", recipe: { ...nativeRecipe(), bits: 8, granularity: "tensor" } },
  { id: "int4-channel", name: "INT4 per channel", hint: "4-bit integers: 15 levels per output channel.", recipe: { ...nativeRecipe(), bits: 4 } },
  { id: "fp4-group", name: "FP4, groups of 32", hint: "4-bit floats on the E2M1 grid, a scale for every 32 weights.", recipe: { ...nativeRecipe(), bits: 4, granularity: "group", group: 32, levels: "fp" } },
  { id: "cut-l16", name: "Cut L16 by 30%", hint: "Masks the weakest 30% of the channels in L16, a downsampling conv in the neck.", recipe: { ...nativeRecipe(), masks: { "model.16": 0.3 } } },
  { id: "neck-20", name: "Trim the neck 20%", hint: "Masks the weakest 20% of the channels in every neck stage, L12 to L21.", recipe: { ...nativeRecipe(), masks: Object.fromEntries(NECK.map((s) => [s, 0.2])) } },
];

export const sameRecipe = (a: DetectorRecipe, b: DetectorRecipe) => keyOf(a) === keyOf(b);
