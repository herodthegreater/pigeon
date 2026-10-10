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

/**
 * A recipe's identity, normalized so equal recipes share one key (and one cache entry) whatever
 * order their cuts came in. 16 bits ignores the scale and the grid, and 1 bit has no float grid.
 */
export function keyOf(r: DetectorRecipe): string {
  const masks = Object.entries(r.masks).filter(([, v]) => v > 0).sort(([a], [b]) => a.localeCompare(b));
  return r.bits >= 16 ? JSON.stringify([16, masks]) : JSON.stringify([r.bits, granularityOf(r), r.bits === 1 ? "int" : r.levels, masks]);
}

/** "L4" for "model.4": Ultralytics numbers the layers of the model in order. */
export const layerName = (stage: string) => stage.replace("model.", "L");

/** The cuts in layer order, as [stage, fraction] pairs. */
const cutsOf = (r: DetectorRecipe) =>
  Object.entries(r.masks)
    .filter(([, v]) => v > 0)
    .sort(([a], [b]) => Number(a.split(".")[1]) - Number(b.split(".")[1]));

/** The label for the runs table and the life list, naming every setting: "INT4 per channel · cut L4 20%, L6 10%". */
export function describe(r: DetectorRecipe): string {
  const g = r.granularity === "group" ? `groups of ${r.group}` : r.granularity === "tensor" ? "one scale per tensor" : "per channel";
  const precision = r.bits >= 16 ? "Native fp16" : r.bits === 1 ? `1-bit, ${g}` : `${r.levels === "fp" ? "FP" : "INT"}${r.bits} ${g}`;
  const cuts = cutsOf(r).map(([s, v]) => `${layerName(s)} ${Math.round(v * 100)}%`);
  return cuts.length ? `${precision} · cut ${cuts.join(", ")}` : precision;
}

const NECK = ["model.12", "model.15", "model.16", "model.18", "model.19", "model.21"];

export const PRESETS: { id: string; name: string; hint: string; recipe: DetectorRecipe }[] = [
  { id: "native", name: "Native fp16", hint: "Ultralytics' weights, with BatchNorm folded into each convolution.", recipe: nativeRecipe() },
  { id: "int8-channel", name: "INT8 per channel", hint: "8-bit integers with one scale for every output channel.", recipe: { ...nativeRecipe(), bits: 8 } },
  { id: "int8-tensor", name: "INT8, one scale", hint: "8-bit integers, one scale shared by a whole tensor.", recipe: { ...nativeRecipe(), bits: 8, granularity: "tensor" } },
  { id: "int4-channel", name: "INT4 per channel", hint: "4-bit integers: 15 levels per output channel.", recipe: { ...nativeRecipe(), bits: 4 } },
  { id: "fp4-group", name: "FP4, groups of 32", hint: "4-bit floats on the E2M1 grid, a scale for each block of up to 32 weights in an output channel.", recipe: { ...nativeRecipe(), bits: 4, granularity: "group", group: 32, levels: "fp" } },
  { id: "cut-l16", name: "Cut L16 by 30%", hint: "Masks the weakest 30% of the channels in L16, a downsampling conv in the neck.", recipe: { ...nativeRecipe(), masks: { "model.16": 0.3 } } },
  { id: "neck-20", name: "Trim the neck 20%", hint: "Masks the weakest 20% of the channels in every neck stage, L12 to L21.", recipe: { ...nativeRecipe(), masks: Object.fromEntries(NECK.map((s) => [s, 0.2])) } },
];

export const sameRecipe = (a: DetectorRecipe, b: DetectorRecipe) => keyOf(a) === keyOf(b);

/** The preset a recipe is, if any ("native" included). */
export const presetOf = (r: DetectorRecipe) => PRESETS.find((p) => sameRecipe(p.recipe, r))?.id ?? null;

/* ------------------------------------------------------------ the recipe in a link */

const BITS = [1, 2, 3, 4, 5, 6, 7, 8, 16];
const LINK_KEYS = ["bits", "scale", "group", "grid", "cut"];

/**
 * The query that reopens a recipe in the cage: "" for native, "preset=<id>" for a preset, and
 * otherwise every setting, e.g. "bits=4&scale=group&group=32&grid=fp&cut=L4:20,L16:30".
 */
export function linkQuery(r: DetectorRecipe): string {
  const preset = presetOf(r);
  if (preset) return preset === "native" ? "" : `preset=${preset}`;
  const q: string[] = [];
  if (r.bits < 16) {
    q.push(`bits=${r.bits}`, `scale=${r.granularity}`);
    if (r.granularity === "group") q.push(`group=${r.group}`);
    if (r.bits > 1) q.push(`grid=${r.levels}`);
  }
  const cuts = cutsOf(r).map(([s, v]) => `${layerName(s)}:${Math.round(v * 100)}`);
  if (cuts.length) q.push(`cut=${cuts.join(",")}`);
  return q.join("&");
}

export type FromLink = { recipe: DetectorRecipe } | { error: string } | null;

const nearest = (v: number, options: readonly number[]) => options.reduce((a, b) => (Math.abs(b - v) < Math.abs(a - v) ? b : a));
const number = (s: string) => (s.trim() ? Number(s) : NaN);

/**
 * The recipe a link carries: linkQuery's parameters, or ?preset=<id>. Every number is clamped to
 * what the controls can set. `stages` are the stages that can be cut ("model.4"). Returns null when
 * the link carries no recipe, and the reason in words when it can't be read.
 */
export function recipeFromLink(params: URLSearchParams, stages: string[]): FromLink {
  if (!LINK_KEYS.some((k) => params.has(k))) {
    const id = params.get("preset");
    if (id === null) return null;
    const p = PRESETS.find((x) => x.id === id);
    return p ? { recipe: JSON.parse(JSON.stringify(p.recipe)) } : { error: `there is no preset called "${id}"` };
  }
  const r = nativeRecipe();
  const bits = params.get("bits");
  if (bits !== null) {
    const n = number(bits);
    if (!Number.isFinite(n)) return { error: `"${bits}" is not a number of bits` };
    r.bits = n >= 16 ? 16 : nearest(Math.min(8, Math.max(1, n)), BITS);
  }
  const scale = params.get("scale");
  if (scale !== null) {
    if (scale !== "tensor" && scale !== "channel" && scale !== "group") return { error: `"${scale}" is not a way to share scales` };
    r.granularity = scale;
  }
  const group = params.get("group");
  if (group !== null) {
    const n = number(group);
    if (!Number.isFinite(n)) return { error: `"${group}" is not a group size` };
    r.group = nearest(n, GROUP_SIZES);
  }
  const grid = params.get("grid");
  if (grid !== null) {
    if (grid !== "int" && grid !== "fp") return { error: `"${grid}" is not a grid` };
    r.levels = grid;
  }
  for (const c of (params.get("cut") ?? "").split(",").filter((c) => c.trim())) {
    const m = /^L(\d+):(\d+(?:\.\d+)?)$/.exec(c.trim());
    if (!m) return { error: `"${c}" is not a cut` };
    const stage = `model.${Number(m[1])}`;
    if (!stages.includes(stage)) return { error: `L${Number(m[1])} has no channels to cut` };
    const v = Math.min(MAX_MASK * 100, Math.max(0, Math.round(Number(m[2]) / 10) * 10)) / 100;
    if (v > 0) r.masks[stage] = v;
    else delete r.masks[stage];
  }
  return { recipe: r };
}
