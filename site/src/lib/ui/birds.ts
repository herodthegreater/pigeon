/**
 * The game layer: every model that gets smaller ends in a bird (DESIGN.md: Motif).
 *   species   <- size ratio: the reference's storage / this model's storage (SPECIES in mascot.ts)
 *   condition <- quality kept: this model's score / the reference's score
 * One rule for every model: healthy while it keeps at least 95% of its reference's
 * score, bones at 60% or less, starving in between. Pigeon in the Coop keeps its
 * own documented "still works" line (accuracy and pointing, mascot.ts WORKS).
 *
 * Every bird someone hatches is saved in this browser as a life list, the way a
 * birder keeps one: each species counts once, at its best sighting.
 */

import { BROKEN, WORKS, conditionOf, speciesFor, type Condition } from "./mascot";

export const HEALTHY_KEEP = 0.95;
export const SKELETON_KEEP = 0.6;

/** The rule in words, for captions. */
const pct = (x: number) => Math.round(x * 100);
export const KEEP_RULE = `A bird stays healthy while its model keeps at least ${pct(HEALTHY_KEEP)}% of the reference's score, and turns to bones at ${pct(SKELETON_KEEP)}% or less.`;

/** Pigeon in the Coop is graded by its own documented line (mascot.ts WORKS and BROKEN). */
export const PIGEON_RULE = `Pigeon in the Coop has its own line: healthy at ${pct(WORKS.accuracy)}% accuracy or more with ${pct(WORKS.pointing)}% pointing or more, bones at ${pct(BROKEN.accuracy)}% accuracy or less, or ${pct(BROKEN.pointing)}% pointing or less.`;

/** Which rule graded a sighting. */
export type BirdRule = "keep" | "pigeon";

/** Condition and starvation (0 healthy .. 1 bones) from the fraction of the reference's score kept. */
export function keptCondition(kept: number): { condition: Condition; starve: number } {
  if (!(kept > SKELETON_KEEP)) return { condition: "skeleton", starve: 1 };
  if (kept >= HEALTHY_KEEP) return { condition: "healthy", starve: 0 };
  const s = (HEALTHY_KEEP - kept) / (HEALTHY_KEEP - SKELETON_KEEP);
  const starve = Math.min(0.999, Math.max(0.001, s));
  return { condition: conditionOf(starve), starve };
}

/** A broken model below this ratio is drawn as a dodo (mascot.ts does the same). */
const DODO_BELOW = 1.3;

export interface Bird {
  ratio: number; // reference storage / this storage (1 = same size)
  kept: number; // score / reference score (can exceed 1)
  condition: Condition;
  starve: number;
  species: string; // species id, or "dodo"
  name: string; // "Barn Swallow", or "Dodo"
}

export function birdFor(ratio: number, kept: number): Bird {
  const { condition, starve } = keptCondition(kept);
  const dodo = condition === "skeleton" && ratio < DODO_BELOW;
  const sp = speciesFor(ratio);
  return { ratio, kept, condition, starve, species: dodo ? "dodo" : sp.id, name: dodo ? "Dodo" : sp.name };
}

/* ------------------------------------------------------------ lab runs */

type Row = Record<string, unknown>;
export interface LabRun {
  rows: Row[];
  metric?: string;
  bird?: boolean;
  bird_reason?: string;
}
export interface LabBird extends Bird {
  label: string; // the row's label, e.g. "4-bit, group 16"
  kbits: number;
}

/**
 * Birds for one lab method run. The lab marks the uncompressed reference row with
 * `ref`, gives every model row its true storage in `kbits`, and says `bird: false`
 * (with `bird_reason`) when the method doesn't make anything smaller.
 */
export function labBirds(run: LabRun):
  | { bird: false; reason: string }
  | { bird: true; reference: string; birds: LabBird[]; best: LabBird | null } {
  if (run.bird === false) return { bird: false, reason: run.bird_reason || "This method doesn't make the model smaller, so there's no bird." };
  const metric = run.metric || "acc";
  const ref = run.rows.find((r) => r.ref === true);
  const score = (r: Row) => (typeof r[metric] === "number" ? (r[metric] as number) : NaN);
  if (!ref || !(Number(ref.kbits) > 0) || !(score(ref) > 0)) {
    return { bird: false, reason: "This run has no reference model to measure against, so there's no bird." };
  }
  // Only models that really got smaller hatch a bird (DESIGN.md: "every place a model gets
  // smaller"), and a row that repeats an earlier model (same storage, same score) isn't a new bird.
  const seen = new Set<string>();
  const birds: LabBird[] = run.rows
    .filter((r) => r !== ref && r.model !== false && Number(r.kbits) > 0 && Number.isFinite(score(r)))
    .filter((r) => Number(ref.kbits) / Number(r.kbits) > 1)
    .filter((r) => {
      const key = `${r.kbits}|${score(r)}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .map((r) => ({ ...birdFor(Number(ref.kbits) / Number(r.kbits), score(r) / score(ref)), label: String(r.label ?? ""), kbits: Number(r.kbits) }));
  if (!birds.length) {
    return { bird: false, reason: `No model in this run is smaller than ${String(ref.label ?? "the reference")}, so there's no bird.` };
  }
  return { bird: true, reference: String(ref.label ?? "the reference"), birds, best: bestBird(birds) };
}

/** The bird a run is proudest of: the smallest healthy one, else the healthiest of the rest. */
export function bestBird<T extends Bird>(birds: T[]): T | null {
  if (!birds.length) return null;
  const healthy = birds.filter((b) => b.condition === "healthy");
  if (healthy.length) return healthy.reduce((a, b) => (b.ratio > a.ratio ? b : a));
  return birds.reduce((a, b) => (b.starve < a.starve || (b.starve === a.starve && b.ratio > a.ratio) ? b : a));
}

/* ------------------------------------------------------------ the life list */

export interface Sighting extends Bird {
  model: string; // "pigeon", "yolov8n", or "lab:<method id>"
  modelName: string; // what a reader calls it: "Pigeon", "YOLOv8n", "Post-training quantization"
  recipe: string; // what was done, in the site's words
  recipeKey?: string; // a recipe's full identity when the label is shortened for reading
  href: string; // where to try it again
  rule?: BirdRule; // which rule graded the condition ("keep" when absent)
  pointing?: number; // Pigeon only: its pointing score, part of its rule
  when: string; // ISO timestamp
}

const KEY = "pigeon.lifelist";
const MAX = 400;
/** Fired on window whenever the life list changes (detail: the new list). */
export const LIFE_LIST_EVENT = "pigeon:lifelist";

export function readLifeList(): { ok: boolean; sightings: Sighting[] } {
  try {
    const raw = localStorage.getItem(KEY);
    const list = raw ? JSON.parse(raw) : [];
    return { ok: true, sightings: Array.isArray(list) ? list.filter((s) => s && typeof s.species === "string") : [] };
  } catch {
    return { ok: false, sightings: [] };
  }
}

/**
 * Save a sighting. The same model and recipe again replaces its earlier sighting.
 * `firstHealthy` is true when this is the first healthy bird of its species in the list.
 */
export function recordSighting(s: Omit<Sighting, "when">, when = new Date().toISOString()): { ok: boolean; firstHealthy: boolean } {
  const { ok, sightings } = readLifeList();
  if (!ok) return { ok: false, firstHealthy: false };
  const firstHealthy = s.condition === "healthy" && !sightings.some((x) => x.species === s.species && x.condition === "healthy");
  const same = (x: Omit<Sighting, "when">) => x.model === s.model && (x.recipeKey ?? x.recipe) === (s.recipeKey ?? s.recipe);
  const next = trim([...sightings.filter((x) => !same(x)), { ...s, when }]);
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    return { ok: false, firstHealthy: false };
  }
  dispatchEvent(new CustomEvent(LIFE_LIST_EVENT, { detail: next }));
  return { ok: true, firstHealthy };
}

/** Keep the list under MAX by dropping the oldest sightings, but never a species' best one. */
function trim(list: Sighting[]): Sighting[] {
  if (list.length <= MAX) return list;
  const keep = new Set(bestBySpecies(list).values());
  const out = [...list];
  for (let i = 0; i < out.length && out.length > MAX; ) {
    if (keep.has(out[i])) i++;
    else out.splice(i, 1);
  }
  return out;
}

export function clearLifeList(): boolean {
  try {
    localStorage.removeItem(KEY);
  } catch {
    return false;
  }
  dispatchEvent(new CustomEvent(LIFE_LIST_EVENT, { detail: [] }));
  return true;
}

/** Each species' best sighting (healthy first, then the smallest), in sightings' species order. */
export function bestBySpecies(sightings: Sighting[]): Map<string, Sighting> {
  const best = new Map<string, Sighting>();
  for (const s of sightings) {
    const b = best.get(s.species);
    if (!b || bestBird([b, s]) === s) best.set(s.species, s);
  }
  return best;
}
