/**
 * The Coop's side of the life list (birds.ts). A final scoring run on every test
 * photo becomes a sighting; a quick estimate never does. Only a model that really
 * got smaller hatches a bird, the same rule as a lab run's rows: the reference
 * (Pigeon at full precision, native YOLOv8n) is not a sighting, and neither is a
 * recipe that leaves the model as big as it was (a 0% prune, steps on no layers).
 */

import { bestBySpecies, LIFE_LIST_EVENT, readLifeList, recordSighting, type Bird, type BirdRule } from "./birds";
import { BROKEN, SPECIES } from "./mascot";

export interface CoopSighting {
  bird: Bird;
  model: "pigeon" | "yolov8n";
  modelName: string;
  recipe: string; // the run's label, as the cage's own table shows it
  recipeKey?: string; // the recipe's identity, when the cage has one apart from its label
  href: string; // the cage, with the recipe in its query (cageHref)
  rule?: BirdRule; // "pigeon" for Pigeon's own accuracy-and-pointing line
  pointing?: number; // Pigeon only
}

/** The cage's URL for a recipe, given the recipe's query (linkQuery in recipe.ts or detector/recipe.ts; "" for the reference). */
export const cageHref = (cage: string, query: string) => (query ? `${cage}?${query}` : cage);

/** Whether a run hatches a bird: its model is smaller than the reference (ratio = reference bytes / its bytes). */
export const hatches = (ratio: number) => ratio > 1;

/** Which size makes which bird, for a cage's rule line: "dove from 1.8×, swift from 3.5×, …". */
export const SPECIES_LADDER = SPECIES.filter((s) => s.minRatio > 0)
  .map((s) => `${s.name.split(" ").pop()!.toLowerCase()} from ${s.minRatio}×`)
  .join(", ");

/** Pigeon's skeleton line in words, beside WORKS_LINE (mascot.ts). */
export const BROKEN_LINE = `≤ ${Math.round(BROKEN.accuracy * 100)}% accuracy or ≤ ${Math.round(BROKEN.pointing * 100)}% pointing`;

export const STORAGE_BLOCKED = "This browser won't let the page save anything, so this bird can't go on your life list.";

/**
 * Save a sighting. `said` is the sentence the cage adds to its status line ("" when there's
 * nothing new); `announce` is false for news a screen reader has already heard this visit.
 */
export function sight(s: CoopSighting, warned: { blocked: boolean }): { said: string; announce: boolean } {
  const { bird, ...about } = s;
  const { ok, firstHealthy } = recordSighting({ ...bird, ...about });
  if (!ok) {
    const first = !warned.blocked;
    warned.blocked = true;
    return { said: STORAGE_BLOCKED, announce: first };
  }
  return firstHealthy ? { said: `First healthy ${s.bird.name} on your life list.`, announce: true } : { said: "", announce: false };
}

/** "3 species so far", kept current as sightings land (in this tab or another one). */
export function lifeListCount(): string {
  const { ok, sightings } = readLifeList();
  if (!ok) return "can't be kept in this browser";
  const n = bestBySpecies(sightings).size;
  return n ? `${n} species so far` : "no birds yet";
}

export function bindLifeListCount(el: HTMLElement) {
  const draw = () => { el.textContent = `· ${lifeListCount()}`; };
  draw();
  addEventListener(LIFE_LIST_EVENT, draw);
  addEventListener("storage", (e) => { if (e.key === null || e.key === "pigeon.lifelist") draw(); });
}

type BirdBox = Pick<HTMLElement, "innerHTML" | "classList">;

/**
 * Draws the cage's bird. A new species fades out for `wait` ms first (the `swap` class); any
 * drawing asked for meanwhile replaces the pending one, so the bird shown last is always the
 * latest result, never one an earlier timer captured.
 */
export function birdDrawer(box: BirdBox, wait = 180) {
  let shown = "", latest = "";
  let timer: ReturnType<typeof setTimeout> | undefined;
  return (species: string, svg: string) => {
    latest = svg;
    if (timer === undefined) {
      if (species !== shown && shown) {
        box.classList.add("swap");
        timer = setTimeout(() => {
          timer = undefined;
          box.innerHTML = latest;
          box.classList.remove("swap");
        }, wait);
      } else box.innerHTML = svg;
    }
    shown = species;
  };
}
