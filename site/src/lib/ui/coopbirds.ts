/**
 * The Coop's side of the life list (birds.ts). A final scoring run on every test
 * photo becomes a sighting; a quick estimate never does. The reference model
 * (Pigeon at full precision, native YOLOv8n) is not a sighting either, the same
 * as a lab run's reference row: nothing was compressed.
 */

import { bestBySpecies, LIFE_LIST_EVENT, readLifeList, recordSighting, type Bird } from "./birds";

export interface CoopSighting {
  bird: Bird;
  model: "pigeon" | "yolov8n";
  modelName: string;
  recipe: string; // the run's label, as the cage's own table shows it
  href: string; // the cage, with ?preset= when the recipe is a preset
}

/** The cage's URL for a recipe: a preset can be reopened by its id, any other recipe only by the cage itself. */
export const cageHref = (cage: string, preset: string | null) => (preset ? `${cage}?preset=${encodeURIComponent(preset)}` : cage);

export const STORAGE_BLOCKED = "This browser won't let the page save anything, so this bird can't go on your life list.";

/**
 * Save a sighting. `said` is the sentence the cage adds to its status line ("" when there's
 * nothing new); `announce` is false for news a screen reader has already heard this visit.
 */
export function sight(s: CoopSighting, warned: { blocked: boolean }): { said: string; announce: boolean } {
  const { ok, firstHealthy } = recordSighting({ ...s.bird, model: s.model, modelName: s.modelName, recipe: s.recipe, href: s.href });
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
