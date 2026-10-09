/**
 * The life list page's content, as HTML strings the page renders on the server
 * (the key, before anything is read) and again in the browser once the list is
 * read from storage. Everything shown comes from a sighting the visitor's own
 * runs saved; nothing here is ever made up.
 */

import { esc } from "../../lib/lab/markdown";
import { bestBySpecies, type Sighting } from "../../lib/ui/birds";
import { DODO, SPECIES, type Condition } from "../../lib/ui/mascot";
import { drawBird } from "./draw";
import { CONDITION_WORD, dateText, pct, ratioShort } from "./format";

export interface KeyEntry {
  id: string;
  name: string;
  latin: string;
  size: string; // the sizes this bird stands for
}

/** The five species in SPECIES order, then the dodo. */
export const KEY: KeyEntry[] = [
  ...SPECIES.map((sp, i) => {
    const next = SPECIES[i + 1];
    const size = sp.minRatio === 0 ? `Uncompressed, or less than ${next.minRatio} times smaller` : next ? `${sp.minRatio} to ${next.minRatio} times smaller` : `${sp.minRatio} times smaller or more`;
    return { id: sp.id, name: sp.name, latin: sp.latin, size };
  }),
  { id: "dodo", name: DODO.name, latin: DODO.latin, size: "A broken model less than 1.3 times smaller" },
];

const SPECIES_COUNT = SPECIES.length;

/** Only the site's own pages are linked from a sighting: storage is the visitor's, but a link must go somewhere real. */
export const safeHref = (href: unknown): string | null =>
  typeof href === "string" && /^\/(?!\/)[^\s"'<>]*$/.test(href) ? href : null;

const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : NaN);
const condWord = (c: unknown) => CONDITION_WORD[c as Condition] ?? "condition unknown";
const condClass = (c: unknown) => (c === "healthy" || c === "emaciated" || c === "skeleton" ? `c-${c}` : "");

export interface Summary {
  total: number;
  healthy: number; // species (of the five) seen healthy at least once
  best: Map<string, Sighting>;
  counts: Map<string, number>;
}

export function summarize(sightings: Sighting[]): Summary {
  const counts = new Map<string, number>();
  for (const s of sightings) counts.set(s.species, (counts.get(s.species) ?? 0) + 1);
  const healthy = SPECIES.filter((sp) => sightings.some((s) => s.species === sp.id && s.condition === "healthy")).length;
  return { total: sightings.length, healthy, best: bestBySpecies(sightings), counts };
}

/** "3 of 5 species seen healthy, from 42 sightings." */
export function countLine(s: Summary): string {
  const of = `${s.healthy} of ${SPECIES_COUNT} species seen healthy`;
  return s.total ? `${of}, from ${s.total} ${s.total === 1 ? "sighting" : "sightings"}.` : `${of}: no sightings yet.`;
}

/**
 * One species on the list. A bird is drawn only once it has been seen, in the
 * best condition seen. `summary` null means the list hasn't been read (no script
 * yet, or storage blocked): the entry is then only the key's words.
 */
export function entryHTML(k: KeyEntry, summary: Summary | null, prefix = "ll"): string {
  const best = summary?.best.get(k.id) ?? null;
  const count = summary?.counts.get(k.id) ?? 0;
  const ratio = best ? num(best.ratio) : NaN;
  // The drawing sits next to the bird's name and state, so screen readers skip it. A bird not
  // seen yet gets an empty plate of the same size, so the key's names line up.
  const draw = best && ratio > 0
    ? `<div class="sp-draw">${drawBird(ratio, num(best.starve) || 0, { decorative: true, prefix })}</div>`
    : `<div class="sp-draw sp-blank"><div class="sp-blank-in">${summary ? `<p class="sp-state">Not seen yet</p>` : ""}</div></div>`;
  let h = `<div class="sp-head">${draw}<div class="sp-id">
    <h3 class="sp-name">${esc(k.name)}</h3>
    <p class="sp-latin">${esc(k.latin)}</p>
    <p class="sp-size">${esc(k.size)}</p>
  </div></div>`;
  if (!summary || !best) return h;
  const state = k.id === "dodo" ? "Seen" : best.condition === "healthy" ? "Seen healthy" : `Best seen ${condWord(best.condition)}`;
  const kept = Number.isFinite(num(best.kept)) ? `keeps ${pct(num(best.kept))}` : "share kept unknown";
  h += `<div class="sp-detail">
    <p class="sp-state ${condClass(best.condition)}">${state}, ${count} ${count === 1 ? "sighting" : "sightings"}</p>
    <p class="sp-best">
      <span class="sp-model">Best sighting: ${esc(best.modelName)}</span>
      <span class="sp-recipe">${esc(best.recipe)}</span>
      <span class="sp-nums">${esc(ratioShort(num(best.ratio)))}, ${kept}, <time datetime="${esc(best.when)}">${esc(dateText(best.when))}</time></span>
    </p>`;
  const href = safeHref(best.href);
  if (href) h += `<p class="sp-again"><a href="${esc(href)}">Try this recipe again<span class="visually-hidden">: ${esc(k.name)} from ${esc(best.modelName)}</span></a></p>`;
  return h + `</div>`;
}

/**
 * Every sighting, newest first. The list is stored oldest first, and one run can
 * save several birds within the same millisecond, so ties keep the stored order reversed.
 */
export const newestFirst = (sightings: Sighting[]) =>
  [...sightings].reverse().sort((a, b) => String(b.when).localeCompare(String(a.when)));

export function rowHTML(s: Sighting): string {
  const href = safeHref(s.href);
  const model = href ? `<a href="${esc(href)}">${esc(s.modelName)}</a>` : esc(s.modelName);
  return `<tr>
    <td class="when"><time datetime="${esc(s.when)}">${esc(dateText(s.when, true))}</time></td>
    <td><span class="row-bird">${esc(s.name)}</span>, <span class="${condClass(s.condition)}">${esc(condWord(s.condition))}</span></td>
    <td class="row-model">${model}<span class="row-recipe">${esc(s.recipe)}</span></td>
    <td class="num">${esc(ratioShort(num(s.ratio)))}</td>
    <td class="num">${Number.isFinite(num(s.kept)) ? pct(num(s.kept)) : ""}</td>
  </tr>`;
}
