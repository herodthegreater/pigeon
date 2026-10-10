/**
 * The life list page's content, as HTML strings the page renders on the server
 * (the key, before anything is read) and again in the browser once the list is
 * read from storage. Everything shown comes from a sighting the visitor's own
 * runs saved; nothing here is ever made up.
 */

import { esc } from "../../lib/lab/markdown";
import { bestBySpecies, KEEP_RULE, PIGEON_RULE, type BirdRule, type Sighting } from "../../lib/ui/birds";
import { BROKEN, DODO, SPECIES, WORKS, type Condition } from "../../lib/ui/mascot";
import { drawBird } from "./draw";
import { CONDITION_WORD, dateText, pct, ratioShort } from "./format";

/**
 * The two rules a bird's condition can come from, by the names the page uses for them.
 * Every sighting says which one graded it, and its numbers are the ones that rule reads,
 * so no row can be checked against the wrong rule.
 */
export const RULES: Record<BirdRule, { name: string; inSentence: string; who: string; text: string }> = {
  keep: { name: "Share kept", inSentence: "the share-kept rule", who: "YOLOv8n in the Coop and every Field Guide run", text: KEEP_RULE },
  pigeon: { name: "Pigeon's line", inSentence: "Pigeon's line", who: "Pigeon in the Coop", text: PIGEON_RULE },
};

/**
 * The rule that graded a sighting. A Pigeon sighting saved before sightings carried their
 * rule was graded by Pigeon's line all the same: the Coop has always drawn Pigeon by it.
 */
export const ruleOf = (s: Pick<Sighting, "rule" | "model">): BirdRule =>
  s.rule === "pigeon" || s.rule === "keep" ? s.rule : s.model === "pigeon" ? "pigeon" : "keep";

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

const whole = (v: number) => `${Math.round(v * 100)}%`;

/**
 * Pigeon's accuracy, as Pigeon's line read it, or null when what was saved can't say. The
 * Coop saves the share of accuracy kept, not the accuracy, but the line pins the accuracy
 * down from what it did save (pointing, condition, starvation; mascot.ts starvation()):
 * a healthy bird has at least 90%; a skeleton whose pointing is above 35% was broken by
 * accuracy, so it has at most 60%; a starving bird whose pointing alone would starve it
 * less was starved by accuracy, which its starvation then gives exactly.
 */
export function pigeonAccuracy(s: Sighting): string | null {
  const saved = num((s as Sighting & { accuracy?: unknown }).accuracy);
  if (Number.isFinite(saved)) return `accuracy ${pct(saved)}`;
  const pointing = num(s.pointing);
  if (!Number.isFinite(pointing)) return null;
  if (s.condition === "healthy") return `accuracy ${whole(WORKS.accuracy)} or more`;
  if (s.condition === "skeleton") return pointing > BROKEN.pointing ? `accuracy ${whole(BROKEN.accuracy)} or less` : null;
  const byPointing = (WORKS.pointing - pointing) / (WORKS.pointing - BROKEN.pointing);
  const starve = num(s.starve);
  if (s.condition !== "emaciated" || !(starve > byPointing + 1e-9) || !(starve < 1)) return null;
  return `accuracy ${pct(WORKS.accuracy - starve * (WORKS.accuracy - BROKEN.accuracy))}`;
}

/**
 * The numbers the sighting's rule read, in words. Share kept: "keeps 98.2%". Pigeon's line
 * reads accuracy and pointing: "accuracy 89.6%, pointing 80.0%". Where the accuracy can't be
 * known, the share of it kept is said as a share, so it can't be taken for the accuracy the
 * line is drawn on.
 */
export function scoreParts(s: Sighting): string[] {
  const kept = num(s.kept);
  const keeps = Number.isFinite(kept) ? `keeps ${pct(kept)}` : "share kept unknown";
  if (ruleOf(s) === "keep") return [keeps];
  const pointing = num(s.pointing);
  return [
    pigeonAccuracy(s) ?? (Number.isFinite(kept) ? `${keeps} of its accuracy` : "accuracy unknown"),
    Number.isFinite(pointing) ? `pointing ${pct(pointing)}` : "pointing not saved",
  ];
}
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
  h += `<div class="sp-detail">
    <p class="sp-state ${condClass(best.condition)}">${state}, ${count} ${count === 1 ? "sighting" : "sightings"}</p>
    <p class="sp-best">
      <span class="sp-model">Best sighting: ${esc(best.modelName)}</span>
      <span class="sp-recipe">${esc(best.recipe)}</span>
      <span class="sp-nums">${esc([ratioShort(num(best.ratio)), ...scoreParts(best)].join(", "))}, <time datetime="${esc(best.when)}">${esc(dateText(best.when))}</time></span>
      <span class="sp-rule">Graded by ${esc(RULES[ruleOf(best)].inSentence)}</span>
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
    <td class="num score">${scoreParts(s).map((p) => `<span>${esc(p)}</span>`).join("")}</td>
    <td class="rule">${esc(RULES[ruleOf(s)].name)}</td>
  </tr>`;
}
