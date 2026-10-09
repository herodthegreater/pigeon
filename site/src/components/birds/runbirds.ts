/**
 * The birds a Field Guide run hatches, shown under its result table the way a
 * field guide shows specimens: the run's best bird drawn large and named, every
 * other model as one small drawing on a ruled line, and the rule once.
 * Scored by labBirds() in lib/ui/birds.ts; each bird is also a life-list sighting.
 */

import { esc } from "../../lib/lab/markdown";
import { KEEP_RULE, labBirds, type LabBird, type LabRun, type Sighting } from "../../lib/ui/birds";
import { DODO, SPECIES } from "../../lib/ui/mascot";
import { drawBird } from "./draw";
import { andList, CONDITION_WORD, keptPhrase, pct, ratioPhrase, ratioShort } from "./format";

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export interface RunInfo {
  methodId: string;
  methodName: string;
  href: string; // the method page's run box
  settings: string; // the run's settings in words
  lifeListHref: string;
}

/**
 * What happened when the run's birds were saved: null when there was nothing to
 * save; `saved` counts the birds stored before storage refused one.
 */
export type Saved = { ok: true; firstHealthy: string[] } | { ok: false; saved: number } | null;

const latin = (b: LabBird) => (b.species === "dodo" ? DODO.latin : SPECIES.find((s) => s.id === b.species)?.latin ?? "");

/**
 * The method page with a run's settings in its address ("/pigeon/guide/ptq/?bits=3#run"),
 * which the run box loads, so a life list's "try this recipe again" means this recipe.
 */
export function runHref(page: string, params: Record<string, unknown>): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (typeof v === "number" || typeof v === "boolean" || typeof v === "string") q.set(k, String(v));
  }
  const qs = q.toString();
  return `${page}${qs ? `?${qs}` : ""}#run`;
}

/** The recipe a sighting records: the row, and the settings of the run that produced it. */
export const recipeOf = (label: string, settings: string) => `${label} (run with ${settings || "default settings"})`;

export function runBirds(run: LabRun, info: RunInfo) {
  const scored = labBirds(run);
  const sightings: Omit<Sighting, "when">[] = scored.bird
    ? scored.birds.map((b) => ({
        ratio: b.ratio, kept: b.kept, condition: b.condition, starve: b.starve, species: b.species, name: b.name,
        model: `lab:${info.methodId}`, modelName: info.methodName, recipe: recipeOf(b.label, info.settings), href: info.href,
      }))
    : [];

  function render(saved: Saved): string {
    const none = !scored.bird || !scored.best;
    const head = `<h3 id="run-birds-title">${none ? "No birds from this run" : "Birds from this run"}</h3>`;
    if (!scored.bird) {
      return `<section class="run-birds" aria-labelledby="run-birds-title">${head}<p class="no-bird">${esc(scored.reason)}</p></section>`;
    }
    const { birds, best } = scored;
    // The lab often labels the row "teacher 3x256 (reference)"; a sentence already says it's the reference.
    const reference = scored.reference.replace(/\s*\(reference\)\s*$/i, "") || scored.reference;
    if (!best) {
      return `<section class="run-birds" aria-labelledby="run-birds-title">${head}<p class="no-bird">This run has no smaller model to score against ${esc(reference)}, so there's no bird.</p></section>`;
    }
    const others = birds.filter((b) => b !== best);
    const metric = run.metric;
    let h = `<section class="run-birds" aria-labelledby="run-birds-title">${head}`;
    h += `<figure class="best-bird">
      <div class="best-plate">${drawBird(best.ratio, best.starve, { decorative: true, prefix: "run" })}</div>
      <figcaption>
        <p class="bird-title"><span class="bird-species">${esc(best.name)}</span> <span class="bird-cond c-${best.condition}">${CONDITION_WORD[best.condition]}</span></p>
        ${latin(best) ? `<p class="bird-latin">${esc(latin(best))}</p>` : ""}
        <p class="bird-row">${esc(best.label)}</p>
        <p class="bird-nums">${esc(cap(ratioPhrase(best.ratio, reference)))}, ${esc(keptPhrase(best.kept, metric))}.</p>
      </figcaption>
    </figure>`;
    if (others.length) {
      h += `<ul class="bird-rows" aria-label="The run's other models">${others
        .map(
          (b) => `<li>
        <span class="mini">${drawBird(b.ratio, b.starve, { decorative: true, prefix: "run" })}</span>
        <span class="mini-text">
          <span class="mini-label">${esc(b.label)}</span>
          <span class="mini-bird">${esc(b.name)}, <span class="c-${b.condition}">${CONDITION_WORD[b.condition]}</span></span>
          <span class="mini-nums num">${esc(ratioShort(b.ratio))}, keeps ${pct(b.kept)}</span>
        </span>
      </li>`,
        )
        .join("")}</ul>`;
    }
    h += `<p class="bird-rule">Measured against ${esc(reference)}, this run's reference. Species follows how many times smaller a model is. ${esc(KEEP_RULE)}</p>`;
    if (saved && saved.ok) {
      const which = sightings.length === 1 ? "This bird is" : sightings.length === 2 ? "Both birds are" : `All ${sightings.length} birds are`;
      h += `<p class="bird-saved">${which} saved to your <a href="${esc(info.lifeListHref)}">life list</a> in this browser.</p>`;
    } else if (saved && !saved.ok) {
      h += saved.saved > 0
        ? `<p class="bird-saved">Only ${saved.saved} of ${sightings.length} birds were saved to your <a href="${esc(info.lifeListHref)}">life list</a>: this browser's storage refused the rest.</p>`
        : `<p class="bird-saved">This browser won't let the site store anything (private windows often block it), so this run can't be saved to your life list.</p>`;
    }
    return h + `</section>`;
  }

  return { scored, sightings, render };
}

/** The status line's addition when a run hatches a species' first healthy bird. */
export function firstHealthyLine(names: string[]): string {
  if (!names.length) return "";
  return `First healthy ${andList([...new Set(names)])} on your life list.`;
}
