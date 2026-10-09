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

/** What happened when the run's birds were saved: null when there was nothing to save. */
export type Saved = { ok: true; firstHealthy: string[] } | { ok: false } | null;

const latin = (b: LabBird) => (b.species === "dodo" ? DODO.latin : SPECIES.find((s) => s.id === b.species)?.latin ?? "");

/** The recipe a sighting records: the row, and the settings that produced it. */
export const recipeOf = (label: string, settings: string) => `${label} (${settings || "default settings"})`;

export function runBirds(run: LabRun, info: RunInfo) {
  const scored = labBirds(run);
  const sightings: Omit<Sighting, "when">[] = scored.bird
    ? scored.birds.map((b) => ({
        ratio: b.ratio, kept: b.kept, condition: b.condition, starve: b.starve, species: b.species, name: b.name,
        model: `lab:${info.methodId}`, modelName: info.methodName, recipe: recipeOf(b.label, info.settings), href: info.href,
      }))
    : [];

  function render(saved: Saved): string {
    const head = `<h3 id="run-birds-title">Birds from this run</h3>`;
    if (!scored.bird) {
      return `<section class="run-birds" aria-labelledby="run-birds-title">${head}<p class="no-bird">${esc(scored.reason)}</p></section>`;
    }
    const { reference, birds, best } = scored;
    if (!best) {
      return `<section class="run-birds" aria-labelledby="run-birds-title">${head}<p class="no-bird">This run has no smaller model to score against ${esc(reference)}, so there's no bird.</p></section>`;
    }
    const others = birds.filter((b) => b !== best);
    const metric = run.metric;
    let h = `<section class="run-birds" aria-labelledby="run-birds-title">${head}`;
    h += `<figure class="best-bird">
      <div class="best-plate">${drawBird(best.ratio, best.starve, { prefix: "run" })}</div>
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
      h += `<p class="bird-saved">${sightings.length === 1 ? "This bird is" : `All ${sightings.length} birds are`} saved to your <a href="${esc(info.lifeListHref)}">life list</a> in this browser.</p>`;
    } else if (saved && !saved.ok) {
      h += `<p class="bird-saved">This browser blocks site storage, so this run can't be saved to your life list here.</p>`;
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
