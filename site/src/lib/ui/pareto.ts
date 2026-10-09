/**
 * Size vs accuracy: every setting the visitor has tried, the best-tradeoff
 * frontier, and the healthy line. Plain SVG, drawn to one scale.
 *
 * The chart is drawn at the width it is shown at (one SVG unit = one CSS px),
 * so its labels render at their real type-scale sizes on a phone and on a desktop.
 *
 * Every run is named with the bird's three words (CONDITION_WORD in mascot.ts),
 * and each has its own mark, so color is never the only cue: a filled dot is
 * healthy, a hollow dot is starving, a cross is a skeleton.
 */

import { CONDITION_WORD, conditionOf, starvation, WORKS, type Condition } from "./mascot";

export interface Attempt {
  label: string;
  bytes: number;
  accuracy: number;
  pointing: number;
}

const M = { l: 52, r: 28, t: 16, b: 46 }; // r leaves room for half of the last tick label ("5.0 MB")
const X_MIN = 20e3, X_MAX = 5e6, Y_MIN = 0.45, Y_MAX = 1.0;
const X_TICKS = [20e3, 50e3, 100e3, 250e3, 500e3, 1e6, 2e6, 5e6];
const Y_TICKS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
const MIN_TICK_GAP = 56; // px between x-axis labels, so "250 KB" never collides with its neighbour

/** The bird's condition for a run, by the Coop's still-works line (accuracy and pointing). */
export const conditionOfAttempt = (a: Attempt): Condition => conditionOf(starvation(a.accuracy, a.pointing));
export const works = (a: Attempt) => conditionOfAttempt(a) === "healthy";

/** Non-dominated attempts: nothing else is both smaller and more accurate. */
export function frontier(all: Attempt[]): Attempt[] {
  const sorted = [...all].sort((a, b) => a.bytes - b.bytes || b.accuracy - a.accuracy);
  const out: Attempt[] = [];
  let best = -1;
  for (const a of sorted) if (a.accuracy > best) { out.push(a); best = a.accuracy; }
  return out;
}

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
const kb = (b: number) => (b >= 1e6 ? `${(b / 1e6).toFixed(b >= 1e7 ? 0 : 1)} MB` : `${Math.round(b / 1e3)} KB`);

/** One run's mark: filled teal dot (healthy), hollow amber dot (starving), red cross (skeleton). */
function mark(c: Condition, cx: number, cy: number, title: string): string {
  const t = `<title>${title}</title>`;
  if (c === "healthy") return `<circle cx="${cx}" cy="${cy}" r="4.5" fill="var(--teal)" stroke="var(--teal)" stroke-width="1.6">${t}</circle>`;
  if (c === "emaciated") return `<circle cx="${cx}" cy="${cy}" r="4.5" fill="var(--card)" stroke="var(--warn)" stroke-width="1.6">${t}</circle>`;
  const d = 4;
  return `<path d="M ${cx - d} ${cy - d} L ${cx + d} ${cy + d} M ${cx - d} ${cy + d} L ${cx + d} ${cy - d}" stroke="var(--bad)" stroke-width="2" stroke-linecap="round">${t}</path>`;
}

/** `width` is the chart's on-screen width in CSS px; the height follows it within limits. */
export function paretoSVG(all: Attempt[], current: Attempt | null, width = 560): string {
  const W = Math.max(300, Math.round(width));
  const H = Math.round(Math.min(340, Math.max(240, W * 0.55)));
  const lx = (b: number) => Math.log10(Math.min(Math.max(b, X_MIN), X_MAX));
  const x = (b: number) => M.l + ((lx(b) - lx(X_MIN)) / (lx(X_MAX) - lx(X_MIN))) * (W - M.l - M.r);
  const y = (a: number) => M.t + (1 - (Math.max(a, Y_MIN) - Y_MIN) / (Y_MAX - Y_MIN)) * (H - M.t - M.b);
  const xt: number[] = [];
  for (const t of X_TICKS) if (!xt.length || x(t) - x(xt[xt.length - 1]) >= MIN_TICK_GAP) xt.push(t);

  const f = frontier(all);
  const p: string[] = [];
  p.push(`<rect x="${M.l}" y="${M.t}" width="${W - M.l - M.r}" height="${y(WORKS.accuracy) - M.t}" fill="var(--teal-wash)" opacity="0.55"/>`);
  for (const t of Y_TICKS) p.push(`<line x1="${M.l}" x2="${W - M.r}" y1="${y(t)}" y2="${y(t)}" stroke="var(--rule)" stroke-width="1"/><text x="${M.l - 8}" y="${y(t) + 4}" text-anchor="end" class="tick">${Math.round(t * 100)}%</text>`);
  for (const t of xt) p.push(`<line x1="${x(t)}" x2="${x(t)}" y1="${H - M.b}" y2="${H - M.b + 5}" stroke="var(--muted)"/><text x="${x(t)}" y="${H - M.b + 18}" text-anchor="middle" class="tick">${kb(t)}</text>`);
  p.push(`<text x="${(M.l + W - M.r) / 2}" y="${H - 6}" text-anchor="middle" class="axis">Size (log scale): smaller to the left</text>`);
  p.push(`<text transform="translate(14 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle" class="axis">Accuracy</text>`);
  p.push(`<line x1="${M.l}" x2="${W - M.r}" y1="${y(WORKS.accuracy)}" y2="${y(WORKS.accuracy)}" stroke="var(--teal)" stroke-dasharray="5 4" stroke-width="1.5"/>`);
  // Labelled just under the line at the small end, where few runs get. A run above it can still starve on pointing,
  // so the label states the accuracy half of the rule and the caption gives both halves.
  p.push(`<text x="${M.l + 6}" y="${y(WORKS.accuracy) + 16}" class="works">Healthy needs ${Math.round(WORKS.accuracy * 100)}% accuracy</text>`);
  if (f.length > 1) p.push(`<polyline points="${f.map((a) => `${x(a.bytes)},${y(a.accuracy)}`).join(" ")}" fill="none" stroke="var(--magenta)" stroke-width="1.5" opacity="0.7"/>`);
  for (const a of all) {
    const c = conditionOfAttempt(a);
    p.push(mark(c, x(a.bytes), y(a.accuracy), `${esc(a.label)}: ${kb(a.bytes)}, ${(a.accuracy * 100).toFixed(1)}% accuracy, ${(a.pointing * 100).toFixed(0)}% pointing, ${CONDITION_WORD[c].toLowerCase()}`));
  }
  if (current) p.push(`<circle cx="${x(current.bytes)}" cy="${y(current.accuracy)}" r="9" fill="none" stroke="var(--ink)" stroke-width="1.5"/>`);
  return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Size versus accuracy of every setting tried. The table below lists every point.">${p.join("")}</svg>`;
}

/** The chart's data as a table, smallest first, so keyboard, touch and screen-reader users get every point. */
export function paretoTable(all: Attempt[], current: Attempt | null): string {
  const best = new Set(frontier(all));
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const rows = [...all]
    .sort((a, b) => a.bytes - b.bytes)
    .map((a) => {
      const c = conditionOfAttempt(a);
      return `<tr${a === current ? ' aria-current="true"' : ""}><td>${esc(a.label)}</td><td class="num">${kb(a.bytes)}</td><td class="num">${pct(a.accuracy)}</td><td class="num">${pct(a.pointing)}</td><td class="c-${c}">${CONDITION_WORD[c]}${best.has(a) ? " · best tradeoff" : ""}</td></tr>`;
    })
    .join("");
  return `<table><thead><tr><th scope="col">Recipe</th><th scope="col" class="num">Size</th><th scope="col" class="num">Accuracy</th><th scope="col" class="num">Pointing</th><th scope="col">Condition</th></tr></thead><tbody>${rows}</tbody></table>`;
}

/**
 * Smallest attempt that is healthy and actually compressed: under `baselineBytes`, the same line
 * a run must cross to hatch a bird (coopbirds.ts hatches), so the record and the life list agree.
 */
export function smallestWorking(all: Attempt[], baselineBytes: number): Attempt | null {
  return all.filter((a) => works(a) && a.bytes < baselineBytes).sort((a, b) => a.bytes - b.bytes)[0] ?? null;
}

export { kb as formatSize };
