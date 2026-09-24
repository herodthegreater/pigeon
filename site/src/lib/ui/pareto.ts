/**
 * Size vs accuracy: every setting the visitor has tried, the best-tradeoff
 * frontier, and the "still works" line. Plain SVG, drawn to one scale.
 */

import { WORKS } from "./mascot";

export interface Attempt {
  label: string;
  bytes: number;
  accuracy: number;
  pointing: number;
}

const W = 560, H = 300, M = { l: 52, r: 16, t: 16, b: 42 };
const X_MIN = 20e3, X_MAX = 5e6, Y_MIN = 0.45, Y_MAX = 1.0;
const x = (b: number) => M.l + ((Math.log10(Math.min(Math.max(b, X_MIN), X_MAX)) - Math.log10(X_MIN)) / (Math.log10(X_MAX) - Math.log10(X_MIN))) * (W - M.l - M.r);
const y = (a: number) => M.t + (1 - (Math.max(a, Y_MIN) - Y_MIN) / (Y_MAX - Y_MIN)) * (H - M.t - M.b);

export const works = (a: Attempt) => a.accuracy >= WORKS.accuracy && a.pointing >= WORKS.pointing;

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

export function paretoSVG(all: Attempt[], current: Attempt | null): string {
  const xt = [20e3, 50e3, 100e3, 250e3, 500e3, 1e6, 2e6, 5e6];
  const yt = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  const f = frontier(all);
  const p: string[] = [];
  p.push(`<rect x="${M.l}" y="${M.t}" width="${W - M.l - M.r}" height="${y(WORKS.accuracy) - M.t}" fill="var(--teal-wash)" opacity="0.55"/>`);
  for (const t of yt) p.push(`<line x1="${M.l}" x2="${W - M.r}" y1="${y(t)}" y2="${y(t)}" stroke="var(--rule)" stroke-width="1"/><text x="${M.l - 8}" y="${y(t) + 4}" text-anchor="end" class="tick">${Math.round(t * 100)}%</text>`);
  for (const t of xt) p.push(`<line x1="${x(t)}" x2="${x(t)}" y1="${H - M.b}" y2="${H - M.b + 5}" stroke="var(--faint)"/><text x="${x(t)}" y="${H - M.b + 18}" text-anchor="middle" class="tick">${kb(t)}</text>`);
  p.push(`<text x="${(M.l + W - M.r) / 2}" y="${H - 4}" text-anchor="middle" class="axis">model size (log scale) → smaller is to the left</text>`);
  p.push(`<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle" class="axis">accuracy</text>`);
  p.push(`<line x1="${M.l}" x2="${W - M.r}" y1="${y(WORKS.accuracy)}" y2="${y(WORKS.accuracy)}" stroke="var(--teal)" stroke-dasharray="5 4" stroke-width="1.5"/>`);
  p.push(`<text x="${W - M.r - 4}" y="${y(WORKS.accuracy) - 6}" text-anchor="end" class="works">still works ↑</text>`);
  if (f.length > 1) p.push(`<polyline points="${f.map((a) => `${x(a.bytes)},${y(a.accuracy)}`).join(" ")}" fill="none" stroke="var(--magenta)" stroke-width="1.5" opacity="0.7"/>`);
  for (const a of all) {
    const ok = works(a);
    p.push(`<circle cx="${x(a.bytes)}" cy="${y(a.accuracy)}" r="4.5" fill="${ok ? "var(--teal)" : "var(--card)"}" stroke="${ok ? "var(--teal)" : "var(--warn)"}" stroke-width="1.6"><title>${esc(a.label)}: ${kb(a.bytes)}, ${(a.accuracy * 100).toFixed(1)}% accuracy, ${(a.pointing * 100).toFixed(0)}% pointing</title></circle>`);
  }
  if (current) p.push(`<circle cx="${x(current.bytes)}" cy="${y(current.accuracy)}" r="9" fill="none" stroke="var(--ink)" stroke-width="1.5"/>`);
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Size versus accuracy of every setting tried">${p.join("")}</svg>`;
}

/** Smallest attempt that still works — and is actually compressed (under `baselineBytes`). */
export function smallestWorking(all: Attempt[], baselineBytes: number): Attempt | null {
  return all.filter((a) => works(a) && a.bytes < baselineBytes * 0.99).sort((a, b) => a.bytes - b.bytes)[0] ?? null;
}

export { kb as formatSize };
