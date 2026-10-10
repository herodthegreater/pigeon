/**
 * Words and numbers for a bird, shared by the run box and the life list, so a
 * bird reads the same wherever it shows up.
 */

import { CONDITION_WORD as MASCOT_WORD, type Condition } from "../../lib/ui/mascot";

/** A condition standing alone as a label, sentence case, as the Coop's bird card shows it: "Healthy". */
export const CONDITION_LABEL: Record<Condition, string> = MASCOT_WORD;

/** Condition words in running text: the same words the bird card uses (mascot.ts), lower case. */
export const CONDITION_WORD = Object.fromEntries(
  Object.entries(MASCOT_WORD).map(([k, v]) => [k, v.toLowerCase()]),
) as Record<Condition, string>;

/**
 * How many times smaller (ratio above 1) or bigger (below 1) a model is than its reference
 * (ratio = the reference's storage / this model's). Only an exact 1 is the same size: a model
 * that changed at all gets the digits it takes to show it (1.03), so a model 3% bigger never
 * reads as "the same size" or "1.0 times". `n` is null for a change under 0.1%.
 */
function times(ratio: number): { way: "same" } | { way: "smaller" | "bigger"; n: string | null } | null {
  if (!(ratio > 0) || !Number.isFinite(ratio)) return null;
  if (ratio === 1) return { way: "same" };
  const way = ratio > 1 ? "smaller" : "bigger";
  const x = ratio > 1 ? ratio : 1 / ratio;
  for (const digits of [1, 2, 3]) {
    const n = x.toFixed(digits);
    if (!/^1\.0+$/.test(n)) return { way, n };
  }
  return { way, n: null };
}

/** "6.2 times smaller than fp32", "1.03 times bigger than fp32", or (exactly 1) "the same size as fp32". */
export function ratioPhrase(ratio: number, reference: string): string {
  const t = times(ratio);
  if (!t) return `size unknown against ${reference}`;
  if (t.way === "same") return `the same size as ${reference}`;
  return t.n ? `${t.n} times ${t.way} than ${reference}` : `under 0.1% ${t.way} than ${reference}`;
}

/** The short form for tables and lists: "6.2× smaller", "1.03× bigger", or (exactly 1) "same size". */
export function ratioShort(ratio: number): string {
  const t = times(ratio);
  if (!t) return "unknown";
  if (t.way === "same") return "same size";
  return t.n ? `${t.n}× ${t.way}` : `under 0.1% ${t.way}`;
}

/** Percent with one decimal: 0.9987 -> "99.9%". */
export const pct = (v: number): string => `${(v * 100).toFixed(1)}%`;

/** What a lab run's metric is called in a sentence. */
export const metricWord = (metric: string | undefined): string => (metric === "target_ap" ? "target-class AP" : "accuracy");

/** "keeps 99.9% of its accuracy", or "scores 100.4% of its accuracy" when it beats the reference. */
export function keptPhrase(kept: number, metric?: string): string {
  return `${kept > 1.0005 ? "scores" : "keeps"} ${pct(kept)} of its ${metricWord(metric)}`;
}

/** A sighting's date for a reader: "9 Oct 2026". */
export function dateText(iso: string, withTime = false): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "date unknown";
  const day = d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
  if (!withTime) return day;
  return `${day}, ${d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })}`;
}

/** Join names the way a sentence does: "A", "A and B", "A, B and C". */
export const andList = (xs: string[]): string =>
  xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`;
