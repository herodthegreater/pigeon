/**
 * Words and numbers for a bird, shared by the run box and the life list, so a
 * bird reads the same wherever it shows up.
 */

import type { Condition } from "../../lib/ui/mascot";

/** The condition in the site's words (the Coop says the same). */
export const CONDITION_WORD: Record<Condition, string> = { healthy: "healthy", emaciated: "starving", skeleton: "skeleton" };

/** "6.2 times smaller than fp32", "the same size as fp32", or "1.1 times bigger than fp32". */
export function ratioPhrase(ratio: number, reference: string): string {
  if (!(ratio > 0) || !Number.isFinite(ratio)) return `size unknown against ${reference}`;
  const x = ratio >= 1 ? ratio : 1 / ratio;
  const n = x.toFixed(1);
  if (n === "1.0") return `the same size as ${reference}`;
  return ratio > 1 ? `${n} times smaller than ${reference}` : `${n} times bigger than ${reference}`;
}

/** The short form for tables and lists: "6.2× smaller", "same size", "1.1× bigger". */
export function ratioShort(ratio: number): string {
  if (!(ratio > 0) || !Number.isFinite(ratio)) return "unknown";
  const x = ratio >= 1 ? ratio : 1 / ratio;
  const n = x.toFixed(1);
  if (n === "1.0") return "same size";
  return ratio > 1 ? `${n}× smaller` : `${n}× bigger`;
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
