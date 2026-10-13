/**
 * The full table of an experiment and its reference points, as plain data the
 * page renders. Columns use the same filter the lab's own page used.
 */

import { fmt } from "../../lib/lab/results";
import { METRICS, cap, label } from "../../lib/lab/labels";

type Row = Record<string, number | string>;

const HIDDEN = new Set(["n_seeds", "series", "arm", "macs", "ece"]);

/** Every key across the rows except _std, seed counts and plot-only keys:
 *  the settings first, in the order they appear, then the metrics in report order. */
export function columns(rows: Row[]): string[] {
  const keys: string[] = [];
  for (const r of rows) for (const k of Object.keys(r)) if (!keys.includes(k)) keys.push(k);
  const shown = keys.filter((k) => !k.endsWith("_std") && !HIDDEN.has(k));
  return [...shown.filter((k) => !METRICS.includes(k)), ...METRICS.filter((k) => shown.includes(k))];
}

/** A column is numeric when every value it has is a number. */
export const isNumeric = (rows: Row[], k: string) =>
  rows.some((r) => r[k] !== undefined) && rows.every((r) => r[k] === undefined || typeof r[k] === "number");

export const columnLabel = (k: string) => cap(label(k));

/** Text of one cell: numbers through the lab's fmt(), anything else as written. */
export function cellText(v: number | string | undefined, k: string): string {
  if (v === undefined || v === null) return "";
  return typeof v === "number" ? fmt(v, k) : String(v);
}

/** Sizes, not measurements: they only vary across seeds when pruning does. */
const SIZES = new Set(["params", "macs", "kbits"]);

/**
 * The spread across seeds, when more than one seed ran (the lab only writes a
 * _std key then). A count or size every seed agreed on (a zero spread on a
 * whole number or a model size) gets none: "± 0.000" there is only noise.
 */
export function spread(r: Row, k: string): string | null {
  const sd = r[`${k}_std`], v = r[k];
  if (typeof sd !== "number" || typeof v !== "number") return null;
  if (sd === 0 && (Number.isInteger(v) || SIZES.has(k))) return null;
  return `± ${Math.abs(sd) < 10 ? sd.toFixed(3) : fmt(sd, k)}`;
}

const REF_WORDS: Record<string, { term: string; about: string }> = {
  teacher: {
    term: "Teacher (the large model)",
    about: "The full-size network trained on the labeled data; the smaller models here are measured against it.",
  },
  bayes: {
    term: "Bayes-optimal (the best any model can do on this problem)",
    about: "Predictions from the exact class probabilities, which the lab can compute because it generated the data.",
  },
};

export interface RefPoint {
  key: string;
  term: string;
  about: string;
  values: { label: string; text: string }[];
}

/** The experiment's reference points, with the metrics the table also shows. */
export function refPoints(refs: Record<string, Record<string, number>> | undefined, cols: string[]): RefPoint[] {
  const rank = (list: string[], k: string) => (list.includes(k) ? list.indexOf(k) : list.length);
  const order = (k: string) => (cols.includes(k) ? cols.indexOf(k) : cols.length + rank(METRICS, k));
  const known = Object.keys(REF_WORDS);
  return Object.entries(refs ?? {})
    .map(([key, vals]) => {
      const measured = Object.keys(vals).filter((k) => typeof vals[k] === "number" && !HIDDEN.has(k));
      const inTable = measured.filter((k) => cols.includes(k));
      const keys = (inTable.length ? inTable : measured).sort((a, b) => order(a) - order(b));
      const words = REF_WORDS[key] ?? { term: cap(key), about: "" };
      return { key, ...words, values: keys.map((k) => ({ label: label(k), text: fmt(vals[k], k) })) };
    })
    .filter((p) => p.values.length)
    .sort((a, b) => rank(known, a.key) - rank(known, b.key));
}
