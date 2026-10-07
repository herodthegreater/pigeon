/**
 * Names and alt text for an experiment's figures. The figure files are numbered
 * (`<name>_<i>.png`), so the words come from the plot spec that drew figure i:
 * the lab exports each experiment's specs as `plots` (lab/report.py renders
 * `<name>_<i>.png` from plots[i]). Every figure still gets a distinct name
 * without a spec.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { cap, label } from "./labels";
import type { PlotSpec } from "./data";

export type { PlotSpec };

/** Where the reader finds the figure's numbers, said at the end of the alt text. */
export type NumbersAt = "below" | "results";
const NUMBERS: Record<NumbersAt, string> = {
  below: "The numbers are in the table below.",
  results: "The numbers are in the experiment's full table on the Results page.",
};

const REF_NAME: Record<string, string> = { teacher: "the teacher", bayes: "the Bayes-optimal classifier" };

const list = (xs: string[]) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);

export interface FigureText {
  /** Short name, shown as the caption. */
  name: string;
  /** What the figure shows, for people who can't see it. */
  alt: string;
}

/**
 * Words for figure `index` (0-based, from the file name) of an experiment.
 * `refs` are the experiment's reference points; a dashed line is only
 * mentioned when the figure really draws one.
 */
export function figureText(
  file: string,
  position: number,
  count: number,
  exp: { title: string; refs?: Record<string, Record<string, number>>; plots?: PlotSpec[] },
  numbersAt: NumbersAt = "below",
): FigureText {
  const specs = exp.plots ?? [];
  const index = Number(file.match(/_(\d+)\.png$/)?.[1] ?? NaN);
  const spec = Number.isInteger(index) ? specs[index] : undefined;
  if (!spec) {
    const name = `Figure ${position + 1} of ${count}`;
    return { name, alt: `${name} for “${exp.title}”. ${NUMBERS[numbersAt]}` };
  }
  const kind = spec.kind ?? "line";
  const x = label(spec.x), y = label(spec.y);
  const sub = spec.title && spec.title.toLowerCase() !== y.toLowerCase() ? `: ${spec.title}` : "";
  const name = `${cap(y)} ${kind === "bar" ? "for each" : "against"} ${x}${sub}`;

  const what =
    kind === "bar" ? `Bar chart of ${y} for each ${x}` : `${kind === "scatter" ? "Scatter plot" : "Line chart"} of ${y} against ${x}`;
  const scale = spec.xlog && kind !== "bar" ? " (log scale)" : "";
  const group = !spec.group
    ? ""
    : ["series", "arm"].includes(spec.group)
      ? `, one ${kind === "bar" ? "bar" : kind === "scatter" ? "marker" : "line"} per configuration`
      : kind === "bar"
        ? `, bars split by ${label(spec.group)}`
        : `, one ${kind === "scatter" ? "marker" : "line"} per ${label(spec.group)}`;
  const tags = spec.label ? `, points labeled by ${label(spec.label)}` : "";
  const metric = spec.ref_metric ?? spec.y;
  const lines = (spec.hlines ?? []).filter((h) => typeof exp.refs?.[h]?.[metric] === "number").map((h) => REF_NAME[h] ?? h);
  const refs = lines.length ? `, with a dashed reference line for ${list(lines)}` : "";
  const alt = `${what}${scale}${group}${tags}${refs}. Titled “${spec.title ?? exp.title}”. ${NUMBERS[numbersAt]}`;
  return { name, alt };
}

/** Pixel size of a built figure, so the page reserves its space before it loads. */
export function pngSize(rel: string): { width: number; height: number } | null {
  for (const root of [join(process.cwd(), "public", "lab"), join(process.cwd(), "site", "public", "lab")]) {
    try {
      const b = readFileSync(join(root, rel));
      if (b.toString("latin1", 1, 4) === "PNG") return { width: b.readUInt32BE(16), height: b.readUInt32BE(20) };
    } catch {
      // not built here; try the next root, or render without a reserved size
    }
  }
  return null;
}
