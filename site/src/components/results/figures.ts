/**
 * Names and alt text for an experiment's figures. The figure files are numbered
 * (`<name>_<i>.png`), so the words come from the plot spec that drew figure i:
 * lab/results/<name>.json keeps one spec per figure (axes, title, chart kind),
 * and lab/report.py renders `<name>_<i>.png` from spec i. The page data the
 * site reads does not carry the specs yet, so they are read from the committed
 * results files here; every figure still gets a distinct name without them.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { cap, label } from "./labels";

export interface PlotSpec {
  x: string;
  y: string;
  kind?: "line" | "bar" | "scatter";
  group?: string;
  title?: string;
  xlog?: boolean;
  hlines?: string[];
  ref_metric?: string;
  label?: string;
  connect?: boolean;
}

const specFiles = import.meta.glob<PlotSpec[] | undefined>("../../../../lab/results/*.json", {
  eager: true,
  import: "plots",
});

/** The plot specs of one experiment, by figure index ([] when unknown). */
export function plotSpecs(name: string): PlotSpec[] {
  const hit = Object.entries(specFiles).find(([path]) => path.endsWith(`/lab/results/${name}.json`));
  return (hit?.[1] as PlotSpec[] | undefined) ?? [];
}

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
  exp: { title: string; refs?: Record<string, Record<string, number>> },
  specs: PlotSpec[],
): FigureText {
  const index = Number(file.match(/_(\d+)\.png$/)?.[1] ?? NaN);
  const spec = Number.isInteger(index) ? specs[index] : undefined;
  if (!spec) {
    const name = `Figure ${position + 1} of ${count}`;
    return { name, alt: `${name} for “${exp.title}”. The numbers are in the table below.` };
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
  const alt = `${what}${scale}${group}${tags}${refs}. Titled “${spec.title ?? exp.title}”. The numbers are in the table below.`;
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
