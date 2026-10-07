/**
 * Alt text and pixel sizes for the lab's simulation figures, worked out at build time.
 *
 * The figure files are named <experiment>_<n>.png, which says nothing about what
 * they show. What each one plots (title, x, y, chart kind, grouping) lives in the
 * plot specs of lab/results/<experiment>.json, the same specs lab/report.py draws
 * from, so the alt text describes the real chart. Axis wording mirrors LABELS in
 * lab/report.py.
 */

import { existsSync, openSync, readSync, closeSync } from "node:fs";
import { join } from "node:path";
import type { ExperimentSummary } from "../../lib/lab/data";

interface PlotSpec { x: string; y: string; kind?: string; group?: string; title?: string }

const results = import.meta.glob<{ plots?: PlotSpec[] }>("../../../../lab/results/*.json", { eager: true, import: "default" });
const specsFor = (name: string): PlotSpec[] => results[`../../../../lab/results/${name}.json`]?.plots ?? [];

const LABELS: Record<string, string> = {
  acc: "test accuracy (20-way)", target_ap: "target-class AP (bird)", fidelity: "agreement with teacher",
  nll: "test NLL", kl_bayes: "KL divergence from the Bayes-optimal model", params: "parameters (nonzero)",
  macs: "MACs per example (nonzero)", kbits: "storage (kbit)", teacher_acc: "teacher accuracy",
  teacher_kl_bayes: "teacher KL divergence from the Bayes-optimal model", width: "hidden width",
  sparsity: "fraction of weights removed", bits: "bits per weight", n_labels: "labeled examples",
  steps: "training steps", T: "temperature T", noise: "label-noise rate", teacher_width: "teacher width",
  dropped: "layers removed (of 8)", start_width: "width trained before pruning", generation: "generation",
  n_teachers: "teachers in the ensemble", smoothing: "teacher label smoothing", rank_or_width: "rank or width",
  overlap: "overlap",
  // x-axis keys report.py leaves unlabeled (it prints the raw key)
  alpha: "KD weight alpha", path: "training path", method: "method", data: "recovery data set",
  init: "initialization", source: "transfer-set source",
};
const GROUPS: Record<string, string> = {
  series: "setting", arm: "setting", method: "method", student: "student", alpha: "KD weight alpha",
  model: "model", family: "model family",
};
const label = (k: string) => LABELS[k] ?? k.replace(/_/g, " ");

function describe(spec: PlotSpec): string {
  const x = label(spec.x), y = label(spec.y), g = spec.group ? GROUPS[spec.group] ?? spec.group.replace(/_/g, " ") : "";
  if (spec.kind === "bar") return `Bar chart of ${y} for each ${x}${g ? `, one bar per ${g}` : ""}`;
  if (spec.kind === "scatter") return `Scatter plot of ${y} against ${x}${g ? `, one marker shape per ${g}` : ""}`;
  return `Line chart of ${y} against ${x}${g ? `, one line per ${g}` : ""}`;
}

/** Width and height from a PNG's header, so lazy-loaded figures don't shift the page. */
function pngSize(rel: string): { width: number; height: number } | null {
  for (const root of [join(process.cwd(), "public", "lab"), join(process.cwd(), "site", "public", "lab")]) {
    const file = join(root, rel);
    if (!existsSync(file)) continue;
    const buf = Buffer.alloc(24);
    const fd = openSync(file, "r");
    try { readSync(fd, buf, 0, 24, 0); } finally { closeSync(fd); }
    if (buf.toString("ascii", 1, 4) !== "PNG") return null;
    return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
  }
  return null;
}

export interface Figure { rel: string; alt: string; width?: number; height?: number }

/** One entry per figure of an experiment, with alt text that differs between siblings. */
export function figuresOf(e: ExperimentSummary): Figure[] {
  const specs = specsFor(e.name);
  const out: Figure[] = e.figures.map((rel, i) => {
    const n = Number(rel.match(/_(\d+)\.png$/)?.[1] ?? i);
    const spec = specs[n];
    const title = /[.?!]$/.test(e.title) ? e.title : `${e.title}.`;
    const alt = spec
      ? `${title} ${describe(spec)}${spec.title ? `, titled “${spec.title}”` : ""}.`
      : `${title} Figure ${i + 1} of ${e.figures.length}.`;
    return { rel, alt, ...(pngSize(rel) ?? {}) };
  });
  // Siblings must never share alt text: number any that still collide.
  const seen = new Map<string, number>();
  out.forEach((f) => seen.set(f.alt, (seen.get(f.alt) ?? 0) + 1));
  out.forEach((f, i) => { if (seen.get(f.alt)! > 1) f.alt = `${f.alt.replace(/\.$/, "")}, figure ${i + 1} of ${out.length}.`; });
  return out;
}
