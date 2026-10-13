/**
 * Renders a method run's result: a table with a bar per row, an optional line
 * chart, the note and what ran. Ported from lab/web/static/app.js; the chart now
 * gives every series its own dash and marker (color is never the only cue) and
 * ships its numbers as a table, so nothing depends on hovering. Columns and
 * settings use the same words as the Results tables (labels.ts).
 */

import { esc } from "./markdown";
import { cap, label as keyLabel } from "./labels";
import type { RunResult } from "./python";

export const fmt = (v: unknown, k?: string): string => {
  if (v === undefined || v === null) return "";
  if (typeof v !== "number") return esc(v);
  if (k === "params" || Number.isInteger(v)) return v.toLocaleString("en-US");
  return Math.abs(v) < 10 ? v.toFixed(3) : v.toFixed(1);
};

const SERIES = [
  { color: "var(--series-1)", dash: "", marker: "circle" },
  { color: "var(--series-2)", dash: "6 4", marker: "square" },
  { color: "var(--series-3)", dash: "2 3", marker: "triangle" },
  { color: "var(--series-4)", dash: "10 3 2 3", marker: "diamond" },
] as const;

function marker(kind: (typeof SERIES)[number]["marker"], x: number, y: number, color: string): string {
  const s = 4.5;
  const style = `fill="${color}" stroke="var(--card)" stroke-width="1.5"`;
  if (kind === "circle") return `<circle cx="${x}" cy="${y}" r="${s}" ${style}/>`;
  if (kind === "square") return `<rect x="${x - s}" y="${y - s}" width="${2 * s}" height="${2 * s}" ${style}/>`;
  if (kind === "triangle") return `<path d="M ${x} ${y - s - 1} L ${x + s + 1} ${y + s} L ${x - s - 1} ${y + s} Z" ${style}/>`;
  return `<path d="M ${x} ${y - s - 1} L ${x + s + 1} ${y} L ${x} ${y + s + 1} L ${x - s - 1} ${y} Z" ${style}/>`;
}

function legendKey(i: number): string {
  const k = SERIES[i % SERIES.length];
  return `<svg width="34" height="12" viewBox="0 0 34 12" aria-hidden="true"><line x1="1" x2="33" y1="6" y2="6" stroke="${k.color}" stroke-width="2" ${k.dash ? `stroke-dasharray="${k.dash}"` : ""}/>${marker(k.marker, 17, 6, k.color)}</svg>`;
}

export function lineChart(chart: NonNullable<RunResult["chart"]>, ylabel: string): string {
  const names = Object.keys(chart.series);
  const pts = names.flatMap((n) => chart.series[n]);
  if (!pts.length) return "";
  const W = 360, H = 210, l = 44, r = 12, t = 12, b = 34;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (x0 === x1) { x0 -= 1; x1 += 1; }
  const pad = (y1 - y0) * 0.1 || 0.05;
  y0 = Math.max(0, y0 - pad); y1 = y1 + pad;
  const X = (v: number) => l + ((v - x0) / (x1 - x0)) * (W - l - r);
  const Y = (v: number) => H - b - ((v - y0) / (y1 - y0)) * (H - t - b);
  const xlabel = cap(keyLabel(chart.x)); // the same words as the Results figures' axes
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(ylabel)} by ${esc(keyLabel(chart.x))}, ${names.length} series. The numbers are in the table below.">`;
  for (let k = 0; k <= 4; k++) {
    const v = y0 + ((y1 - y0) * k) / 4, y = Y(v);
    s += `<line class="grid" x1="${l}" x2="${W - r}" y1="${y}" y2="${y}"/><text x="${l - 6}" y="${y + 4}" text-anchor="end">${v.toFixed(2)}</text>`;
  }
  [...new Set(xs)].sort((a, c) => a - c).forEach((v) => {
    s += `<text x="${X(v)}" y="${H - b + 16}" text-anchor="middle">${+v.toFixed(3)}</text>`;
  });
  s += `<line class="axis" x1="${l}" x2="${W - r}" y1="${H - b}" y2="${H - b}"/>`;
  s += `<text x="${(W + l) / 2}" y="${H - 2}" text-anchor="middle">${esc(xlabel)}</text>`;
  names.forEach((n, k) => {
    const st = SERIES[k % SERIES.length];
    const p = [...chart.series[n]].sort((a, c) => a[0] - c[0]);
    s += `<polyline fill="none" stroke="${st.color}" stroke-width="2" ${st.dash ? `stroke-dasharray="${st.dash}"` : ""} points="${p.map((q) => `${X(q[0])},${Y(q[1])}`).join(" ")}"/>`;
    p.forEach((q) => { s += marker(st.marker, X(q[0]), Y(q[1]), st.color); });
  });
  s += "</svg>";
  if (names.length > 1) s += `<ul class="chart-legend">${names.map((n, k) => `<li>${legendKey(k)}${esc(n)}</li>`).join("")}</ul>`;

  // The same numbers as a table: every point, readable without a mouse.
  const xsSorted = [...new Set(xs)].sort((a, c) => a - c);
  const cell = (n: string, x: number) => chart.series[n].find((q) => q[0] === x);
  s += `<details class="chart-data"><summary>Chart data</summary><div class="tablewrap" tabindex="0"><table><thead><tr><th scope="col">${esc(xlabel)}</th>${names.map((n) => `<th scope="col" class="num">${esc(n)}</th>`).join("")}</tr></thead><tbody>${xsSorted
    .map((x) => `<tr><th scope="row" class="num">${+x.toFixed(3)}</th>${names.map((n) => { const q = cell(n, x); return `<td class="num">${q ? q[1].toFixed(3) : ""}</td>`; }).join("")}</tr>`)
    .join("")}</tbody></table></div></details>`;
  return s;
}

/** A run's settings in the method's own words: "student width 16, temperature 4, student sees pool". */
export function settingsText(params: Record<string, unknown>, labels: Record<string, string> = {}): string {
  return Object.entries(params)
    .map(([k, v]) => {
      const name = labels[k] ?? k;
      return typeof v === "boolean" ? `${name}: ${v ? "yes" : "no"}` : `${name} ${v}`;
    })
    .join(", ");
}

/*
 * What a run does while the status line counts seconds. The method data has no field that
 * says whether a run trains, so the runs that don't are listed here, from lab/methods.py:
 * each loads a model the site ships (lab/models), rounds or prunes its weights, and scores
 * it (lab/tests/test_quotes.py checks most of them for that reason). Keep this in step with
 * the run_* functions there.
 */
/** Runs that only round a shipped model's weights. */
const ROUNDS_ONLY = new Set(["ptq", "rtn", "gptq", "awq"]);
/** Runs that train nothing but do more than round weights: SmoothQuant also rounds activations, the rest prune or drop layers. */
const NO_TRAINING = new Set(["smoothquant", "wanda", "sparsegpt", "block_influence"]);
/** Runs that train only to recover what the cut lost: with that setting off, they train nothing. */
const RECOVERY_OFF: Record<string, (p: Record<string, unknown>) => boolean> = {
  magnitude_pruning: (p) => p.recovery === "none",
  layer_dropping: (p) => p.recovery === "none",
  low_rank: (p) => p.recovery === "none",
  weight_sharing: (p) => p.finetune === false,
};

/** The status word for a run in progress: "Training" only for a run that trains. */
export function runActivity(method: string, params: Record<string, unknown> = {}): "Training" | "Rounding the weights" | "Running" {
  if (ROUNDS_ONLY.has(method)) return "Rounding the weights";
  if (NO_TRAINING.has(method) || RECOVERY_OFF[method]?.(params)) return "Running";
  return "Training";
}

/**
 * Methods whose runs never make a model smaller: the lab says so with an unconditional
 * bird_reason (lab/methods.py: born_again, overparameterize, small_tricks).
 */
export const SAME_SIZE = new Set(["born_again", "overparameterize", "small_tricks"]);

/** The run box's line before a first run, true of what this method's run does. */
export function idleLine(method: string): string {
  if (SAME_SIZE.has(method)) {
    return "A run trains real models on the lab's toy problem and scores them, here in your browser. It changes how a model is trained, not its size, so it hatches no bird. The first run downloads Python (about 20 MB); most runs then take 5 to 60 seconds.";
  }
  const what = ROUNDS_ONLY.has(method)
    ? "A run loads the lab's trained model, rounds its weights and scores the result on the lab's toy problem, here in your browser. Nothing is trained."
    : NO_TRAINING.has(method)
      ? "A run loads the lab's trained model, compresses it and scores the result on the lab's toy problem, here in your browser. Nothing is trained."
      : "A run builds or loads real models on the lab's toy problem, compresses them and scores them, here in your browser.";
  return `${what} The first run downloads Python (about 20 MB); most runs then take 5 to 60 seconds. Every model a run makes smaller ends in a bird.`;
}

/** Extra columns a run can carry, in the order the table shows them. Storage is what a run's birds are sized by. */
const EXTRA = ["kbits", "params", "fidelity", "sparsity"];

/** `labels` maps each setting's key to the words its field uses on the page. */
export function resultView(res: RunResult, labels: Record<string, string> = {}): string {
  const metric = res.metric || "acc";
  const head = (k: string) => esc(cap(keyLabel(k)));
  const rows = res.rows as Record<string, unknown>[];
  const max = Math.max(...rows.map((r) => Number(r[metric] ?? 0)), 1e-9);
  const extra = EXTRA.filter((k) => rows.some((r) => r[k] !== undefined) && k !== metric);
  const other = metric === "acc" ? "target_ap" : "acc";
  const showOther = rows.some((r) => r[other] !== undefined);
  let h = `<div class="tablewrap" tabindex="0" role="region" aria-label="Run results"><table class="restable"><thead><tr><th scope="col">Model</th><th scope="col" class="num">${head(metric)}</th>${showOther ? `<th scope="col" class="num">${head(other)}</th>` : ""}${extra.map((k) => `<th scope="col" class="num">${head(k)}</th>`).join("")}</tr></thead><tbody>`;
  for (const r of rows) {
    const v = r[metric] as number | undefined;
    // The row every bird is measured against, said in words when its label doesn't already.
    const ref = r.ref === true && !/reference/i.test(String(r.label)) ? ` <span class="ref-note">(reference)</span>` : "";
    h += `<tr><td>${esc(r.label)}${ref}${v !== undefined ? `<div class="bar" aria-hidden="true"><i style="width:${(100 * v) / max}%"></i></div>` : ""}</td><td class="num">${fmt(v)}</td>${showOther ? `<td class="num">${fmt(r[other])}</td>` : ""}${extra.map((k) => `<td class="num">${fmt(r[k], k)}</td>`).join("")}</tr>`;
  }
  h += "</tbody></table></div>";
  if (res.chart) h += lineChart(res.chart, cap(keyLabel(metric)));
  if (res.note) h += `<p class="run-note">${esc(res.note)}</p>`;
  const ran = settingsText(res.params, labels) || "default settings";
  h += `<p class="run-meta">Ran in ${res.seconds.toFixed(1)} s with ${esc(ran)}.</p>`;
  return h;
}
