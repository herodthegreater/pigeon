/**
 * Renders a method run's result: a table with a bar per row, an optional line
 * chart, the note and what ran. Ported from lab/web/static/app.js; the chart now
 * gives every series its own dash and marker (color is never the only cue) and
 * ships its numbers as a table, so nothing depends on hovering.
 */

import { esc } from "./markdown";
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
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(ylabel)} by ${esc(chart.x)}, ${names.length} series. The numbers are in the table below.">`;
  for (let k = 0; k <= 4; k++) {
    const v = y0 + ((y1 - y0) * k) / 4, y = Y(v);
    s += `<line class="grid" x1="${l}" x2="${W - r}" y1="${y}" y2="${y}"/><text x="${l - 6}" y="${y + 4}" text-anchor="end">${v.toFixed(2)}</text>`;
  }
  [...new Set(xs)].sort((a, c) => a - c).forEach((v) => {
    s += `<text x="${X(v)}" y="${H - b + 16}" text-anchor="middle">${+v.toFixed(3)}</text>`;
  });
  s += `<line class="axis" x1="${l}" x2="${W - r}" y1="${H - b}" y2="${H - b}"/>`;
  s += `<text x="${(W + l) / 2}" y="${H - 2}" text-anchor="middle">${esc(chart.x)}</text>`;
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
  s += `<details class="chart-data"><summary>Chart data</summary><div class="tablewrap" tabindex="0"><table><thead><tr><th scope="col">${esc(chart.x)}</th>${names.map((n) => `<th scope="col" class="num">${esc(n)}</th>`).join("")}</tr></thead><tbody>${xsSorted
    .map((x) => `<tr><th scope="row" class="num">${+x.toFixed(3)}</th>${names.map((n) => { const q = cell(n, x); return `<td class="num">${q ? q[1].toFixed(3) : ""}</td>`; }).join("")}</tr>`)
    .join("")}</tbody></table></div></details>`;
  return s;
}

export function resultView(res: RunResult): string {
  const metric = res.metric || "acc";
  const label = metric === "target_ap" ? "bird AP" : "accuracy";
  const max = Math.max(...res.rows.map((r) => Number(r[metric] ?? 0)), 1e-9);
  const extra = ["params", "fidelity", "sparsity"].filter((k) => res.rows.some((r) => r[k] !== undefined) && k !== metric);
  const other = metric === "acc" ? "target_ap" : "acc";
  const showOther = res.rows.some((r) => r[other] !== undefined);
  let h = `<div class="tablewrap" tabindex="0"><table class="restable"><thead><tr><th scope="col">Model</th><th scope="col" class="num">${label}</th>${showOther ? `<th scope="col" class="num">${other === "acc" ? "accuracy" : "bird AP"}</th>` : ""}${extra.map((k) => `<th scope="col" class="num">${k}</th>`).join("")}</tr></thead><tbody>`;
  for (const r of res.rows) {
    const v = r[metric] as number | undefined;
    h += `<tr><td>${esc(r.label)}${v !== undefined ? `<div class="bar" aria-hidden="true"><i style="width:${(100 * v) / max}%"></i></div>` : ""}</td><td class="num">${fmt(v)}</td>${showOther ? `<td class="num">${fmt(r[other])}</td>` : ""}${extra.map((k) => `<td class="num">${fmt(r[k], k)}</td>`).join("")}</tr>`;
  }
  h += "</tbody></table></div>";
  if (res.chart) h += lineChart(res.chart, label);
  if (res.note) h += `<p class="run-note">${esc(res.note)}</p>`;
  const ran = Object.entries(res.params).map(([k, v]) => `${k} ${v}`).join(", ") || "default settings";
  h += `<p class="run-meta">Ran in ${res.seconds.toFixed(1)} s with ${esc(ran)}.</p>`;
  return h;
}
