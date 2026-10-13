import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { experiment, experiments, report } from "../../lib/lab/data";
import { figureText } from "../../lib/lab/figures";
import { LABELS, METRICS } from "../../lib/lab/labels";
import { reportSource, reportView } from "./report";
import { columns, isNumeric, refPoints, spread } from "./table";

const reportPy = readFileSync(new URL("../../../../lab/report.py", import.meta.url), "utf8");

describe("labels", () => {
  it("match the axis labels lab/report.py draws on the figures", () => {
    const block = reportPy.match(/^LABELS = \{([\s\S]*?)\}/m)?.[1] ?? "";
    const py = Object.fromEntries([...block.matchAll(/"([^"]+)":\s*"([^"]*)"/g)].map((m) => [m[1], m[2]]));
    expect(Object.keys(py).length).toBeGreaterThan(10);
    expect(LABELS).toEqual(py);
  });

  it("order metrics the way lab/report.py does", () => {
    const list = reportPy.match(/^METRICS = \[([\s\S]*?)\]/m)?.[1] ?? "";
    expect(METRICS).toEqual([...list.matchAll(/"([^"]+)"/g)].map((m) => m[1]));
  });
});

describe("table", () => {
  const rows = [
    { method: "KD", width: 32, acc: 0.81, acc_std: 0.004, params: 2100, params_std: 0, nll: 0.6, nll_std: 0, kbits: 721.5, kbits_std: 0, n_seeds: 3, series: "x", macs: 1, ece: 0.1 },
    { method: "labels", width: 32, target_ap: 0.9, acc: 0.8, params: 2100, n_seeds: 1 },
  ];

  it("drops _std, seed counts and plot-only keys, settings before metrics", () => {
    expect(columns(rows)).toEqual(["method", "width", "acc", "target_ap", "nll", "params", "kbits"]);
  });

  it("knows numeric columns", () => {
    expect(isNumeric(rows, "width")).toBe(true);
    expect(isNumeric(rows, "method")).toBe(false);
  });

  it("shows the spread only where seeds ran, and not on counts every seed agreed on", () => {
    expect(spread(rows[0], "acc")).toBe("± 0.004");
    expect(spread(rows[0], "nll")).toBe("± 0.000");
    expect(spread(rows[0], "params")).toBeNull();
    expect(spread(rows[0], "kbits")).toBeNull();
    expect(spread(rows[1], "acc")).toBeNull();
  });

  it("puts the teacher before the Bayes-optimal ceiling, with the table's metrics", () => {
    const pts = refPoints({ bayes: { acc: 0.9734, target_ap: 0.999 }, teacher: { acc: 0.914, params: 140052, macs: 9, ece: 0.05 } }, ["method", "acc", "params"]);
    expect(pts.map((p) => p.key)).toEqual(["teacher", "bayes"]);
    expect(pts[0].values.map((v) => v.text)).toEqual(["0.914", "140,052"]);
    expect(pts[1].values.map((v) => v.text)).toEqual(["0.973"]);
  });
});

describe("report", () => {
  it("drops its own title and rule lines, and keeps a code-span link label", () => {
    const src = reportSource("# Title\n\nSee [`a/B.md`](a/B.md) and [site](https://x.org).\n\n---\n\n## Part\n\n```\n---\n```");
    expect(src).not.toMatch(/^# /m);
    expect(src).toContain("See `a/B.md` and [site](https://x.org).");
    expect(src.match(/^---$/gm)).toEqual(["---"]); // only the one inside the code fence
  });

  it("renders as one outline under the page's h1, linking every experiment it names", () => {
    const names = experiments().map((e) => e.name);
    const { html, toc } = reportView(report(), names, (n) => `/results/${n}/`);
    expect(html).not.toMatch(/<h1/);
    expect(html).not.toContain("<p>---</p>");
    expect(html).not.toContain("](");
    expect(toc.length).toBeGreaterThan(2);
    expect(new Set(toc.map((t) => t.id)).size).toBe(toc.length);
    for (const n of names) if (report().includes(`\`${n}\``)) expect(html).toContain(`href="/results/${n}/"`);
  });
});

describe("figures", () => {
  it("give every figure of an experiment its own name and alt text, from its plot spec", () => {
    for (const s of experiments()) {
      const e = experiment(s.name);
      const texts = e.figures.map((f, i) => figureText(f, i, e.figures.length, e));
      expect(e.plots?.length ?? 0, e.name).toBeGreaterThanOrEqual(e.figures.length);
      expect(new Set(texts.map((t) => t.name)).size, e.name).toBe(texts.length);
      expect(new Set(texts.map((t) => t.alt)).size, e.name).toBe(texts.length);
      for (const t of texts) {
        expect(t.name).not.toMatch(/^Figure \d/);
        expect(t.alt + t.name).not.toMatch(/[\u2014\u2013]/);
      }
    }
  });

  it("only mention a reference line the figure draws", () => {
    const spec = [{ x: "width", y: "acc", hlines: ["teacher", "bayes"] }];
    const one = figureText("figures/x_0.png", 0, 1, { title: "T", refs: { teacher: { acc: 0.9 } }, plots: spec });
    expect(one.alt).toContain("reference line for the teacher.");
    const both = figureText("figures/x_0.png", 0, 1, { title: "T", refs: { teacher: { acc: 0.9 }, bayes: { acc: 0.97 } }, plots: spec });
    expect(both.alt).toContain("for the teacher and the Bayes-optimal classifier.");
  });

  it("fall back to a numbered name when a figure has no spec", () => {
    const t = figureText("figures/x_3.png", 1, 2, { title: "T" });
    expect(t.name).toBe("Figure 2 of 2");
  });

  it("say where the numbers are: below on an experiment page, on Results elsewhere", () => {
    const spec = [{ x: "width", y: "acc" }];
    expect(figureText("figures/x_0.png", 0, 1, { title: "T", plots: spec }).alt).toMatch(/table below\.$/);
    expect(figureText("figures/x_0.png", 0, 1, { title: "T", plots: spec }, "results").alt).toMatch(/on the Results page\.$/);
  });
});
