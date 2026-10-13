import { describe, expect, it } from "vitest";
import { lessons, methods } from "../../lib/lab/data";
import { slug } from "../../lib/lab/markdown";
import { PRESETS } from "../../lib/ui/recipe";
import { bareTitle, CHAPTERS, course, sectionsOf } from "./course";

describe("Flight School reading map", () => {
  const chapters = course();

  it("has twelve chapters, numbered in order", () => {
    expect(chapters.map((c) => c.n)).toEqual(Array.from({ length: 12 }, (_, i) => i + 1));
  });

  it("cites only sections, lessons and methods that exist (nothing dropped)", () => {
    for (const c of chapters) expect(c.dropped, `chapter ${c.n}`).toEqual([]);
  });

  it("links every cited section to the id the Field Guide gives that heading", () => {
    const ids = new Set(methods().map((m) => m.id));
    for (const c of CHAPTERS) {
      for (const r of c.reading) {
        expect(ids.has(r.method), r.method).toBe(true);
        expect(sectionsOf(r.method), `${r.method}: ${r.section}`).toContain(r.section);
        const link = course()[c.n - 1].links.find((l) => l.href.endsWith(`guide/${r.method}/#${slug(r.section)}`));
        expect(link, `${r.method}#${slug(r.section)}`).toBeDefined();
      }
    }
  });

  it("uses only presets the Coop has", () => {
    const presets = new Set(PRESETS.map((p) => p.id));
    for (const c of CHAPTERS) for (const p of c.coop) expect(presets.has(p.preset), p.preset).toBe(true);
  });

  it("says what is missing for every chapter with a gap, in plain words", () => {
    for (const c of CHAPTERS) {
      if (c.gap) expect(c.gap.startsWith("Not written yet: ")).toBe(true);
      expect(`${c.title} ${c.subtitle} ${c.gap ?? ""}`).not.toMatch(/[\u2014\u2013\u2192]/);
    }
    expect(CHAPTERS[0].gap).toBeTruthy();
  });

  it("names each lesson link after the lesson's own title", () => {
    const titles = new Map(lessons().map((l) => [l.id, bareTitle(l.title)]));
    const fake = chapters[1].links.find((l) => l.kind === "code")!;
    expect(fake.label).toBe(`Code it: ${titles.get("fake_quant")}`);
  });
});

describe("bareTitle", () => {
  it("drops the lesson number", () => {
    expect(bareTitle("10. GPTQ's error-compensation loop")).toBe("GPTQ's error-compensation loop");
    expect(bareTitle("No number")).toBe("No number");
  });
});
