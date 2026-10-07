import { describe, expect, it } from "vitest";
import { esc, highlight, inline, md, slug, type Heading } from "./markdown";
import { fmt, lineChart, resultView } from "./results";

describe("markdown", () => {
  it("escapes HTML in text, code and highlighting", () => {
    expect(esc(`<a href="x">&'`)).toBe("&lt;a href=&quot;x&quot;&gt;&amp;&#39;");
    expect(inline("a <b> `<c>`")).toBe("a &lt;b&gt; <code>&lt;c&gt;</code>");
    expect(md("```\nx = '<script>'\n```")).not.toContain("<script>");
  });

  it("renders the inline subset", () => {
    expect(inline("**bold** and *it*")).toBe("<strong>bold</strong> and <em>it</em>");
    expect(inline("2 * 3 * 4")).toBe("2 * 3 * 4");
    expect(inline("[x](https://e.org)")).toContain('href="https://e.org" target="_blank" rel="noopener"');
    expect(inline("[x](https://e.org)")).toContain("(opens in a new tab)");
    // other schemes, including javascript:, become plain text
    const js = inline("[x](javascript:alert(1))");
    expect(js).not.toContain("<a");
    expect(js).not.toContain("javascript");
    expect(inline("[repo file](results/RESULTS.md)")).toBe("repo file");
  });

  it("slugs headings the way anchors are written", () => {
    expect(slug("The code, line by line")).toBe("the-code-line-by-line");
    expect(slug("  When it breaks! ")).toBe("when-it-breaks");
  });

  it("collects h2s for the contents list and can shift levels", () => {
    const toc: Heading[] = [];
    const h = md("# Top\n\n## The idea\n\ntext\n\n## How it works", { toc });
    expect(h).toContain('<h1 id="top">Top</h1>');
    expect(toc).toEqual([{ id: "the-idea", text: "The idea" }, { id: "how-it-works", text: "How it works" }]);
    const shifted: Heading[] = [];
    expect(md("# Report\n\n## Part", { toc: shifted, shift: 1 })).toContain('<h2 id="report">Report</h2><h3 id="part">Part</h3>');
    expect(shifted).toEqual([{ id: "report", text: "Report" }]);
  });

  it("renders lists, tables, paragraphs and code", () => {
    expect(md("- a\n- b")).toBe("<ul><li>a</li><li>b</li></ul>");
    expect(md("1. a\n   continued\n2. b")).toBe("<ol><li>a continued</li><li>b</li></ol>");
    const t = md("| a | b |\n|---|---|\n| 1 | 2 |");
    expect(t).toContain('<th scope="col">a</th>');
    expect(t).toContain("<td>1</td><td>2</td>");
    expect(t).not.toContain("---");
    expect(md("one\ntwo\n\nthree")).toBe("<p>one two</p><p>three</p>");
    expect(md("```python\ndef f(): return 1\n```")).toContain('<span class="tok-kw">def</span>');
  });

  it("highlights Python tokens", () => {
    const h = highlight('x = 1  # one\ns = "hi"');
    expect(h).toContain('<span class="tok-num">1</span>');
    expect(h).toContain('<span class="tok-com"># one</span>');
    expect(h).toContain('<span class="tok-str">&quot;hi&quot;</span>');
  });
});

describe("results", () => {
  it("formats numbers like the lab", () => {
    expect(fmt(0.91234)).toBe("0.912");
    expect(fmt(12.345)).toBe("12.3");
    expect(fmt(137000, "params")).toBe("137,000");
    expect(fmt(undefined)).toBe("");
    expect(fmt("<x>")).toBe("&lt;x&gt;");
  });

  const run = {
    rows: [
      { label: "teacher", acc: 0.914, params: 137000 },
      { label: "student <b>", acc: 0.828, params: 2132 },
    ],
    params: { bits: 4 },
    seconds: 3.21,
    note: "a note",
    chart: { x: "bits", series: { teacher: [[8, 0.91], [4, 0.9]] as [number, number][], student: [[8, 0.83], [4, 0.7]] as [number, number][] } },
  };

  it("renders a result table with escaped labels and what ran", () => {
    const h = resultView(run);
    expect(h).toContain('<th scope="col" class="num">accuracy</th>');
    expect(h).toContain("student &lt;b&gt;");
    expect(h).toContain("137,000");
    expect(h).toContain("Ran in 3.2 s with bits 4.");
    expect(h).toContain('<p class="run-note">a note</p>');
  });

  it("gives every series its own dash and marker, and the data as a table", () => {
    const c = lineChart(run.chart, "accuracy");
    expect(c).toContain('role="img"');
    expect(c).toContain('stroke-dasharray="6 4"'); // second series is dashed
    expect(c).toContain("<rect"); // second series uses square markers
    expect(c).toContain("<summary>Chart data</summary>");
    expect(c).toContain('<th scope="row" class="num">4</th><td class="num">0.900</td><td class="num">0.700</td>');
    expect(lineChart({ x: "bits", series: {} }, "accuracy")).toBe("");
  });
});
