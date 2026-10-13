import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { inline, mathErrors, md, untypesetMath } from "./markdown";

describe("math", () => {
  it("typesets inline $...$ with MathML for screen readers", () => {
    const h = inline("the loss $\\KL(p \\,\\|\\, q)$ grows");
    expect(h).toContain('class="katex"');
    expect(h).toContain("<math");
    expect(h.startsWith("the loss ")).toBe(true);
    expect(h.endsWith(" grows")).toBe(true);
  });

  it("typesets inline math that ends in an escaped symbol", () => {
    expect(inline("levels $\\{-s, 0, s\\}$ here")).toContain('class="katex"');
    expect(untypesetMath(md("levels $\\{-s, 0, s\\}$ here"))).toEqual([]);
    expect(untypesetMath("<p>raw $x_i$ left</p>")).toEqual(["$x_i$"]);
  });

  it("leaves money, code and escaped dollars alone", () => {
    expect(inline("costs $5 and $10")).toBe("costs $5 and $10");
    expect(inline("`x = $y$`")).toBe("<code>x = $y$</code>");
    expect(inline("a \\$3 fee")).toBe("a $3 fee");
  });

  it("keeps emphasis outside formulas and never inside", () => {
    const h = inline("**bold** and $a * b * c$");
    expect(h).toContain("<strong>bold</strong>");
    expect(h).not.toContain("<em>");
  });

  it("renders $$ blocks as display equations, one-line or multi-line", () => {
    const one = md("$$ e^{i\\pi} + 1 = 0 $$");
    const many = md("Before.\n\n$$\nx = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}\n$$\n\nAfter.");
    for (const h of [one, many]) {
      expect(h).toContain('<div class="math-display" tabindex="0">');
      expect(h).toContain("katex-display");
    }
    expect(many).toContain("<p>Before.</p>");
    expect(many).toContain("<p>After.</p>");
  });

  it("shows and records a formula KaTeX can't read", () => {
    const before = mathErrors.length;
    expect(inline("$\\frac{1}{$ and more")).toContain('class="math-error"');
    expect(md("$$\\frac{1}{2$$")).toContain('class="math-error"');
    expect(mathErrors.map((e) => e.tex)).toEqual([...mathErrors.slice(0, before).map((e) => e.tex), "\\frac{1}{", "\\frac{1}{2"]);
    mathErrors.length = before;
  });
});

// Every write-up, lesson and the report: each formula must typeset. The page data is
// built by `python -m lab.web.build_static` (the workflow runs it before these tests).
const DATA = join(__dirname, "../../data/lab");
const REPO = join(__dirname, "../../../..");
describe.skipIf(!existsSync(DATA))("math in the lab's content", () => {
  const read = (p: string) => JSON.parse(readFileSync(p, "utf8"));
  const sources: [string, () => string][] = [];
  for (const f of readdirSync(join(DATA, "methods")).filter((f) => f.endsWith(".json"))) {
    sources.push([`write-up ${f}`, () => md(read(join(DATA, "methods", f)).page)]);
  }
  for (const f of readdirSync(join(DATA, "lessons")).filter((f) => f.endsWith(".json"))) {
    sources.push([`lesson ${f}`, () => {
      const L = read(join(DATA, "lessons", f));
      return md(L.intro) + (L.hints || []).map((h: string) => inline(h)).join("");
    }]);
  }
  sources.push(["REPORT.md", () => md(readFileSync(join(REPO, "lab/REPORT.md"), "utf8"))]);

  it.each(sources)("%s typesets cleanly", (_, render) => {
    const before = mathErrors.length;
    const html = render();
    const bad = mathErrors.splice(before);
    expect(bad, bad.map((e) => `${e.tex}\n  ${e.message}`).join("\n")).toEqual([]);
    expect(untypesetMath(html), "math the parser didn't pick up").toEqual([]);
  });
});

describe("lists around display math", () => {
  it("a numbered list resumes at its own number after an equation", () => {
    const h = md("1. first\n2. second\n\n$$\na = b\n$$\n\n3. third");
    expect(h).toContain("<ol><li>first</li><li>second</li></ol>");
    expect(h).toContain('<ol start="3"><li>third</li></ol>');
  });
});
