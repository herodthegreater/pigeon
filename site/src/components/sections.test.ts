import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const read = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");

describe("the 404 page's directory", () => {
  // The header's sections live in Base.astro's frontmatter, out of reach of an import, so this
  // reads both files: every section the header links must be among the 404's "Pages that do exist".
  const header = [...read("../layouts/Base.astro").matchAll(/\["\w+", "[^"]*", "([^"]+)", "([\w-]+\/)"\]/g)].map((m) => ({ name: m[1], href: m[2] }));
  const notFound = read("../pages/404.astro");

  it("lists every section the header does, the life list included", () => {
    expect(header.map((h) => h.href)).toContain("life-list/");
    for (const h of header) expect(notFound, h.href).toContain("href: `${base}" + h.href + "`");
  });

  it("counts the sections from its own list, never a number written by hand", () => {
    expect(notFound).not.toMatch(/site's (one|two|three|four|five|six|seven) sections/);
    expect(notFound).toContain("site's {sectionsText} are below");
  });
});
