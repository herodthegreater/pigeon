/**
 * The lab's write-up (lab/REPORT.md) as the body of the Results page.
 */

import { md, type Heading } from "../../lib/lab/markdown";

/**
 * Prepare the report's markdown for a page that has its own h1:
 * - its "# " title goes (the page's h1 is "Results"), so its "## " sections are the page's h2s;
 * - "---" breaks go: md() has no rule for them (they would print as text), and each
 *   one sits right above an h2, which already draws a rule;
 * - a relative link whose label is a code span keeps just the label, which is what
 *   md() does for any non-http link (a code-span label otherwise splits the link).
 */
export function reportSource(text: string): string {
  const lines = (text || "").replace(/\r/g, "").split("\n");
  const first = lines.findIndex((l) => l.trim() !== "");
  if (first >= 0 && /^# /.test(lines[first])) lines.splice(first, 1);
  let fenced = false;
  return lines
    .filter((l) => {
      if (/^```/.test(l)) fenced = !fenced;
      return fenced || !/^\s*-{3,}\s*$/.test(l);
    })
    .join("\n")
    .replace(/\[(`[^`\n]+`)\]\((?!https?:)[^)\s]+\)/g, "$1");
}

/**
 * The report as HTML plus its contents (the h2s). The report names each
 * experiment in code (`size_ladder`); those names link to the experiment's page.
 */
export function reportView(text: string, names: Iterable<string>, href: (name: string) => string): { html: string; toc: Heading[] } {
  const toc: Heading[] = [];
  const known = new Set(names);
  const html = md(reportSource(text), { toc }).replace(/(?<!<pre[^>]*>)<code>([a-z0-9_]+)<\/code>/g, (m, name: string) =>
    known.has(name) ? `<a href="${href(name)}">${m}</a>` : m,
  );
  return { html, toc };
}
