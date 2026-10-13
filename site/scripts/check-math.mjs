// Typeset the math in markdown files with the site's own renderer and report any
// formula KaTeX can't read. For writers, before the full `npm test`:
//   node --experimental-strip-types scripts/check-math.mjs ../lab/web/content/methods/gptq.md ...
//   node --experimental-strip-types scripts/check-math.mjs --lessons   # lab/lessons.py intros and hints
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { inline, mathErrors, md } from "../src/lib/lab/markdown.ts";

const args = process.argv.slice(2);
const docs = [];
for (const a of args) {
  if (a === "--lessons") {
    const py = "import json; from lab.lessons import LESSONS; print(json.dumps([dict(id=L.id, intro=L.intro, hints=L.hints) for L in LESSONS]))";
    const out = execFileSync("python3", ["-c", py], { cwd: new URL("../..", import.meta.url).pathname, encoding: "utf8" });
    for (const L of JSON.parse(out)) docs.push([`lesson ${L.id}`, () => { md(L.intro); L.hints.forEach((h) => inline(h)); }]);
  } else {
    docs.push([a, () => md(readFileSync(a, "utf8"))]);
  }
}
let bad = 0, formulas = 0;
for (const [name, render] of docs) {
  const before = mathErrors.length;
  render();
  const errs = mathErrors.slice(before);
  for (const e of errs) console.log(`${name}: ${e.message}\n    ${e.tex}`);
  bad += errs.length;
}
// count typeset formulas, so a page that quietly lost its math shows up as 0
for (const a of args.filter((a) => a !== "--lessons")) {
  const n = (md(readFileSync(a, "utf8")).match(/class="katex"/g) || []).length;
  formulas += n;
  console.log(`${String(n).padStart(4)} formulas  ${a}`);
}
console.log(bad ? `${bad} formula(s) failed` : `all formulas typeset (${docs.length} document(s))`);
process.exit(bad ? 1 : 0);
