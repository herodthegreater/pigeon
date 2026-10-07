#!/usr/bin/env node
/**
 * Checks the built site (dist/) for the things a broken page hides best:
 *   - every internal link and #anchor resolves
 *   - every page has a unique <title> and exactly one <h1>
 *   - heading levels never skip (h2 -> h4)
 *   - every <img> has alt text, and no two images on a page share it
 *   - no em dash or emoji in visible text
 *   - every form control has a label
 * Usage: node scripts/check-site.mjs [distDir]   (run after `astro build`)
 * Exits 1 and lists every problem when anything fails.
 */
import { readdirSync, readFileSync, statSync, existsSync } from "node:fs";
import { join, relative, dirname, posix } from "node:path";

const DIST = process.argv[2] ?? "dist";
const BASE = "/pigeon/";
const problems = [];
const fail = (page, msg) => problems.push(`${page}: ${msg}`);

function walk(dir) {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

const files = walk(DIST);
const pages = files.filter((f) => f.endsWith(".html"));
const fileSet = new Set(files.map((f) => "/" + relative(DIST, f).split("\\").join("/")));

const strip = (html) =>
  html
    .replace(/<script\b[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[\s\S]*?<\/style>/gi, " ")
    .replace(/<template\b[\s\S]*?<\/template>/gi, " ")
    .replace(/<pre\b[\s\S]*?<\/pre>/gi, " ")
    .replace(/<code\b[\s\S]*?<\/code>/gi, " ")
    .replace(/<[^>]+>/g, " ");

const ids = new Map(); // page path -> Set of ids
const pageOf = (f) => "/" + relative(DIST, f).split("\\").join("/");
for (const f of pages) {
  const html = readFileSync(f, "utf8");
  ids.set(pageOf(f), new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1])));
}

/** Map a site URL path (/pigeon/x/) to a dist file path (/x/index.html). */
function resolveUrl(path) {
  if (!path.startsWith(BASE)) return null;
  let p = "/" + path.slice(BASE.length);
  if (p.endsWith("/")) p += "index.html";
  if (fileSet.has(p)) return p;
  if (fileSet.has(p + "/index.html")) return p + "/index.html";
  if (fileSet.has(p + ".html")) return p + ".html";
  return null;
}

const titles = new Map();
const EMOJI = /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{1F000}-\u{1F2FF}]/u;

for (const f of pages) {
  const page = pageOf(f);
  const html = readFileSync(f, "utf8");
  const here = BASE + page.slice(1).replace(/index\.html$/, "");

  // titles
  const title = html.match(/<title>([\s\S]*?)<\/title>/)?.[1]?.trim();
  if (!title) fail(page, "no <title>");
  else titles.set(title, [...(titles.get(title) ?? []), page]);

  // headings
  const hs = [...html.matchAll(/<h([1-6])\b/g)].map((m) => Number(m[1]));
  const h1s = hs.filter((n) => n === 1).length;
  if (h1s !== 1) fail(page, `${h1s} <h1> elements (want exactly 1)`);
  hs.forEach((n, i) => { if (i && n > hs[i - 1] + 1) fail(page, `heading jumps from h${hs[i - 1]} to h${n}`); });

  // images
  const alts = [];
  for (const m of html.matchAll(/<img\b[^>]*>/g)) {
    const alt = m[0].match(/\salt="([^"]*)"/);
    if (!alt) fail(page, `img without alt: ${m[0].slice(0, 80)}`);
    else alts.push(alt[1]);
  }
  const dupAlts = alts.filter((a, i) => a && alts.indexOf(a) !== i);
  if (dupAlts.length) fail(page, `duplicate alt text: ${[...new Set(dupAlts)].join(" | ")}`);

  // visible text
  const text = strip(html);
  if (text.includes("\u2014")) fail(page, `em dash in text: "...${text.slice(Math.max(0, text.indexOf("\u2014") - 40), text.indexOf("\u2014") + 20).replace(/\s+/g, " ")}..."`);
  if (EMOJI.test(text)) fail(page, `emoji in text: ${text.match(EMOJI)[0]}`);

  // form controls need a label
  const labelled = new Set([...html.matchAll(/<label\b[^>]*\sfor="([^"]+)"/g)].map((m) => m[1]));
  for (const m of html.matchAll(/<(input|select|textarea)\b([^>]*)>/g)) {
    const attrs = m[2];
    if (/type="(hidden|submit|button)"/.test(attrs)) continue;
    const id = attrs.match(/\sid="([^"]+)"/)?.[1];
    const ok = /aria-label(ledby)?="/.test(attrs) || (id && labelled.has(id));
    // a control wrapped by its <label> counts too
    const before = html.slice(0, m.index);
    const wrapped = before.lastIndexOf("<label") > before.lastIndexOf("</label>");
    if (!ok && !wrapped) fail(page, `${m[1]} without a label: <${m[1]}${attrs.slice(0, 60)}>`);
  }

  // links
  for (const m of html.matchAll(/<a\b[^>]*\shref="([^"]+)"/g)) {
    const href = m[1].replace(/&amp;/g, "&");
    if (href === "#") { fail(page, 'href="#" goes nowhere'); continue; }
    if (/^(https?:|mailto:)/.test(href)) continue;
    const url = new URL(href, "https://site.test" + here);
    if (url.host !== "site.test") continue;
    const target = resolveUrl(url.pathname);
    if (!target) { fail(page, `broken link: ${href}`); continue; }
    const anchor = decodeURIComponent(url.hash.slice(1));
    if (anchor && !ids.get(target)?.has(anchor)) fail(page, `missing anchor: ${href}`);
  }
}

for (const [title, ps] of titles) if (ps.length > 1) fail(ps.join(", "), `share the title "${title}"`);

if (problems.length) {
  console.error(`check-site: ${problems.length} problem(s) in ${pages.length} pages\n`);
  for (const p of problems) console.error("  " + p);
  process.exit(1);
}
console.log(`check-site: ${pages.length} pages, every link, anchor, title, heading and label checks out`);
