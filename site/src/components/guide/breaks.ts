/**
 * Pigeon's warning on a method page quotes the write-up's own "When it breaks"
 * section: the first sentence or two of its first point, verbatim, so the note
 * never says anything the write-up doesn't. No section, no note.
 */

export const BREAKS_HEADING = "When it breaks";

export interface BreaksNote {
  /** Markdown, cut from the write-up: render it with inline(). */
  text: string;
  /** How many points the section lists (1 when it is prose, not a list). */
  points: number;
}

const ITEM = /^\s*(?:[-*]|\d+\.)\s+/;

/** The body of a "## <heading>" section, up to the next heading of level 1 or 2. */
export function section(page: string, heading: string): string[] | null {
  const lines = page.replace(/\r/g, "").split("\n");
  const start = lines.findIndex((l) => l.trim() === `## ${heading}`);
  if (start < 0) return null;
  const rest = lines.slice(start + 1);
  const end = rest.findIndex((l) => /^#{1,2} /.test(l));
  return end < 0 ? rest : rest.slice(0, end);
}

/** The section's first point (a list item, or the first paragraph) as one line, and how many points it has. */
function firstPoint(body: string[]): { text: string; points: number } | null {
  let i = body.findIndex((l) => l.trim() && !/^#{3,} /.test(l));
  if (i < 0) return null;
  const isItem = ITEM.test(body[i]);
  const parts = [body[i].replace(ITEM, "").trim()];
  for (i++; i < body.length; i++) {
    const l = body[i];
    if (!l.trim() || ITEM.test(l) || /^#{1,6} |^```|^\|/.test(l)) break;
    if (isItem && !/^\s{2,}\S/.test(l)) break;
    parts.push(l.trim());
  }
  const points = isItem ? body.filter((l) => ITEM.test(l) && !/^\s{2,}/.test(l)).length : 1;
  return { text: parts.join(" "), points };
}

/**
 * Where each sentence of `text` ends: after ., ! or ? (and any closing ** * ) or quote),
 * outside `code`, when a space and a capital, a backtick or emphasis follows.
 */
function sentenceEnds(text: string): number[] {
  const ends: number[] = [];
  let code = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === "`") code = !code;
    if (code || !".!?".includes(c)) continue;
    let j = i + 1;
    while (j < text.length && "*)\"'".includes(text[j])) j++;
    if (j >= text.length) ends.push(j);
    else if (/\s/.test(text[j]) && /^\s+[A-Z`*"(]/.test(text.slice(j))) ends.push(j);
  }
  if (!ends.length || ends[ends.length - 1] !== text.length) ends.push(text.length);
  return ends;
}

const balanced = (s: string) => (s.match(/`/g) ?? []).length % 2 === 0 && (s.match(/\*\*/g) ?? []).length % 2 === 0;

/**
 * The first sentence, or two when the first is short, cut so that no `code` or
 * **bold** is left open. Returns null when the write-up has no such section.
 */
export function whenItBreaks(page: string, opts: { maxChars?: number } = {}): BreaksNote | null {
  const { maxChars = 220 } = opts;
  const body = section(page, BREAKS_HEADING);
  if (!body) return null;
  const first = firstPoint(body);
  if (!first || !first.text) return null;
  const ends = sentenceEnds(first.text).filter((e) => balanced(first.text.slice(0, e)));
  // A point that opens with a bold title ("**Outliers.** One large weight...") gets
  // the title plus its first sentence or two: the title alone says too little.
  const titled = /^\*\*[^*]+\*\*$/.test(first.text.slice(0, ends[0]).trim()) && ends.length > 1;
  const from = titled ? 1 : 0;
  let cut = ends[from];
  if (ends.length > from + 1 && ends[from + 1] <= maxChars) cut = ends[from + 1];
  return { text: first.text.slice(0, cut).trim(), points: first.points };
}
