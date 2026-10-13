/**
 * The markdown subset the lab's write-ups, lessons and report use, ported from
 * lab/web/static/app.js so pages render the same, now at build time. Pure
 * string functions: safe on the server and in the browser.
 *
 * Math is LaTeX, typeset by KaTeX into HTML (plus MathML for screen readers):
 * `$...$` inline, and a `$$` ... `$$` block on lines of its own for display
 * equations. `\$` is a literal dollar sign. Code spans are never math.
 */
import katex from "katex";

/** Shorthands every page can use, so the same thing is always written the same way. */
export const MATH_MACROS: Record<string, string> = {
  "\\softmax": "\\operatorname{softmax}",
  "\\KL": "\\operatorname{KL}",
  "\\diag": "\\operatorname{diag}",
  "\\round": "\\operatorname{round}",
  "\\clip": "\\operatorname{clip}",
  "\\sign": "\\operatorname{sign}",
  "\\relu": "\\operatorname{ReLU}",
  "\\rank": "\\operatorname{rank}",
  "\\argmax": "\\operatorname*{arg\\,max}",
  "\\argmin": "\\operatorname*{arg\\,min}",
  "\\R": "\\mathbb{R}",
  "\\E": "\\mathbb{E}",
};

/** Formulas KaTeX couldn't parse, so tests (and the build log) can name them. */
export const mathErrors: { tex: string; message: string }[] = [];

/** Typeset one formula. A parse error is shown as the raw source, marked, and recorded. */
export function renderMath(tex: string, display = false): string {
  try {
    return katex.renderToString(tex, {
      displayMode: display,
      throwOnError: true,
      strict: "error",
      output: "htmlAndMathml",
      macros: { ...MATH_MACROS },
    });
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    mathErrors.push({ tex, message });
    return `<code class="math-error" title="${esc(message)}">${esc(tex)}</code>`;
  }
}

// $...$ : not opened by a space or closed after one, never touching a word or a
// digit outside, so "$5 and $10" stays money and `$x$-th` still works.
const INLINE_MATH = /(?<![\\\w$])\$(?=[^\s$])((?:\\.|[^$\\\n])*?(?:\\.|[^\s\\]))\$(?![\w$])/g;

/** Math-looking `$...$` left in rendered HTML (outside code and KaTeX output): a
 *  formula the parser didn't pick up, which would otherwise show as raw TeX. */
export function untypesetMath(html: string): string[] {
  const text = html
    .replace(/<pre[\s\S]*?<\/pre>/g, "")
    .replace(/<code[\s\S]*?<\/code>/g, "")
    .replace(/<span class="katex[\s\S]*?<\/math><\/span>/g, "")
    .replace(/<[^>]+>/g, " ");
  return text.match(/\$[^$\n]*[\\^_{}][^$\n]*\$/g) ?? [];
}

export const esc = (s: unknown) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

const PY_RE =
  /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|[rbf]?"(?:\\.|[^"\\\n])*"|[rbf]?'(?:\\.|[^'\\\n])*')|(@[\w.]+)|\b(def|class|return|if|elif|else|for|while|in|not|and|or|import|from|as|with|try|except|finally|raise|lambda|None|True|False|is|yield|pass|break|continue|del|global|nonlocal|assert)\b|\b(\d+(?:\.\d*)?(?:e[-+]?\d+)?)\b/g;

/** Python syntax highlighting: spans with tok-* classes, everything escaped. */
export function highlight(code: string): string {
  let out = "", last = 0, m: RegExpExecArray | null;
  const re = new RegExp(PY_RE.source, "g");
  while ((m = re.exec(code))) {
    out += esc(code.slice(last, m.index));
    const cls = m[1] ? "com" : m[2] ? "str" : m[3] ? "dec" : m[4] ? "kw" : "num";
    out += `<span class="tok-${cls}">${esc(m[0])}</span>`;
    last = re.lastIndex;
  }
  return out + esc(code.slice(last));
}

/** Inline markdown: `code`, $math$, **bold**, *italic*, [links](https://…). Other link schemes become plain text. */
export function inline(s: string): string {
  return s
    .split(/(`[^`]*`)/)
    .map((part) => {
      if (part.startsWith("`") && part.endsWith("`") && part.length > 1) return `<code>${esc(part.slice(1, -1))}</code>`;
      // Math first, into placeholders, so **, * and links never reach inside a formula.
      const maths: string[] = [];
      const text = part
        .replace(/\\\$/g, "\u0001")
        .replace(INLINE_MATH, (_, tex: string) => `\u0000${maths.push(renderMath(tex)) - 1}\u0000`)
        .replace(/\u0001/g, "$");
      return esc(text)
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        // *emphasis* can't open or close on a space, so "2 * 3 * 4" stays arithmetic
        .replace(/(^|[^*\w])\*(?=\S)([^*\n]*?\S)\*(?!\w)/g, "$1<em>$2</em>")
        .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, t: string, u: string) =>
          /^https?:/.test(u) ? `<a href="${u}" target="_blank" rel="noopener">${t}<span class="visually-hidden"> (opens in a new tab)</span></a>` : t,
        )
        .replace(/\u0000(\d+)\u0000/g, (_, k: string) => maths[Number(k)]);
    })
    .join("");
}

export const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");

export interface Heading { id: string; text: string }

/**
 * Block markdown. `shift` demotes every heading (shift 1 turns # into h2), so a
 * document can sit under a page's own h1. Level-2 headings (after shifting) are
 * collected into `toc` when given.
 */
export function md(src: string, opts: { toc?: Heading[]; shift?: number } = {}): string {
  const { toc, shift = 0 } = opts;
  const L = (src || "").replace(/\r/g, "").split("\n");
  let h = "", i = 0;
  const isBlock = (l: string) => /^```|^\$\$|^#{1,4} |^\s*[-*] |^\s*\d+\. |^\||^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(l);
  while (i < L.length) {
    const l = L[i];
    if (/^```/.test(l)) {
      const lang = l.slice(3).trim();
      const buf: string[] = [];
      i++;
      while (i < L.length && !/^```/.test(L[i])) buf.push(L[i++]);
      i++;
      const code = buf.join("\n");
      h += `<pre tabindex="0"><code>${lang === "python" || lang === "py" || !lang ? highlight(code) : esc(code)}</code></pre>`;
    } else if (/^\$\$/.test(l)) {
      // display math: "$$ ... $$" on one line, or $$ / lines / $$
      const one = l.trim().match(/^\$\$(.+)\$\$$/);
      let tex: string;
      if (one) {
        tex = one[1];
        i++;
      } else {
        const buf: string[] = [l.trim().slice(2)];
        i++;
        while (i < L.length && !/\$\$\s*$/.test(L[i])) buf.push(L[i++]);
        if (i < L.length) buf.push(L[i++].replace(/\$\$\s*$/, ""));
        tex = buf.join("\n");
      }
      h += `<div class="math-display" tabindex="0">${renderMath(tex.trim(), true)}</div>`;
    } else if (/^#{1,4} /.test(l)) {
      const raw = l.match(/^#+/)![0].length, n = Math.min(6, raw + shift);
      const text = l.slice(raw + 1).trim(), id = slug(text);
      if (toc && n === 2) toc.push({ id, text });
      h += `<h${n} id="${id}">${inline(text)}</h${n}>`;
      i++;
    } else if (/^\s*[-*] /.test(l) || /^\s*\d+\. /.test(l)) {
      const ordered = /^\s*\d+\. /.test(l), re = ordered ? /^\s*\d+\. / : /^\s*[-*] /;
      const items: string[] = [];
      while (i < L.length && re.test(L[i])) {
        let item = L[i++].replace(re, "");
        while (i < L.length && /^\s{2,}\S/.test(L[i]) && !re.test(L[i])) item += " " + L[i++].trim();
        items.push(`<li>${inline(item)}</li>`);
      }
      // an ordered list keeps its first number, so a list that resumes after a display
      // equation ("3.") continues at 3 instead of starting over
      const start = ordered ? parseInt(l.trim(), 10) : 1;
      h += ordered ? `<ol${start !== 1 ? ` start="${start}"` : ""}>${items.join("")}</ol>` : `<ul>${items.join("")}</ul>`;
    } else if (/^\|/.test(l)) {
      const rows: string[] = [];
      while (i < L.length && /^\|/.test(L[i])) rows.push(L[i++]);
      const cells = (r: string) => r.replace(/^\||\|\s*$/g, "").split("|").map((c) => c.trim());
      const body = rows.filter((r, k) => !(k === 1 && /^\|[\s:|-]+$/.test(r)));
      h +=
        '<div class="tablewrap" tabindex="0"><table><thead><tr>' +
        cells(body[0]).map((c) => `<th scope="col">${inline(c)}</th>`).join("") +
        "</tr></thead><tbody>" +
        body.slice(1).map((r) => "<tr>" + cells(r).map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>").join("") +
        "</tbody></table></div>";
    } else if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(l)) {
      i++; // a thematic break: sections already open with a rule (prose h2), so draw nothing extra
    } else if (!l.trim()) {
      i++;
    } else {
      const buf: string[] = [];
      while (i < L.length && L[i].trim() && !isBlock(L[i])) buf.push(L[i++]);
      h += `<p>${inline(buf.join(" "))}</p>`;
    }
  }
  return h;
}
