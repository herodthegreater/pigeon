// pigeon lab: a small hash-routed single page app. No dependencies.
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const app = $("#app");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const store = {
  get(k, d = null) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};

// Two backends behind one function. Served by `python -m lab.web`, everything
// goes to the FastAPI server. In the static build (GitHub Pages) page data is
// pre-rendered JSON, and runs + lesson checks execute in the browser through
// Pyodide in a Web Worker (worker.js).
const STATIC = !!window.LAB_STATIC;

async function api(path, body) {
  if (STATIC) return staticApi(path, body);
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `${r.status} ${r.statusText}`);
  return data;
}

let worker = null, seq = 0;
const pending = new Map();
function pyStatus(text) { window.dispatchEvent(new CustomEvent("pystatus", { detail: text })); }
function pyCall(kind, args, timeoutMs) {
  if (!worker) {
    worker = new Worker("worker.js");
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "status") return pyStatus(m.text);
      const p = pending.get(m.id);
      if (!p) return;
      pending.delete(m.id); clearTimeout(p.timer);
      m.ok ? p.resolve(m.result) : p.reject(new Error(m.error));
    };
  }
  return new Promise((resolve, reject) => {
    const id = ++seq;
    const timer = timeoutMs ? setTimeout(() => {
      // a runaway check: kill the whole worker (Python can't be interrupted otherwise)
      worker.terminate(); worker = null;
      for (const [, p] of pending) { clearTimeout(p.timer); p.reject(new Error(`timed out after ${timeoutMs / 1000}s (Python was restarted)`)); }
      pending.clear();
    }, timeoutMs) : null;
    pending.set(id, { resolve, reject, timer });
    worker.postMessage({ id, kind, args });
  });
}

async function staticApi(path, body) {
  let m;
  if (body !== undefined && (m = path.match(/^\/api\/methods\/([^/]+)\/run$/))) {
    const t0 = performance.now();
    const res = await pyCall("run", { mid: m[1], params: body.params });
    return { ...res, seconds: ((performance.now() - t0) / 1000).toFixed(1) };
  }
  if (body !== undefined && (m = path.match(/^\/api\/lessons\/([^/]+)\/check$/))) {
    await pyCall("warm", {});                       // loading Python doesn't count against the limit
    return pyCall("check", { lid: m[1], code: body.code }, 30000);
  }
  const r = await fetch(path.replace(/^\/api\//, "data/") + ".json");
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

// ---------------------------------------------------------------- theme
(function theme() {
  const btn = $("#theme");
  const apply = (t) => {
    if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
    const dark = t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    btn.textContent = dark ? "Light" : "Dark";
  };
  apply(store.get("theme"));
  btn.onclick = () => {
    const dark = btn.textContent === "Dark";
    store.set("theme", dark ? "dark" : "light");
    apply(dark ? "dark" : "light");
  };
})();

// ---------------------------------------------------------------- python highlighting
const PY_RE = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|[rbf]?"(?:\\.|[^"\\\n])*"|[rbf]?'(?:\\.|[^'\\\n])*')|(@[\w.]+)|\b(def|class|return|if|elif|else|for|while|in|not|and|or|import|from|as|with|try|except|finally|raise|lambda|None|True|False|is|yield|pass|break|continue|del|global|nonlocal|assert)\b|\b(\d+(?:\.\d*)?(?:e[-+]?\d+)?)\b/g;
function highlight(code) {
  let out = "", last = 0, m;
  PY_RE.lastIndex = 0;
  while ((m = PY_RE.exec(code))) {
    out += esc(code.slice(last, m.index));
    const cls = m[1] ? "com" : m[2] ? "str" : m[3] ? "dec" : m[4] ? "kw" : "num";
    out += `<span class="tok-${cls}">${esc(m[0])}</span>`;
    last = PY_RE.lastIndex;
  }
  return out + esc(code.slice(last));
}

// ---------------------------------------------------------------- math (same rules as site/src/lib/lab/markdown.ts)
const MATH_MACROS = {
  "\\softmax": "\\operatorname{softmax}", "\\KL": "\\operatorname{KL}", "\\diag": "\\operatorname{diag}",
  "\\round": "\\operatorname{round}", "\\clip": "\\operatorname{clip}", "\\sign": "\\operatorname{sign}",
  "\\relu": "\\operatorname{ReLU}", "\\rank": "\\operatorname{rank}", "\\argmax": "\\operatorname*{arg\\,max}",
  "\\argmin": "\\operatorname*{arg\\,min}", "\\R": "\\mathbb{R}", "\\E": "\\mathbb{E}",
};
const INLINE_MATH = /(?<![\\\w$])\$(?=[^\s$])((?:\\.|[^$\\\n])*?(?:\\.|[^\s\\]))\$(?![\w$])/g;
function renderMath(tex, display = false) {
  if (!window.katex) return `<code>${esc(tex)}</code>`;  // vendor/katex didn't load: show the source
  try {
    return window.katex.renderToString(tex, { displayMode: display, throwOnError: true, output: "htmlAndMathml", macros: { ...MATH_MACROS } });
  } catch (e) {
    return `<code class="math-error" title="${esc(e.message)}">${esc(tex)}</code>`;
  }
}

// ---------------------------------------------------------------- markdown (the subset our pages use)
function inline(s) {
  return s.split(/(`[^`]*`)/).map((part) => {
    if (part.startsWith("`") && part.endsWith("`") && part.length > 1) return `<code>${esc(part.slice(1, -1))}</code>`;
    const maths = [];
    const text = part.replace(/\\\$/g, "\u0001")
      .replace(INLINE_MATH, (_, tex) => `\u0000${maths.push(renderMath(tex)) - 1}\u0000`)
      .replace(/\u0001/g, "$");
    return esc(text)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>")
      .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, t, u) =>
        /^(https?:|#|\/)/.test(u) ? `<a href="${u}"${u.startsWith("http") ? ' target="_blank" rel="noopener"' : ""}>${t}</a>` : t)
      .replace(/\u0000(\d+)\u0000/g, (_, k) => maths[Number(k)]);
  }).join("");
}
const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
function md(src, toc) {
  const L = (src || "").replace(/\r/g, "").split("\n");
  let h = "", i = 0;
  const isBlock = (l) => /^```|^\$\$|^#{1,4} |^\s*[-*] |^\s*\d+\. |^\|/.test(l);
  while (i < L.length) {
    const l = L[i];
    if (/^```/.test(l)) {
      const lang = l.slice(3).trim(); const buf = []; i++;
      while (i < L.length && !/^```/.test(L[i])) buf.push(L[i++]);
      i++;
      const code = buf.join("\n");
      h += `<pre><code>${lang === "python" || lang === "py" || !lang ? highlight(code) : esc(code)}</code></pre>`;
    } else if (/^\$\$/.test(l)) {
      const one = l.trim().match(/^\$\$(.+)\$\$$/);
      let tex;
      if (one) { tex = one[1]; i++; }
      else {
        const buf = [l.trim().slice(2)]; i++;
        while (i < L.length && !/\$\$\s*$/.test(L[i])) buf.push(L[i++]);
        if (i < L.length) buf.push(L[i++].replace(/\$\$\s*$/, ""));
        tex = buf.join("\n");
      }
      h += `<div class="math-display" tabindex="0">${renderMath(tex.trim(), true)}</div>`;
    } else if (/^#{1,4} /.test(l)) {
      const n = l.match(/^#+/)[0].length, text = l.slice(n + 1).trim(), id = slug(text);
      if (toc && n === 2) toc.push({ id, text });
      h += `<h${n} id="${id}">${inline(text)}</h${n}>`; i++;
    } else if (/^\s*[-*] /.test(l) || /^\s*\d+\. /.test(l)) {
      const ordered = /^\s*\d+\. /.test(l), re = ordered ? /^\s*\d+\. / : /^\s*[-*] /;
      const items = [];
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
      const rows = [];
      while (i < L.length && /^\|/.test(L[i])) rows.push(L[i++]);
      const cells = (r) => r.replace(/^\||\|\s*$/g, "").split("|").map((c) => c.trim());
      const body = rows.filter((r, k) => !(k === 1 && /^\|[\s:|-]+$/.test(r)));
      h += "<div class=\"tablewrap\"><table><thead><tr>" + cells(body[0]).map((c) => `<th>${inline(c)}</th>`).join("") + "</tr></thead><tbody>" +
        body.slice(1).map((r) => "<tr>" + cells(r).map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>").join("") +
        "</tbody></table></div>";
    } else if (!l.trim()) {
      i++;
    } else {
      const buf = [];
      while (i < L.length && L[i].trim() && !isBlock(L[i])) buf.push(L[i++]);
      h += `<p>${inline(buf.join(" "))}</p>`;
    }
  }
  return h;
}

// ---------------------------------------------------------------- shared bits
const STATUS = {
  runnable: ["runnable", "runs here"],
  cockpit: ["cockpit", "runs in the cockpit"],
  reference: ["reference", "reference only"],
};
const pill = (s) => `<span class="pill ${STATUS[s][0]}">${STATUS[s][1]}</span>`;
const fmt = (v, k) => {
  if (v === undefined || v === null) return "";
  if (typeof v !== "number") return esc(v);
  if (k === "params" || Number.isInteger(v)) return v.toLocaleString();
  return Math.abs(v) < 10 ? v.toFixed(3) : v.toFixed(1);
};
let META = null, METHODS = null;
async function meta() {
  if (!META) [META, METHODS] = await Promise.all([api("/api/meta"), api("/api/methods")]);
  return { META, METHODS };
}
function setNav(key) {
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("on", a.dataset.k === key));
}
function render(html, navKey) {
  setNav(navKey);
  app.innerHTML = html;
  window.scrollTo(0, 0);
}

// ---------------------------------------------------------------- tooltip + line chart
const tip = document.createElement("div");
tip.className = "tip"; tip.hidden = true; document.body.appendChild(tip);
document.addEventListener("mouseover", (e) => {
  const t = e.target.closest("[data-tip]");
  if (!t) { tip.hidden = true; return; }
  tip.textContent = t.dataset.tip; tip.hidden = false;
});
document.addEventListener("mousemove", (e) => {
  if (!tip.hidden) { tip.style.left = e.clientX + 12 + "px"; tip.style.top = e.clientY - 28 + "px"; }
});

function lineChart(chart, ylabel = "accuracy") {
  const names = Object.keys(chart.series);
  const pts = names.flatMap((n) => chart.series[n]);
  if (!pts.length) return "";
  const W = 340, H = 200, l = 40, r = 10, t = 10, b = 30;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (x0 === x1) { x0 -= 1; x1 += 1; }
  const pad = (y1 - y0) * 0.1 || 0.05; y0 = Math.max(0, y0 - pad); y1 = y1 + pad;
  const X = (v) => l + ((v - x0) / (x1 - x0)) * (W - l - r);
  const Y = (v) => H - b - ((v - y0) / (y1 - y0)) * (H - t - b);
  const cols = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(ylabel)} by ${esc(chart.x)}">`;
  for (let k = 0; k <= 4; k++) {
    const v = y0 + ((y1 - y0) * k) / 4, y = Y(v);
    s += `<line class="grid" x1="${l}" x2="${W - r}" y1="${y}" y2="${y}"/><text x="${l - 6}" y="${y + 4}" text-anchor="end">${v.toFixed(2)}</text>`;
  }
  [...new Set(xs)].sort((a, c) => a - c).forEach((v) => {
    s += `<text x="${X(v)}" y="${H - b + 16}" text-anchor="middle">${+v.toFixed(3)}</text>`;
  });
  s += `<line class="axis" x1="${l}" x2="${W - r}" y1="${H - b}" y2="${H - b}"/>`;
  s += `<text x="${(W + l) / 2}" y="${H - 2}" text-anchor="middle">${esc(chart.x)}</text>`;
  names.forEach((n, k) => {
    const p = [...chart.series[n]].sort((a, c) => a[0] - c[0]);
    s += `<polyline fill="none" stroke="${cols[k % 4]}" stroke-width="2" points="${p.map((q) => `${X(q[0])},${Y(q[1])}`).join(" ")}"/>`;
    p.forEach((q) => {
      s += `<circle class="pt" cx="${X(q[0])}" cy="${Y(q[1])}" r="4" fill="${cols[k % 4]}" stroke="var(--surface)" stroke-width="2" data-tip="${esc(n)}: ${esc(chart.x)} ${+q[0].toFixed(3)} → ${q[1].toFixed(3)}"/>`;
    });
  });
  s += "</svg>";
  if (names.length > 1) s += `<div class="chart-legend">${names.map((n, k) => `<span><i style="background:${cols[k % 4]}"></i>${esc(n)}</span>`).join("")}</div>`;
  return s;
}

// ---------------------------------------------------------------- pages
async function pageHome() {
  const { META, METHODS } = await meta();
  const fam = (f) => {
    const c = META.counts[f];
    return `<a class="card" href="#/family/${f}"><h3>${esc(META.families[f].title)}</h3>
      <p>${esc(META.families[f].blurb.split(". ")[0])}.</p>
      <div class="counts">${c.runnable ? `<span>${c.runnable} run here</span>` : ""}${c.cockpit ? `<span>${c.cockpit} in the cockpit</span>` : ""}${c.reference ? `<span>${c.reference} reference</span>` : ""}</div></a>`;
  };
  const [lessons, exps] = await Promise.all([api("/api/lessons"), api("/api/experiments")]);
  render(`
    <h1>Smaller models from bigger ones</h1>
    <p class="lede">${METHODS.length} ways to shrink a trained network or make a small one better, each with the real code,
    a run button where it can run, and the results of ${exps.length} simulation experiments. Pick a model family, or learn to build the algorithms yourself.</p>
    <div class="grid3">${Object.keys(META.families).map(fam).join("")}</div>
    <h2>Build it yourself</h2>
    <p class="muted">${lessons.length} lessons, from softmax to GPTQ. You write the code; the page runs tests against it.</p>
    <ol class="lessons">${lessons.slice(0, 4).map((L) => lessonItem(L)).join("")}</ol>
    <p><a href="#/learn">All lessons</a></p>
    <h2>How to read a method page</h2>
    <div class="legend-row">${pill("runnable")} <span>trains and compresses a real model here; most runs finish in under a minute</span></div>
    <div class="legend-row">${pill("cockpit")} <span>real code for the YOLOv8n cockpit's detector; run it in the cockpit (<code>uvicorn server:app</code>)</span></div>
    <div class="legend-row">${pill("reference")} <span>needs a model this repo doesn't have yet (e.g. a language model); explained with the closest code here</span></div>`, "");
}

async function pageFamily(f) {
  const { META, METHODS } = await meta();
  if (!META.families[f]) return notFound();
  const ms = METHODS.filter((m) => m.family === f);
  const cats = Object.keys(META.categories).filter((c) => ms.some((m) => m.category === c));
  const order = { runnable: 0, cockpit: 1, reference: 2 };
  render(`
    <h1>${esc(META.families[f].title)}</h1>
    <p class="lede">${esc(META.families[f].blurb)}</p>
    ${cats.map((c) => `<h2>${esc(META.categories[c])}</h2><ul class="mlist">${ms.filter((m) => m.category === c)
      .sort((a, b) => order[a.status] - order[b.status])
      .map((m) => `<li><a href="#/method/${m.id}"><span class="nm">${esc(m.name)}</span>${pill(m.status)}<span class="sm">${esc(m.summary)}</span></a></li>`).join("")}</ul>`).join("")}`, f);
}

function paramField(p) {
  const id = `p-${p.name}`;
  const help = p.help ? `<div class="help">${esc(p.help)}</div>` : "";
  if (p.kind === "choice") {
    return `<div class="field"><label for="${id}">${esc(p.label)}</label><select id="${id}" name="${p.name}">${p.choices.map((c) => `<option${c === p.default ? " selected" : ""}>${esc(c)}</option>`).join("")}</select>${help}</div>`;
  }
  if (p.kind === "bool") {
    return `<div class="field"><label class="check"><input type="checkbox" id="${id}" name="${p.name}"${p.default ? " checked" : ""}> ${esc(p.label)}</label>${help}</div>`;
  }
  return `<div class="field"><label for="${id}">${esc(p.label)} <output>${p.default}</output></label>
    <input type="range" id="${id}" name="${p.name}" min="${p.lo}" max="${p.hi}" step="${p.step || (p.kind === "int" ? 1 : 0.01)}" value="${p.default}">${help}</div>`;
}

function resultView(res) {
  const metric = res.metric || "acc";
  const label = metric === "target_ap" ? "bird AP" : "accuracy";
  const max = Math.max(...res.rows.map((r) => r[metric] ?? 0), 1e-9);
  const extra = ["params", "fidelity", "sparsity"].filter((k) => res.rows.some((r) => r[k] !== undefined) && k !== metric);
  const other = metric === "acc" ? "target_ap" : "acc";
  const showOther = res.rows.some((r) => r[other] !== undefined);
  let h = `<table class="restable"><thead><tr><th>model</th><th class="num">${label}</th>${showOther ? `<th class="num">${other === "acc" ? "accuracy" : "bird AP"}</th>` : ""}${extra.map((k) => `<th class="num">${k}</th>`).join("")}</tr></thead><tbody>`;
  for (const r of res.rows) {
    const v = r[metric];
    h += `<tr><td>${esc(r.label)}${v !== undefined ? `<div class="bar"><i style="width:${(100 * v) / max}%"></i></div>` : ""}</td><td class="num">${fmt(v)}</td>${showOther ? `<td class="num">${fmt(r[other])}</td>` : ""}${extra.map((k) => `<td class="num">${fmt(r[k], k)}</td>`).join("")}</tr>`;
  }
  h += "</tbody></table>";
  if (res.chart) h += lineChart(res.chart, label);
  if (res.note) h += `<p class="note">${esc(res.note)}</p>`;
  h += `<div class="status-line">${res.seconds}s · ${Object.entries(res.params).map(([k, v]) => `${k}=${v}`).join(", ") || "defaults"}</div>`;
  return h;
}

async function pageMethod(id) {
  const { META } = await meta();
  let m;
  try { m = await api(`/api/methods/${id}`); } catch { return notFound(); }
  const toc = [];
  const body = m.page ? md(m.page, toc) : `<p class="muted">No write-up yet. The source code below is the reference.</p>`;
  const runBox = m.status === "runnable" ? `
    <div class="box"><h3>Run it</h3><form id="runform">${m.params.map(paramField).join("")}
      <div class="btns"><button class="btn" type="submit">Run</button></div></form>
      <div id="runout" aria-live="polite"><div class="status-line">${STATIC
        ? "Trains and evaluates real models on the toy world, in your browser. The first run downloads Python (about 20 MB); most runs then take 5–60 s."
        : "Trains and evaluates real models on the toy world, starting from the shipped models in lab/models. Most runs take 2–15 s."}</div></div></div>`
    : m.status === "cockpit" ? `<div class="box"><h3>Run it in the cockpit</h3><p class="muted" style="font-size:.9rem">This is the YOLOv8n cockpit's real detector code. It needs torch, ultralytics and torch-pruning, then:</p><pre><code>uvicorn server:app --port 8000</code></pre><p class="muted" style="font-size:.9rem">The closest thing that runs here is linked in the write-up.</p></div>`
    : `<div class="box"><h3>Reference only</h3><p class="muted" style="font-size:.9rem;margin:0">This method needs a kind of model this repo doesn't have. The write-up explains the algorithm and points to the closest code you can run here.</p></div>`;
  const side = `<aside class="panel">${runBox}
    ${toc.length ? `<div class="box toc"><h3>On this page</h3>${toc.map((t) => `<a href="#/method/${id}" data-jump="${t.id}">${esc(t.text)}</a>`).join("")}${m.code.length ? `<a href="#/method/${id}" data-jump="source">Source code</a>` : ""}${m.experiments.length ? `<a href="#/method/${id}" data-jump="simulations">Simulation figures</a>` : ""}</div>` : ""}
    ${m.lessons.length ? `<div class="box"><h3>Build it yourself</h3>${m.lessons.map((L) => `<div><a href="#/learn/${L.id}">${esc(L.title)}</a></div>`).join("")}</div>` : ""}</aside>`;
  render(`
    <div class="crumbs"><a href="#/family/${m.family}">${esc(META.families[m.family].title)}</a> / ${esc(META.categories[m.category])}</div>
    <h1>${esc(m.name)}</h1>
    <p class="lede">${esc(m.summary)}</p>
    <p>${pill(m.status)}</p>
    <div class="cols"><article class="prose">${body}
      ${m.code.length ? `<h2 id="source">Source code</h2><p class="muted">The exact code that runs, read from the repository.</p>${m.code.map((c, k) => `
        <details class="src"${k === 0 ? " open" : ""}><summary>${esc(c.name)} <span>${esc(c.file)}:${c.line}</span></summary><pre><code>${highlight(c.source)}</code></pre></details>`).join("")}` : ""}
      ${m.experiments.length ? `<h2 id="simulations">Simulation figures</h2>${m.experiments.map((e) => `<h3>${esc(e.title)}</h3><p class="muted">${esc(e.question)}</p>${e.figures.map((f) => `<figure><img src="${f}" loading="lazy" alt="${esc(e.title)}"></figure>`).join("")}<p><a href="#/results/${e.name}">Full table</a></p>`).join("")}` : ""}
      ${m.papers.length ? `<h2>Papers</h2><ul>${m.papers.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>` : ""}
    </article>${side}</div>`, m.family);

  app.querySelectorAll("[data-jump]").forEach((a) => a.addEventListener("click", (e) => {
    e.preventDefault(); document.getElementById(a.dataset.jump)?.scrollIntoView({ behavior: "smooth" });
  }));
  const form = $("#runform");
  if (!form) return;
  form.querySelectorAll("input[type=range]").forEach((r) => r.addEventListener("input", () => { r.previousElementSibling.querySelector("output").textContent = r.value; }));
  form.onsubmit = async (e) => {
    e.preventDefault();
    const params = {};
    for (const p of m.params) {
      const el = form.elements[p.name];
      params[p.name] = p.kind === "bool" ? el.checked : p.kind === "choice" ? el.value : Number(el.value);
    }
    const btn = form.querySelector("button"), out = $("#runout");
    btn.disabled = true; btn.textContent = "Running…";
    out.innerHTML = `<div class="status-line">Training… started ${new Date().toLocaleTimeString()}</div>`;
    const onStatus = (ev) => { const sl = out.querySelector(".status-line"); if (sl) sl.textContent = ev.detail; };
    window.addEventListener("pystatus", onStatus);
    try { out.innerHTML = resultView(await api(`/api/methods/${id}/run`, { params })); }
    catch (err) { out.innerHTML = `<div class="err">${esc(err.message)}</div>`; }
    finally { btn.disabled = false; btn.textContent = "Run"; window.removeEventListener("pystatus", onStatus); }
  };
}

const lessonItem = (L) => `<li><a href="#/learn/${L.id}"><span>${esc(L.title)}</span>${store.get("passed:" + L.id) ? '<span class="pill passed">passed</span>' : ""}</a></li>`;

async function pageLearn() {
  const lessons = await api("/api/lessons");
  render(`<h1>Build it yourself</h1>
    <p class="lede">Write each algorithm from scratch in plain numpy. Each lesson explains the idea, gives you a skeleton, and runs hidden tests against your code: numerical gradient checks, comparisons with the lab's own implementation, or a model that has to reach an accuracy.</p>
    <p class="muted">${STATIC
      ? "Your code runs in your own browser (Python compiled to WebAssembly), in a background worker with a 30-second limit. Nothing is sent to a server. Drafts are saved in this browser."
      : "Your code runs on your machine, in a separate Python process with a time limit. Drafts are saved in this browser."}</p>
    <ol class="lessons">${lessons.map(lessonItem).join("")}</ol>`, "learn");
}

async function pageLesson(id) {
  let L;
  try { L = await api(`/api/lessons/${id}`); } catch { return notFound(); }
  const key = "draft:" + id;
  render(`<div class="crumbs"><a href="#/learn">Build it yourself</a></div>
    <h1>${esc(L.title)}</h1>
    <div class="prose">${md(L.intro)}</div>
    <h2>Your code</h2>
    <textarea class="editor" id="code" spellcheck="false" autocapitalize="off" autocomplete="off" aria-label="Your code"></textarea>
    <div class="btns" style="margin-top:.7rem">
      <button class="btn" id="check" type="button">Check my code</button>
      <button class="btn secondary" id="hint" type="button">Hint</button>
      <button class="btn secondary" id="solution" type="button">Show solution</button>
      <button class="btn secondary" id="reset" type="button">Reset</button>
      <span class="muted" style="font-size:.82rem">Ctrl/⌘ + Enter to check</span>
    </div>
    <div id="hints" style="margin-top:1rem"></div>
    <div id="results" aria-live="polite"></div>
    ${L.methods.length ? `<h2>Where this is used</h2><ul>${L.methods.map((m) => `<li><a href="#/method/${m.id}">${esc(m.name)}</a> ${pill(m.status)}</li>`).join("")}</ul>` : ""}
    <div class="pager">${L.prev ? `<a href="#/learn/${L.prev}">Previous lesson</a>` : "<span></span>"}${L.next ? `<a href="#/learn/${L.next}">Next lesson</a>` : ""}</div>`, "learn");
  const ta = $("#code");
  ta.value = store.get(key, L.skeleton);
  ta.addEventListener("input", () => store.set(key, ta.value));
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Tab" && !e.shiftKey) {
      e.preventDefault();
      const s = ta.selectionStart;
      ta.setRangeText("    ", s, ta.selectionEnd, "end");
      store.set(key, ta.value);
    } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $("#check").click(); }
  });
  let shown = 0;
  $("#hint").onclick = () => {
    if (shown < L.hints.length) $("#hints").insertAdjacentHTML("beforeend", `<div class="hint">${inline(L.hints[shown++])}</div>`);
    if (shown >= L.hints.length) $("#hint").disabled = true;
  };
  $("#reset").onclick = () => { if (confirm("Replace your code with the starting skeleton?")) { ta.value = L.skeleton; store.set(key, ta.value); } };
  $("#solution").onclick = async () => {
    if (!confirm("Replace your code with the reference solution? Your draft will be overwritten.")) return;
    ta.value = (await api(`/api/lessons/${id}/solution`)).solution; store.set(key, ta.value);
  };
  $("#check").onclick = async () => {
    const btn = $("#check"), out = $("#results");
    btn.disabled = true; btn.textContent = "Checking…";
    const onStatus = (ev) => { out.innerHTML = `<div class="status-line">${esc(ev.detail)}</div>`; };
    window.addEventListener("pystatus", onStatus);
    try {
      const r = await api(`/api/lessons/${id}/check`, { code: ta.value });
      if (r.passed) store.set("passed:" + id, "1");
      out.innerHTML = (r.error ? `<div class="err">${esc(r.error)}</div>` : "") +
        (r.tests.length ? `<ul class="tests">${r.tests.map((t) => `<li class="${t.ok ? "ok" : "no"}"><b>${t.ok ? "Pass" : "Fail"}</b> · ${esc(t.doc || t.name)}${t.ok ? "" : `<span class="msg">${esc(t.message)}${t.trace ? "\n" + esc(t.trace) : ""}</span>`}</li>`).join("")}</ul>` : "") +
        (r.passed ? `<p><strong>All tests pass.</strong>${L.next ? ` <a href="#/learn/${L.next}">Next lesson</a>` : ""}</p>` : "");
    } catch (err) { out.innerHTML = `<div class="err">${esc(err.message)}</div>`; }
    finally { btn.disabled = false; btn.textContent = "Check my code"; window.removeEventListener("pystatus", onStatus); }
  };
}

async function pageResults() {
  const [{ text }, exps] = await Promise.all([api("/api/report"), api("/api/experiments")]);
  const groups = [...new Set(exps.map((e) => e.group))];
  render(`<h1>Simulation results</h1>
    <p class="lede">${exps.length} experiments, each run with several seeds on the toy world. The write-up comes first; every experiment's figures and full table follow.</p>
    ${text ? `<div class="prose">${md(text)}</div>` : `<p class="muted">No REPORT.md yet.</p>`}
    <h2>All experiments</h2>
    ${groups.map((g) => `<h3>${esc(g)}</h3><ul class="mlist">${exps.filter((e) => e.group === g).map((e) => `<li><a href="#/results/${e.name}"><span class="nm">${esc(e.title)}</span><span class="muted" style="font-size:.82rem">${e.figures.length} figure${e.figures.length === 1 ? "" : "s"}</span><span class="sm">${esc(e.question)}</span></a></li>`).join("")}</ul>`).join("")}`, "results");
}

async function pageExperiment(name) {
  let e;
  try { e = await api(`/api/experiments/${name}`); } catch { return notFound(); }
  const { METHODS } = await meta();
  const related = METHODS.filter((m) => e.methods.includes(m.id));
  const keys = [...new Set(e.rows.flatMap((r) => Object.keys(r)))].filter((k) => !k.endsWith("_std") && !["n_seeds", "series", "arm", "macs", "ece"].includes(k));
  render(`<div class="crumbs"><a href="#/results">Results</a> / ${esc(e.group)}</div>
    <h1>${esc(e.title)}</h1><p class="lede">${esc(e.question)}</p>
    ${related.length ? `<p>Methods: ${related.map((m) => `<a href="#/method/${m.id}">${esc(m.name)}</a>`).join(" · ")}</p>` : ""}
    ${e.figures.map((f) => `<figure><img src="${f}" alt="${esc(e.title)}"></figure>`).join("")}
    ${e.notes ? `<p class="note">${esc(e.notes)}</p>` : ""}
    <h2>Table</h2><p class="muted">Mean over seeds; ± is the standard deviation where more than one seed ran. Took ${Math.round(e.seconds)} s.</p>
    <div style="overflow-x:auto"><table class="restable"><thead><tr>${keys.map((k) => `<th>${esc(k)}</th>`).join("")}</tr></thead><tbody>
    ${e.rows.map((r) => `<tr>${keys.map((k) => `<td class="${typeof r[k] === "number" ? "num" : ""}">${fmt(r[k], k)}${r[k + "_std"] !== undefined ? ` <span class="muted">± ${r[k + "_std"].toFixed(3)}</span>` : ""}</td>`).join("")}</tr>`).join("")}
    </tbody></table></div>`, "results");
}

function notFound() {
  render(`<h1>Not found</h1><p><a href="#/">Back to the start</a></p>`, "");
}

// ---------------------------------------------------------------- router
async function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  try {
    if (!parts.length) await pageHome();
    else if (parts[0] === "family") await pageFamily(parts[1]);
    else if (parts[0] === "method") await pageMethod(parts[1]);
    else if (parts[0] === "learn" && parts[1]) await pageLesson(parts[1]);
    else if (parts[0] === "learn") await pageLearn();
    else if (parts[0] === "results" && parts[1]) await pageExperiment(parts[1]);
    else if (parts[0] === "results") await pageResults();
    else notFound();
    app.focus({ preventScroll: true });
  } catch (err) {
    app.innerHTML = `<div class="err">Couldn't load this page: ${esc(err.message)}.${STATIC ? "" : " Is the server running (python -m lab.web)?"}</div>`;
  }
}
window.addEventListener("hashchange", route);
route();
