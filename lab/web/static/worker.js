// Runs the lab's Python inside the browser (static / GitHub Pages build).
// Pyodide is CPython compiled to WebAssembly; numpy comes with it. The lab's
// own .py files and pre-trained models arrive in lab_bundle.zip. Living in a
// Web Worker keeps the page responsive while a model trains, and lets the
// page kill a lesson check that runs forever.
"use strict";

let ready = null;
const say = (text) => postMessage({ type: "status", text });

function init() {
  if (!ready) {
    ready = (async () => {
      say("Downloading Python (about 20 MB, only the first time)…");
      importScripts("pyodide/pyodide.js");
      const py = await loadPyodide({ indexURL: new URL("pyodide/", self.location).href });
      say("Loading numpy…");
      await py.loadPackage("numpy");
      say("Loading the lab code and trained models…");
      const res = await fetch("lab_bundle.zip");
      if (!res.ok) throw new Error(`couldn't download the lab code (HTTP ${res.status})`);
      const buf = await res.arrayBuffer();
      py.unpackArchive(buf, "zip", { extractDir: "/home/pyodide/root" });
      py.runPython(`
import json, sys
sys.path.insert(0, "/home/pyodide/root")
from lab.methods import REGISTRY
from lab.lessons import check_inprocess

def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)

def _run(mid, params_json):
    m = REGISTRY[mid]
    params = m.coerce(json.loads(params_json))
    out = m.run(params)
    return json.dumps(dict(out, params=params), default=_jsonable)

def _check(lid, code):
    return json.dumps(check_inprocess(lid, code), default=_jsonable)
`);
      say("Python is ready.");
      return py;
    })();
    // A failed start (a dropped download, say) must not stick: forget it so the
    // next request tries again instead of failing until the page is reloaded.
    ready.catch(() => { ready = null; });
  }
  return ready;
}

const lastLine = (msg) => {
  const lines = String(msg).trim().split("\n").filter(Boolean);
  return lines[lines.length - 1] || String(msg);
};

onmessage = async (e) => {
  const { id, kind, args } = e.data;
  try {
    const py = await init();
    let out = "null";
    if (kind === "warm") out = "null";  // just make sure Python is loaded
    else if (kind === "run") out = py.globals.get("_run")(args.mid, JSON.stringify(args.params || {}));
    else if (kind === "check") out = py.globals.get("_check")(args.lid, args.code);
    postMessage({ id, ok: true, result: JSON.parse(out) });
  } catch (err) {
    postMessage({ id, ok: false, error: lastLine(err && err.message ? err.message : err) });
  }
};
