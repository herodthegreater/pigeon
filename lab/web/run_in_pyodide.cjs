// Run lab methods in Pyodide, the runtime the website's Run button uses, from Node.
//
//   python -m lab.web.build_static --out site/public/lab --data-out site/src/data/lab --no-frontend
//   echo '[["ptq", {}], ["qat", {"bits": 2}]]' > runs.json
//   node lab/web/run_in_pyodide.cjs site/public/lab runs.json > runs.jsonl
//
// Prints one JSON line per run: {mid, params, rows, chart, secs}. These are the
// numbers a visitor's browser gets (same Pyodide, same numpy wheel, same
// lab_bundle.zip). CPython agreed to the digit on every run that only rounds
// or prunes a shipped model's weights; a run that trains, or rounds
// activations, can land a few test examples away on another BLAS, so quote a
// page's Run from here.
"use strict";
const fs = require("fs");
const os = require("os");
const path = require("path");

const [labDir, runsFile] = process.argv.slice(2);
if (!labDir || !runsFile) {
  console.error("usage: node lab/web/run_in_pyodide.cjs <site/public/lab> <runs.json>");
  process.exit(2);
}
const runs = JSON.parse(fs.readFileSync(runsFile, "utf8"));

// site/package.json says "type": "module", which makes Node load the release's
// pyodide.js as an ES module and break it; a copy outside site/ loads as CommonJS.
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "lab-pyodide-"));
fs.cpSync(path.join(labDir, "pyodide"), path.join(tmp, "pyodide"), { recursive: true });

(async () => {
  require(path.join(tmp, "pyodide", "pyodide.js"));
  const py = await globalThis.loadPyodide({ indexURL: path.join(tmp, "pyodide") + path.sep });
  await py.loadPackage("numpy", { messageCallback: () => {} });
  const zip = fs.readFileSync(path.join(labDir, "lab_bundle.zip"));
  py.unpackArchive(new Uint8Array(zip.buffer, zip.byteOffset, zip.byteLength).slice(), "zip",
                   { extractDir: "/home/pyodide/root" });
  py.runPython(`
import json, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "/home/pyodide/root")
from lab.methods import REGISTRY

def _run(mid, params_json):
    m = REGISTRY[mid]
    params = m.coerce(json.loads(params_json))
    return json.dumps(dict(m.run(params), params=params),
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))
`);
  for (const [mid, params] of runs) {
    const t0 = Date.now();
    const out = JSON.parse(py.globals.get("_run")(mid, JSON.stringify(params || {})));
    const line = { mid, params: out.params, rows: out.rows, chart: out.chart, secs: (Date.now() - t0) / 1000 };
    process.stdout.write(JSON.stringify(line) + "\n");
  }
})()
  .catch((err) => { console.error(err); process.exitCode = 1; })
  .finally(() => fs.rmSync(tmp, { recursive: true, force: true }));
