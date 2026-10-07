"""The static (GitHub Pages) build: every page's data is pre-rendered, and the
Python bundle the browser unpacks is complete enough to run methods and check
lessons on its own (simulated here with a plain Python subprocess instead of
Pyodide, which isn't available in CI)."""

import json
import os
import subprocess
import sys
import zipfile

from lab.lessons import LESSONS
from lab.methods import REGISTRY
from lab.web import build_static as bs


def test_static_build(tmp_path):
    out = tmp_path / "site"
    out.mkdir()
    models = bs.collect_models()
    bs.export_bundle(str(out), models)
    bs.export_data(str(out / "data"), str(out / "figures"))
    bs.export_frontend(str(out))

    html = (out / "index.html").read_text()
    assert "window.LAB_STATIC" in html and "/static/" not in html
    for f in ("app.js", "style.css", "worker.js", ".nojekyll"):
        assert (out / f).exists(), f
    for mid in REGISTRY:
        d = json.loads((out / "data" / "methods" / f"{mid}.json").read_text())
        assert d["page"] and all(c["source"] for c in d["code"])
        for e in d["experiments"]:
            for fig in e["figures"]:
                assert fig.startswith("figures/") and (out / fig).exists(), fig
    for L in LESSONS:
        assert (out / "data" / "lessons" / L.id / "solution.json").exists()
    assert (out / "data" / "report.json").exists()

    # the Astro site's mode: page data elsewhere, runtime assets only
    rt, data = tmp_path / "public-lab", tmp_path / "src-data"
    rt.mkdir()
    bs.export_data(str(data), str(rt / "figures"))
    bs.export_worker(str(rt))
    assert (data / "meta.json").exists() and (data / "methods" / "ptq.json").exists()
    assert (rt / "worker.js").exists() and (rt / "figures").is_dir()
    assert not (rt / "index.html").exists() and not (rt / "app.js").exists()

    # unpack the bundle somewhere else and use it with nothing but numpy
    root = tmp_path / "browser"
    with zipfile.ZipFile(out / "lab_bundle.zip") as z:
        z.extractall(root)
    assert not (root / "lab" / "web").exists(), "the bundle should not ship the server"
    script = """
import json, sys
sys.path.insert(0, sys.argv[1])
import lab, lab.common
assert lab.common.__file__.startswith(sys.argv[1]), lab.common.__file__
from lab.methods import REGISTRY
from lab.lessons import LESSONS, check_inprocess
r = REGISTRY["ptq"].execute({"bits": 4, "model": "student"})
assert r["rows"], r
L = LESSONS[0]
assert check_inprocess(L.id, L.solution)["passed"]
assert not check_inprocess(L.id, L.skeleton)["passed"]
print("ok")
"""
    env = {k: v for k, v in os.environ.items() if k != "LAB_QUICK"}
    p = subprocess.run([sys.executable, "-I", "-c", script, str(root)], capture_output=True,
                       text=True, env=env, cwd=str(tmp_path), timeout=600)
    assert p.stdout.strip().endswith("ok"), p.stderr[-2000:]
