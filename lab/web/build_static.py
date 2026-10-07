"""Build the static version of the lab website (for GitHub Pages).

    python -m lab.web.build_static --out _site
    python -m lab.web.build_static --out _site --pyodide-dir /path/to/pyodide   # offline

Everything the server computes on request is either pre-rendered here (method
write-ups, source code, results, figures, lessons) or shipped to the browser
so it can be computed there: the lab's Python files and its pre-trained models
go into ``lab_bundle.zip``, which ``worker.js`` unpacks into Pyodide (CPython
compiled to WebAssembly). Runs and lesson checks then happen on the visitor's
machine, inside a Web Worker, with no server at all.

Pyodide itself is self-hosted from the official release tarball (only the
runtime and the numpy wheel, ~20 MB), so the site doesn't depend on any CDN.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import tarfile
import urllib.request
import zipfile

import numpy as np

from .. import common, methods
from ..common import LAB, RESULTS
from ..nn import MLP
from . import app as webapp

PYODIDE_VERSION = "0.27.7"
PYODIDE_URL = (f"https://github.com/pyodide/pyodide/releases/download/{PYODIDE_VERSION}/"
               f"pyodide-{PYODIDE_VERSION}.tar.bz2")
PYODIDE_FILES = ("pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip",
                 "pyodide-lock.json")
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE_MODULES = ("__init__.py", "common.py", "nn.py", "train.py", "world.py", "compress.py",
                  "layerwise.py", "methods.py", "lessons.py")


def _write_json(out: str, rel: str, obj) -> None:
    path = os.path.join(out, "data", rel + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    text = text.replace('"/figures/', '"figures/')  # relative: Pages serves under /<repo>/
    with open(path, "w") as f:
        f.write(text)


def export_data(out: str) -> None:
    """Pre-render every GET the front end makes (same functions as the server)."""
    _write_json(out, "meta", webapp.meta())
    _write_json(out, "methods", webapp.methods())
    for mid in methods.REGISTRY:
        _write_json(out, f"methods/{mid}", webapp.method(mid))
    _write_json(out, "lessons", webapp.lessons())
    for L in webapp.LESSONS:
        _write_json(out, f"lessons/{L.id}", webapp.lesson(L.id))
        _write_json(out, f"lessons/{L.id}/solution", webapp.lesson_solution(L.id))
    exps = webapp.experiments()
    _write_json(out, "experiments", exps)
    for e in exps:
        _write_json(out, f"experiments/{e['name']}", webapp.experiment(e["name"]))
    _write_json(out, "report", webapp.report())
    shutil.copytree(os.path.join(RESULTS, "figures"), os.path.join(out, "figures"),
                    ignore=shutil.ignore_patterns("quick_*"), dirs_exist_ok=True)


def collect_models() -> dict[str, object]:
    """Train (or load) every cached model the runnable methods use, recording
    the cache key each one is stored under."""
    seen: dict[str, object] = {}
    orig = common.cached

    def recording(key, build):
        obj = orig(key, build)
        seen[key] = obj
        return obj

    common.cached = methods.cached = recording
    try:
        t = common.teacher()
        for seed in range(1, 4):                     # ensemble_kd: up to 4 teachers
            common.teacher(seed=seed)
        for res in (False, True):                    # layer_dropping, block_influence
            common.teacher(hidden=[128] * 8, res=res)
        methods.kd_student()                         # ptq, qat, qlora
        from ..experiments.transfer import invert_teacher
        recording("inversion_4000", lambda: invert_teacher(t, 4000, steps=200))  # data_free_kd default
    finally:
        common.cached = methods.cached = orig
    return seen


def export_bundle(out: str, models: dict[str, object]) -> None:
    """lab_bundle.zip: the Python the browser needs, plus the models as .npz."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in BUNDLE_MODULES:
            z.write(os.path.join(LAB, name), f"lab/{name}")
        for name in sorted(os.listdir(os.path.join(LAB, "experiments"))):
            if name.endswith(".py"):
                z.write(os.path.join(LAB, "experiments", name), f"lab/experiments/{name}")
        for key, obj in models.items():
            mbuf = io.BytesIO()
            if isinstance(obj, MLP):
                obj.save_npz(mbuf)
            else:
                np.savez(mbuf, array=np.asarray(obj))
            z.writestr(f"lab/.cache/{key}.npz", mbuf.getvalue())
    with open(os.path.join(out, "lab_bundle.zip"), "wb") as f:
        f.write(buf.getvalue())


def export_pyodide(out: str, pyodide_dir: str | None) -> None:
    """Copy the Pyodide runtime + numpy wheel into <out>/pyodide/."""
    dest = os.path.join(out, "pyodide")
    os.makedirs(dest, exist_ok=True)

    def wanted(name: str) -> bool:
        base = os.path.basename(name)
        return base in PYODIDE_FILES or (base.startswith("numpy-") and base.endswith(".whl"))

    if pyodide_dir:
        for base in os.listdir(pyodide_dir):
            if wanted(base):
                shutil.copy(os.path.join(pyodide_dir, base), dest)
    else:
        print(f"downloading {PYODIDE_URL} ...", flush=True)
        with urllib.request.urlopen(PYODIDE_URL) as resp, \
                tarfile.open(fileobj=resp, mode="r|bz2") as tar:
            for member in tar:
                if member.isfile() and wanted(member.name):
                    with open(os.path.join(dest, os.path.basename(member.name)), "wb") as f:
                        f.write(tar.extractfile(member).read())
    got = sorted(os.listdir(dest))
    missing = [f for f in PYODIDE_FILES if f not in got]
    if missing or not any(g.startswith("numpy-") for g in got):
        raise SystemExit(f"pyodide files missing: {missing or 'numpy wheel'} (have {got})")


def export_frontend(out: str) -> None:
    static = os.path.join(HERE, "static")
    for name in ("app.js", "style.css", "worker.js"):
        shutil.copy(os.path.join(static, name), out)
    html = open(os.path.join(static, "index.html")).read()
    html = html.replace('href="/static/style.css"', 'href="style.css"')
    html = html.replace('<script src="/static/app.js"></script>',
                        '<script>window.LAB_STATIC = true;</script>\n<script src="app.js"></script>')
    assert "window.LAB_STATIC" in html and "/static/" not in html
    with open(os.path.join(out, "index.html"), "w") as f:
        f.write(html)
    open(os.path.join(out, ".nojekyll"), "w").close()  # serve files as-is


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="_site")
    ap.add_argument("--pyodide-dir", help="a directory with an extracted Pyodide release "
                                          "(default: download the release tarball)")
    args = ap.parse_args()
    if common.QUICK:
        raise SystemExit("unset LAB_QUICK: the site ships full-budget models")
    out = os.path.abspath(args.out)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    print("training / loading models ...", flush=True)
    models = collect_models()
    print(f"  {len(models)} models: {', '.join(models)}", flush=True)
    export_bundle(out, models)
    export_data(out)
    export_frontend(out)
    export_pyodide(out, args.pyodide_dir)
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs)
    print(f"built {out} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
