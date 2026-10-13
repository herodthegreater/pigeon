"""Build the static version of the lab website (for GitHub Pages).

    python -m lab.web.build_static --out _site
    python -m lab.web.build_static --out _site --pyodide-dir /path/to/pyodide   # offline

    # for the Astro site: runtime assets into public/lab, page data where Astro
    # imports it at build time (no single-page app)
    python -m lab.web.build_static --out site/public/lab --data-out site/src/data/lab --no-frontend
    # the same, skipping the slow parts (no bundle, no Pyodide download): enough
    # to build and browse every page locally; Run and Check need the full build
    python -m lab.web.build_static --out site/public/lab --data-out site/src/data/lab --no-frontend --data-only
    # retrain-or-load the shipped models and write them to lab/models (only when you
    # mean to change what the site ships; then re-measure every quoted page run)
    python -m lab.web.build_static --save-models

Everything the server computes on request is either pre-rendered here (method
write-ups, source code, results, figures, lessons) or shipped to the browser
so it can be computed there: the lab's Python files and its pre-trained models
go into ``lab_bundle.zip``, which ``worker.js`` unpacks into Pyodide (CPython
compiled to WebAssembly). Runs and lesson checks then happen on the visitor's
machine, inside a Web Worker, with no server at all.

The models are the committed files in ``lab/models``, packed byte for byte;
this build never trains them. Training the same model on another machine (CI's
Linux, a different BLAS) gives slightly different weights, and every "page run"
number a write-up quotes would drift with them. ``lab/tests/test_quotes.py``
re-runs a sample of those quotes against ``lab/models``.

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
    path = os.path.join(out, rel + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    text = text.replace('"/figures/', '"figures/')  # relative: Pages serves under /<repo>/
    with open(path, "w") as f:
        f.write(text)


def export_data(out: str, figures_out: str) -> None:
    """Pre-render every GET the front end makes (same functions as the server).
    JSON goes to `out`; figures to `figures_out`."""
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
    shutil.copytree(os.path.join(RESULTS, "figures"), figures_out,
                    ignore=shutil.ignore_patterns("quick_*"), dirs_exist_ok=True)


def collect_models(shipped_only: bool = False) -> dict[str, object]:
    """Load (or train) every model the runnable methods use, recording the
    cache key each one is stored under. ``shipped_only``: every one must
    already be in lab/models; missing ones stop the build instead of training."""
    seen: dict[str, object] = {}
    orig = common.cached

    def recording(key, build):
        if shipped_only and not os.path.exists(common.shipped_path(key)):
            raise SystemExit(f"lab/models/{key}.npz is missing. The site ships only committed "
                             "models: run `python -m lab.web.build_static --save-models`, commit "
                             "lab/models, and re-measure the write-ups' page runs.")
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


def npz_bytes(obj) -> bytes:
    if isinstance(obj, bytes):
        return obj
    buf = io.BytesIO()
    if isinstance(obj, MLP):
        obj.save_npz(buf)
    else:
        np.savez(buf, array=np.asarray(obj))
    return buf.getvalue()


def shipped_models() -> dict[str, bytes]:
    """The committed lab/models files the bundle carries, read as they are."""
    keys = collect_models(shipped_only=True)
    out = {}
    for key in keys:
        with open(common.shipped_path(key), "rb") as f:
            out[key] = f.read()
    return out


def save_models() -> None:
    """Write every model the runnable methods use to lab/models: what
    lab/.cache has wins, then what lab/models already has; the rest is
    trained."""
    if common.QUICK:
        raise SystemExit("unset LAB_QUICK: lab/models holds full-budget models")
    os.makedirs(common.MODELS, exist_ok=True)
    for key, obj in collect_models().items():
        with open(common.shipped_path(key), "wb") as f:
            f.write(npz_bytes(obj))
        print(f"  lab/models/{key}.npz", flush=True)


def export_bundle(out: str, models: dict[str, object]) -> None:
    """lab_bundle.zip: the Python the browser needs, plus the models as .npz
    under lab/models, where ``common.cached`` finds them (the browser has no
    lab/.cache)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in BUNDLE_MODULES:
            z.write(os.path.join(LAB, name), f"lab/{name}")
        for name in sorted(os.listdir(os.path.join(LAB, "experiments"))):
            if name.endswith(".py"):
                z.write(os.path.join(LAB, "experiments", name), f"lab/experiments/{name}")
        for key, obj in sorted(models.items()):
            z.writestr(f"lab/models/{key}.npz", npz_bytes(obj))
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


def export_worker(out: str) -> None:
    shutil.copy(os.path.join(HERE, "static", "worker.js"), out)
    open(os.path.join(out, ".nojekyll"), "w").close()  # serve files as-is


def export_frontend(out: str) -> None:
    static = os.path.join(HERE, "static")
    for name in ("app.js", "style.css"):
        shutil.copy(os.path.join(static, name), out)
    shutil.copytree(os.path.join(static, "vendor"), os.path.join(out, "vendor"), dirs_exist_ok=True)
    export_worker(out)
    html = open(os.path.join(static, "index.html")).read()
    html = html.replace('href="/static/style.css"', 'href="style.css"')
    html = html.replace('"/static/vendor/', '"vendor/')
    html = html.replace('<script src="/static/app.js"></script>',
                        '<script>window.LAB_STATIC = true;</script>\n<script src="app.js"></script>')
    assert "window.LAB_STATIC" in html and "/static/" not in html
    with open(os.path.join(out, "index.html"), "w") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="_site")
    ap.add_argument("--pyodide-dir", help="a directory with an extracted Pyodide release "
                                          "(default: download the release tarball)")
    ap.add_argument("--data-out", help="write the page data (JSON) here instead of <out>/data")
    ap.add_argument("--no-frontend", action="store_true",
                    help="skip the single-page app (index.html, app.js, style.css); ship only "
                         "what the Astro site needs at runtime: worker.js, Pyodide, the bundle, figures")
    ap.add_argument("--data-only", action="store_true",
                    help="write page data and figures only: no bundle, no Pyodide")
    ap.add_argument("--save-models", action="store_true",
                    help="write the models the site ships to lab/models (training any that "
                         "neither lab/models nor lab/.cache has), then stop")
    args = ap.parse_args()
    if common.QUICK:
        raise SystemExit("unset LAB_QUICK: the site ships full-budget models")
    if args.save_models:
        save_models()
        return
    out = os.path.abspath(args.out)
    data_out = os.path.abspath(args.data_out) if args.data_out else os.path.join(out, "data")
    # --data-only refreshes data and figures but keeps an existing runtime (Pyodide,
    # the bundle) in `out`, so a quick local rebuild never deletes a full one.
    fresh = [data_out, os.path.join(out, "figures")] if args.data_only else [out, data_out]
    for d in sorted(set(fresh)):  # parent first if data_out sits inside out
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
    if not args.data_only:
        models = shipped_models()
        print(f"packing {len(models)} models from lab/models: {', '.join(models)}", flush=True)
        export_bundle(out, models)
    export_data(data_out, os.path.join(out, "figures"))
    if args.no_frontend:
        export_worker(out)
    else:
        export_frontend(out)
    if not args.data_only:
        export_pyodide(out, args.pyodide_dir)
    for d in sorted({out, data_out}):
        total = sum(os.path.getsize(os.path.join(p, f)) for p, _, fs in os.walk(d) for f in fs)
        print(f"built {d} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
