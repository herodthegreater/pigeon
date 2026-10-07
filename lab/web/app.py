"""The lab website: methods by model family, runnable demos, the real source
code of every algorithm, build-it-yourself lessons, and the simulation results.

    python -m lab.web            # http://127.0.0.1:8100

Listens on localhost only: the lessons' "check my code" runs the code you
type, on your machine.
"""

from __future__ import annotations

import glob
import json
import os
import threading
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..common import RESULTS
from ..lessons import LESSONS, check, get as get_lesson
from ..methods import CATEGORIES, FAMILIES, REGISTRY
from .source import get_source

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
CONTENT = os.path.join(HERE, "content", "methods")
STATIC = os.path.join(HERE, "static")
FIGS = os.path.join(RESULTS, "figures")

app = FastAPI(title="pigeon lab")
_run_lock = threading.Lock()   # numpy runs are CPU-bound; one at a time keeps them fast


def _page(mid: str) -> str:
    path = os.path.join(CONTENT, f"{mid}.md")
    return open(path).read() if os.path.exists(path) else ""


def _experiment(name: str) -> dict | None:
    path = os.path.join(RESULTS, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        res = json.load(f)
    figs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(FIGS, f"{name}_*.png")))
    return dict(name=name, title=res["title"], question=res["question"], group=res["group"],
                seconds=res.get("seconds"), figures=[f"/figures/{f}" for f in figs],
                rows=res["rows"], refs=res.get("refs", {}), notes=res.get("notes", ""))


def _summary(m) -> dict:
    return dict(id=m.id, name=m.name, family=m.family, category=m.category, status=m.status,
                summary=m.summary)


@app.get("/api/meta")
def meta():
    return dict(families=FAMILIES, categories=CATEGORIES,
                counts={f: {s: sum(1 for m in REGISTRY.values() if m.family == f and m.status == s)
                            for s in ("runnable", "cockpit", "reference")} for f in FAMILIES})


@app.get("/api/methods")
def methods():
    return [_summary(m) for m in REGISTRY.values()]


@app.get("/api/methods/{mid}")
def method(mid: str):
    m = REGISTRY.get(mid)
    if m is None:
        raise HTTPException(404, f"no method {mid!r}")
    exps = [e for e in (_experiment(n) for n in m.experiments) if e]
    for e in exps:
        e.pop("rows"); e.pop("refs")
    return dict(**_summary(m), page=_page(mid), papers=m.papers,
                params=[vars(p) for p in m.params],
                code=[get_source(ref) for ref in m.code], experiments=exps,
                lessons=[dict(id=L.id, title=L.title) for L in LESSONS if mid in L.methods])


class RunRequest(BaseModel):
    params: dict = {}


@app.post("/api/methods/{mid}/run")
def run(mid: str, req: RunRequest):
    m = REGISTRY.get(mid)
    if m is None:
        raise HTTPException(404, f"no method {mid!r}")
    if m.run is None:
        raise HTTPException(400, f"{m.name} doesn't run here (status: {m.status})")
    try:
        params = m.coerce(req.params)
    except ValueError as e:
        raise HTTPException(422, str(e))
    with _run_lock:
        t0 = time.time()
        out = m.run(params)
    return dict(**out, params=params, seconds=round(time.time() - t0, 2))


@app.get("/api/lessons")
def lessons():
    return [dict(id=L.id, title=L.title, methods=L.methods) for L in LESSONS]


@app.get("/api/lessons/{lid}")
def lesson(lid: str):
    try:
        L = get_lesson(lid)
    except KeyError:
        raise HTTPException(404, f"no lesson {lid!r}")
    i = LESSONS.index(L)
    return dict(id=L.id, title=L.title, intro=L.intro, skeleton=L.skeleton, hints=L.hints,
                methods=[_summary(REGISTRY[m]) for m in L.methods],
                prev=LESSONS[i - 1].id if i > 0 else None,
                next=LESSONS[i + 1].id if i + 1 < len(LESSONS) else None)


@app.get("/api/lessons/{lid}/solution")
def lesson_solution(lid: str):
    try:
        return dict(solution=get_lesson(lid).solution)
    except KeyError:
        raise HTTPException(404, f"no lesson {lid!r}")


class CheckRequest(BaseModel):
    code: str


@app.post("/api/lessons/{lid}/check")
def lesson_check(lid: str, req: CheckRequest):
    try:
        get_lesson(lid)
    except KeyError:
        raise HTTPException(404, f"no lesson {lid!r}")
    if len(req.code) > 100_000:
        raise HTTPException(413, "code too long")
    return check(lid, req.code)


@app.get("/api/experiments")
def experiments():
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        name = os.path.basename(p)[:-5]
        if name.startswith("quick_"):
            continue
        e = _experiment(name)
        e.pop("rows"); e.pop("refs")
        e["methods"] = [m.id for m in REGISTRY.values() if name in m.experiments]
        out.append(e)
    return sorted(out, key=lambda e: (e["group"], e["name"]))


@app.get("/api/experiments/{name}")
def experiment(name: str):
    if "/" in name or name.startswith("."):
        raise HTTPException(404)
    e = _experiment(name)
    if e is None:
        raise HTTPException(404, f"no results for {name!r}; run python -m lab.run {name}")
    e["methods"] = [m.id for m in REGISTRY.values() if name in m.experiments]
    return e


@app.get("/api/report")
def report():
    path = os.path.join(LAB, "REPORT.md")
    return dict(text=open(path).read() if os.path.exists(path) else "")


os.makedirs(FIGS, exist_ok=True)
app.mount("/figures", StaticFiles(directory=FIGS), name="figures")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))
