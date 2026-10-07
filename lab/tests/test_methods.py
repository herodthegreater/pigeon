"""Every method in the registry: metadata is complete, its code pointers resolve
to real source, its page exists, and (if runnable) it runs with default and
extreme parameters and returns finite metrics."""

from __future__ import annotations

import math
import os

import pytest

from lab.methods import CATEGORIES, FAMILIES, REGISTRY
from lab.web.source import get_source

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNABLE = [m for m in REGISTRY.values() if m.status == "runnable"]


def _check_result(res):
    assert set(res) >= {"rows", "note", "chart", "metric"}
    assert res["rows"], "no rows"
    for r in res["rows"]:
        assert isinstance(r["label"], str) and r["label"]
        assert res["metric"] in r or "acc" in r, r
        for k, v in r.items():
            if isinstance(v, float):
                assert math.isfinite(v), (k, v)
    if res["chart"]:
        for pts in res["chart"]["series"].values():
            assert all(len(p) == 2 and all(math.isfinite(float(x)) for x in p) for p in pts)


@pytest.mark.parametrize("m", REGISTRY.values(), ids=lambda m: m.id)
def test_metadata(m):
    assert m.family in FAMILIES
    assert m.category in CATEGORIES
    assert m.status in ("runnable", "cockpit", "reference")
    assert (m.run is not None) == (m.status == "runnable")
    assert m.summary.endswith(".")
    names = [p.name for p in m.params]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("m", REGISTRY.values(), ids=lambda m: m.id)
def test_code_pointers_resolve(m):
    for ref in m.code:
        src = get_source(ref)
        assert src["source"].strip(), ref
        assert src["line"] >= 1


@pytest.mark.parametrize("m", REGISTRY.values(), ids=lambda m: m.id)
def test_page_exists(m):
    path = os.path.join(ROOT, "lab", "web", "content", "methods", f"{m.id}.md")
    assert os.path.exists(path), f"missing page {path}"
    text = open(path).read()
    for h in ("## The idea", "## How it works", "## The code, line by line", "## When it breaks"):
        assert h in text, f"{m.id}: missing section {h!r}"


@pytest.mark.parametrize("m", RUNNABLE, ids=lambda m: m.id)
def test_runs_with_defaults(m):
    _check_result(m.execute({}))


@pytest.mark.parametrize("m", RUNNABLE, ids=lambda m: m.id)
def test_runs_at_parameter_extremes(m):
    """Every choice value and both ends of every numeric range must work."""
    variants = []
    for p in m.params:
        if p.kind == "choice":
            variants += [{p.name: c} for c in p.choices if c != p.default]
        elif p.kind == "bool":
            variants.append({p.name: not p.default})
        else:
            variants += [{p.name: p.lo}, {p.name: p.hi}]
    for v in variants:
        _check_result(m.execute(v))


def test_bad_params_rejected():
    m = REGISTRY["kd"]
    with pytest.raises(ValueError):
        m.execute({"nope": 1})
    with pytest.raises(ValueError):
        m.execute({"transfer": "elsewhere"})
    assert m.coerce({"width": 10_000})["width"] == 128  # clamped


def test_reference_methods_refuse_to_run():
    for m in REGISTRY.values():
        if m.status != "runnable":
            with pytest.raises(ValueError):
                m.execute({})
