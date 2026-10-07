"""Every API route the website uses, through FastAPI's test client."""

import pytest
from fastapi.testclient import TestClient

from lab.lessons import LESSONS
from lab.methods import REGISTRY
from lab.web.app import app

client = TestClient(app)


def test_index_and_static():
    assert "pigeon lab" in client.get("/").text
    for f in ("app.js", "style.css", "index.html"):
        assert client.get(f"/static/{f}").status_code == 200


def test_meta_counts_match_registry():
    meta = client.get("/api/meta").json()
    total = sum(sum(c.values()) for c in meta["counts"].values())
    assert total == len(REGISTRY)


@pytest.mark.parametrize("mid", list(REGISTRY), ids=str)
def test_method_detail(mid):
    r = client.get(f"/api/methods/{mid}")
    assert r.status_code == 200
    d = r.json()
    assert d["page"].strip(), "empty write-up"
    assert all(c["source"] for c in d["code"])
    for e in d["experiments"]:
        assert e["figures"], f"{e['name']} has no figures"
        for f in e["figures"]:
            assert client.get(f).status_code == 200


def test_run_endpoint_and_errors():
    r = client.post("/api/methods/ptq/run", json={"params": {"bits": 4, "model": "student"}})
    assert r.status_code == 200 and r.json()["rows"]
    assert client.post("/api/methods/ptq/run", json={"params": {"bits": "x"}}).status_code == 422
    assert client.post("/api/methods/ptq/run", json={"params": {"model": "nope"}}).status_code == 422
    assert client.post("/api/methods/sequence_kd/run", json={}).status_code == 400
    assert client.post("/api/methods/nope/run", json={}).status_code == 404


@pytest.mark.parametrize("L", LESSONS, ids=lambda L: L.id)
def test_lesson_routes(L):
    d = client.get(f"/api/lessons/{L.id}").json()
    assert d["skeleton"] and d["intro"] and d["hints"]
    sol = client.get(f"/api/lessons/{L.id}/solution").json()["solution"]
    assert client.post(f"/api/lessons/{L.id}/check", json={"code": sol}).json()["passed"]


def test_experiments_routes():
    exps = client.get("/api/experiments").json()
    assert len(exps) >= 28
    one = client.get(f"/api/experiments/{exps[0]['name']}").json()
    assert one["rows"]
    assert client.get("/api/experiments/nope").status_code == 404
    assert client.get("/api/report").status_code == 200
