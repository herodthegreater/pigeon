"""Every lesson: the reference solution passes all tests, the untouched
skeleton fails, and each lesson links to methods that exist."""

import pytest

from lab.lessons import LESSONS, check
from lab.methods import REGISTRY


@pytest.mark.parametrize("L", LESSONS, ids=lambda L: L.id)
def test_solution_passes(L):
    r = check(L.id, L.solution)
    assert r["passed"], r


@pytest.mark.parametrize("L", LESSONS, ids=lambda L: L.id)
def test_skeleton_fails(L):
    r = check(L.id, L.skeleton)
    assert not r["passed"]
    assert r["tests"], "skeleton should import fine and report failing tests"


@pytest.mark.parametrize("L", LESSONS, ids=lambda L: L.id)
def test_links_and_metadata(L):
    assert L.hints and L.methods
    for m in L.methods:
        assert m in REGISTRY, m


def test_broken_code_reports_import_error():
    r = check(LESSONS[0].id, "def softmax(:\n")
    assert not r["passed"] and "import" in r["error"]


def test_infinite_loop_times_out():
    r = check(LESSONS[0].id, "while True:\n    pass\n", timeout=3)
    assert not r["passed"] and "timed out" in r["error"]
