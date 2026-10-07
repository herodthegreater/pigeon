"""The gradient self-check, and every experiment end to end at quick budgets
(results are returned, not written, so this never touches lab/results)."""

import math

import pytest

from lab import experiments  # noqa: F401  (registers experiments)
from lab import selftest
from lab.common import REGISTRY


def test_gradients():
    assert selftest.main() == 0


@pytest.mark.parametrize("name", list(REGISTRY), ids=str)
def test_experiment_runs(name):
    out = REGISTRY[name]["fn"]()
    assert out["rows"], "no rows"
    cols = {k for r in out["rows"] for k in r}
    for spec in out["plots"]:
        assert spec["x"] in cols and spec["y"] in cols, spec
    for r in out["rows"]:
        for v in r.values():
            if isinstance(v, float):
                assert math.isfinite(v)
