"""Numbers a write-up quotes from its page's Run must be what Run gives.

Each case below re-runs one method at full budget with the models the website
ships (lab/models, never a local lab/.cache) and checks that the numbers the
run produces appear in that method's write-up. Retraining a shipped model, or
editing a quote by hand, fails here.

Only runs that train nothing and round no activations are checked: they load
a shipped model, round or prune its weights, and score it, so CPython and the
browser's Pyodide agree on them to the last digit. Runs that train, or that
round activations (SmoothQuant), can end a test input or more apart on
another BLAS, so their quotes are measured in Pyodide
(lab/web/run_in_pyodide.cjs) and are not checked here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAGES = os.path.join(ROOT, "lab", "web", "content", "methods")


def acc(label):
    """The accuracy of the first row whose label starts with ``label``."""
    return (label, "acc", "{:.4f}")


def nonzero(label):
    """The nonzero-parameter count of that row, written with thousands commas."""
    return (label, "params", "{:,.0f}")


# (method id, params, quoted values). Every value must appear in <id>.md.
CASES = [
    ("ptq", {}, [acc("chosen")]),
    ("ptq", {"bits": 2, "group": "4"}, [acc("chosen")]),
    ("rtn", {"bits": 8}, [acc("RTN"), nonzero("RTN")]),
    ("rtn", {"bits": 3}, [acc("teacher"), acc("RTN"), nonzero("RTN")]),
    ("rtn", {"bits": 3, "group": "16"}, [acc("RTN"), nonzero("RTN")]),
    ("rtn", {"bits": 2}, [acc("RTN"), nonzero("RTN")]),
    ("gptq", {"bits": 3}, [acc("RTN"), acc("GPTQ")]),
    ("gptq", {"bits": 2, "group": "16"}, [acc("RTN"), acc("GPTQ")]),
    ("gptq", {"bits": 2, "act_order": True}, [acc("GPTQ")]),
    ("awq", {"bits": 4}, [acc("RTN"), acc("AWQ")]),
    ("awq", {"bits": 2}, [acc("AWQ")]),
    ("wanda", {"sparsity": 0.5}, [acc("magnitude"), acc("Wanda")]),
    ("wanda", {"sparsity": 0.6, "outliers": 64}, [acc("magnitude"), acc("Wanda")]),
    ("sparsegpt", {"sparsity": 0.7, "block": 32},
     [acc("magnitude"), acc("Wanda"), acc("SparseGPT"), nonzero("SparseGPT")]),
    ("block_influence", {"k": 2},
     [acc("residual"), acc("drop 2 lowest"), acc("drop 2 random"), acc("drop 2 chosen")]),
]

SCRIPT = """
import json, os, sys, tempfile, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, sys.argv[1])
import lab.common as common
assert not common.QUICK
common.CACHE = tempfile.mkdtemp()   # only lab/models: what the website ships
_cached = common.cached

def shipped_only(key, build):       # never train a stand-in for a missing model
    assert os.path.exists(common.shipped_path(key)), f"lab/models/{key}.npz is missing"
    return _cached(key, build)

common.cached = shipped_only        # before lab.methods imports it
from lab.methods import REGISTRY
out = []
for mid, params in json.loads(sys.argv[2]):
    m = REGISTRY[mid]
    out.append(m.run(m.coerce(params))["rows"])
print(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
"""


@pytest.fixture(scope="module")
def runs():
    env = {k: v for k, v in os.environ.items() if k != "LAB_QUICK"}
    todo = json.dumps([[mid, params] for mid, params, _ in CASES])
    p = subprocess.run([sys.executable, "-c", SCRIPT, ROOT, todo], capture_output=True,
                       text=True, env=env, cwd=ROOT, timeout=900)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("i", range(len(CASES)),
                         ids=[f"{mid}-{json.dumps(ps, sort_keys=True)}" for mid, ps, _ in CASES])
def test_quoted_run(i, runs):
    mid, params, quotes = CASES[i]
    rows = runs[i]
    text = open(os.path.join(PAGES, f"{mid}.md")).read()
    for label, field, fmt in quotes:
        row = next((r for r in rows if r["label"].startswith(label)), None)
        assert row is not None, f"{mid} {params}: no row starting {label!r} in {[r['label'] for r in rows]}"
        value = fmt.format(row[field])
        assert value in text, (f"{mid}.md should quote {value} ({label!r} {field}, Run at {params}); "
                               "re-measure the page's runs and update the write-up")
