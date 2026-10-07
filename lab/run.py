"""Run the simulation suite.

    python -m lab.run                 # everything (~30-60 min on 4 CPU cores)
    python -m lab.run size_ladder loss_zoo
    python -m lab.run --group 6       # one group
    python -m lab.run --list
    LAB_QUICK=1 python -m lab.run     # smoke test, ~10x smaller budgets

Results land in lab/results/<name>.json, figures in lab/results/figures/, and
lab/results/RESULTS.md is regenerated from whatever results exist.
"""

from __future__ import annotations

import os

# one BLAS thread per process; we parallelize across experiments instead
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

from . import experiments  # noqa: E402,F401  (registers everything)
from .common import QUICK, REGISTRY, RESULTS, teacher  # noqa: E402

# slowest first, so the pool stays busy
ORDER_HINT = ["compression_frontier", "prune_structured", "lottery_ticket", "small_tricks",
              "capacity_gap", "prune_unstructured", "function_matching", "ensemble_teacher"]


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


def run_one(name: str) -> tuple[str, float, str | None]:
    spec = REGISTRY[name]
    t0 = time.time()
    try:
        out = spec["fn"]()
    except Exception:
        return name, time.time() - t0, traceback.format_exc()
    out = dict(name=name, group=spec["group"], title=spec["title"], question=spec["question"],
               quick=QUICK, seconds=round(time.time() - t0, 1), **out)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, ("quick_" if QUICK else "") + name + ".json")
    with open(path, "w") as f:
        json.dump(_jsonable(out), f, indent=1)
    return name, time.time() - t0, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--group", help="run every experiment whose group starts with this")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()

    if args.list:
        for n, s in REGISTRY.items():
            print(f"{s['group']:<44} {n:<26} {s['title']}")
        return
    names = args.names or [n for n, s in REGISTRY.items()
                           if not args.group or s["group"].startswith(args.group)]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown experiment(s): {unknown}; see --list")
    names.sort(key=lambda n: ORDER_HINT.index(n) if n in ORDER_HINT else len(ORDER_HINT))

    print(f"warming the shared teacher (quick={QUICK}) ...", flush=True)
    t0 = time.time()
    teacher()
    print(f"  teacher ready in {time.time() - t0:.0f}s", flush=True)

    failed = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(run_one, n): n for n in names}
        for f in as_completed(futs):
            name, secs, err = f.result()
            if err:
                failed.append(name)
                print(f"  FAIL {name} ({secs:.0f}s)\n{err}", flush=True)
            else:
                print(f"  done {name} ({secs:.0f}s)", flush=True)

    from .report import build
    build()
    if failed:
        raise SystemExit(f"{len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
