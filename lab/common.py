"""Shared plumbing for experiments: the standard world, cached teachers,
step-budgeted training helpers, seed aggregation and the experiment registry.

Set ``LAB_QUICK=1`` to shrink every budget ~10x (smoke-test the whole suite in
a couple of minutes; numbers are then not meaningful).
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict

import numpy as np

from .nn import MLP
from .train import CE, KD, FixedSource, evaluate, fit
from .world import Data, Split, make_data

LAB = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(LAB, ".cache")
RESULTS = os.path.join(LAB, "results")

QUICK = os.environ.get("LAB_QUICK") == "1"
SEEDS = [0] if QUICK else [0, 1, 2]
STEPS = 300 if QUICK else 3000           # default student budget (~48 epochs of 8k)
T_EPOCHS = 6 if QUICK else 60            # teacher epochs

TEACHER = [256, 256, 256]                # ~137k params
STUDENT = [32, 32]                       # ~1.7k params (80x smaller)
TINY = [16, 16]                          # ~0.8k params


# ---------------------------------------------------------------------------
# data + nets
# ---------------------------------------------------------------------------
_DATA: dict = {}


def data(label_noise: float = 0.0, n_train: int = 8000) -> Data:
    """The standard world. Variants (fewer labels, noisy labels) are carved out
    of the same draw so every experiment is scored on the *same* test set."""
    key = (label_noise, n_train)
    if key not in _DATA:
        if (0.0, 8000) not in _DATA:
            _DATA[(0.0, 8000)] = make_data(n_train=8000)
        base = _DATA[(0.0, 8000)]
        tr = base.train.take(n_train)
        if label_noise:
            rng = np.random.default_rng(77)
            y = tr.y.copy()
            flip = rng.random(len(y)) < label_noise
            y[flip] = rng.integers(0, base.world.C, flip.sum())
            tr = Split(tr.X, y, tr.y_clean, tr.bayes)
        _DATA[key] = Data(base.world, tr, base.pool, base.test)
    return _DATA[key]


def net(hidden: list[int], seed: int = 0, **kw) -> MLP:
    d = data()
    return MLP([d.world.spec.d] + list(hidden) + [d.world.C], seed=seed, **kw)


def cached(key: str, build):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, ("q_" if QUICK else "") + key + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    obj = build()
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)
    return obj


def teacher(hidden=TEACHER, seed: int = 0, epochs: int | None = None, smoothing: float = 0.0,
            label_noise: float = 0.0, n_train: int = 8000, res: bool = False,
            wd: float = 1e-4, lr: float = 2e-3, tag: str = "") -> MLP:
    """A trained 'big model', cached on disk so every experiment shares it."""
    epochs = epochs or T_EPOCHS
    key = (f"teacher_{'x'.join(map(str, hidden))}_s{seed}_e{epochs}_ls{smoothing}"
           f"_n{label_noise}_m{n_train}_r{int(res)}_wd{wd}{tag}")

    def build():
        d = data(label_noise, n_train)
        m = net(hidden, seed=seed, res=res)
        fit(m, FixedSource(d.train.X, d.train.y, d.world.C), [CE(smoothing=smoothing)],
            epochs=epochs, lr=lr, wd=wd, seed=seed)
        return m
    return cached(key, build)


# ---------------------------------------------------------------------------
# training helpers (fixed *step* budget so different transfer-set sizes are fair)
# ---------------------------------------------------------------------------
def train(m: MLP, source, terms, steps: int | None = None, bs: int = 128, **kw) -> MLP:
    steps = steps or STEPS
    epochs = int(np.ceil(steps * bs / len(source)))
    return fit(m, source, terms, epochs=epochs, bs=bs, max_steps=steps, **kw)


def scratch(hidden, seed: int = 0, d: Data | None = None, steps=None, smoothing=0.0, **kw) -> MLP:
    d = d or data()
    m = net(hidden, seed=seed) if not isinstance(hidden, MLP) else hidden
    return train(m, FixedSource(d.train.X, d.train.y, d.world.C), [CE(smoothing=smoothing)],
                 steps=steps, seed=seed, **kw)


def transfer_X(d: Data, transfer: str = "pool") -> tuple[np.ndarray, np.ndarray]:
    """Inputs (and labels, -1 = unlabeled) for a distillation transfer set."""
    if transfer == "train":
        return d.train.X, d.train.y
    if transfer == "pool":  # labeled train + unlabeled pool
        return (np.concatenate([d.train.X, d.pool.X]),
                np.concatenate([d.train.y, -np.ones(len(d.pool), np.int64)]))
    raise ValueError(transfer)


def distill(student, t: MLP, seed: int = 0, transfer: str = "pool", T: float = 1.0,
            alpha: float = 1.0, kind: str = "kl", k: int = 3, d: Data | None = None, steps=None,
            source=None, extra_terms=(), **kw) -> MLP:
    """Train ``student`` (hidden widths or an MLP to continue from) on
    ``alpha * KD + (1 - alpha) * CE``.

    Default T=1: in this world a higher temperature makes capacity-limited
    students *worse* (see the temperature_alpha experiment), so every
    experiment that isn't about temperature distills at T=1."""
    d = d or data()
    m = net(student, seed=seed) if not isinstance(student, MLP) else student
    if source is None:
        X, y = transfer_X(d, transfer)
        source = FixedSource(X, y, d.world.C, teacher=t, hidden=bool(extra_terms))
    terms = [KD(T=T, weight=alpha, kind=kind, k=k)] + list(extra_terms)
    if alpha < 1:
        terms.append(CE(weight=1 - alpha))
    return train(m, source, terms, steps=steps, seed=seed, **kw)


def ev(m: MLP, t: MLP | None = None, d: Data | None = None) -> dict:
    d = d or data()
    return evaluate(m, d.test, None if t is None else t.logits(d.test.X))


# ---------------------------------------------------------------------------
# aggregation + registry
# ---------------------------------------------------------------------------
def agg(rows: list[dict], by: list[str]) -> list[dict]:
    """Average numeric fields over rows that share the ``by`` keys (i.e. seeds)."""
    groups: dict = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in by)].append(r)
    out = []
    for key, rs in groups.items():
        row = dict(zip(by, key))
        for k in rs[0]:
            if k in by or k == "seed":
                continue
            vals = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
            if vals and len(vals) == len(rs):
                row[k] = float(np.mean(vals))
                if len(vals) > 1:
                    row[k + "_std"] = float(np.std(vals))
            else:
                row[k] = rs[0][k]
        row["n_seeds"] = len(rs)
        out.append(row)
    return out


REGISTRY: dict[str, dict] = {}


def experiment(name: str, group: str, title: str, question: str):
    def deco(fn):
        REGISTRY[name] = dict(fn=fn, group=group, title=title, question=question, name=name)
        return fn
    return deco


def refs(t: MLP | None = None) -> dict:
    """Reference lines for plots: teacher and Bayes-optimal."""
    from .train import bayes_metrics
    d = data()
    out = {"bayes": bayes_metrics(d.test)}
    if t is not None:
        out["teacher"] = ev(t)
    return out
