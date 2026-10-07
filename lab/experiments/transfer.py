"""Group 3 -- the transfer set: what inputs do you show the teacher and student?
Includes the no-real-data edge cases (random noise, teacher inversion)."""

from __future__ import annotations

import numpy as np

from ..common import (QUICK, SEEDS, STEPS, STUDENT, agg, data, ev, experiment, net,
                      refs, scratch, teacher, train)
from ..nn import MLP, Adam
from ..train import KD, FixedSource, OnlineSource, softmax

G = "3 · What the student sees (transfer data)"


def invert_teacher(t: MLP, n: int, steps: int = 300, lr: float = 0.05, l2: float = 0.05,
                   seed: int = 0) -> np.ndarray:
    """Data-free distillation: synthesize inputs by gradient descent *on the
    input* so the teacher confidently predicts a random class, with an L2
    prior keeping them near the data scale. No real example is used."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, t.sizes[0])).astype(np.float32)
    c = rng.integers(0, t.sizes[-1], n)
    Y = np.eye(t.sizes[-1], dtype=np.float32)[c]
    opt = Adam([X], lr=lr)
    for _ in range(steps):
        z, _, cache = t.forward(X, cache=True)
        dz = (softmax(z) - Y) / n
        _, dX = t.backward(cache, dz)
        opt.step([dX + l2 * X / n])
    return X


@experiment("transfer_set", G, "Distilling without (much) real data",
            "Which inputs carry the teacher's knowledge best -- real labeled, real "
            "unlabeled, augmented, pure noise, or inputs dreamed up by inverting the teacher?")
def transfer_set():
    d = data()
    t = teacher()
    C = d.world.C
    rng = np.random.default_rng(0)
    N = len(d.train) + len(d.pool)
    few = d.train.take(500)
    inv = invert_teacher(t, 2000 if QUICK else 10000)
    srcs = {
        "real: labeled set (8k)": lambda: FixedSource(d.train.X, None, C, teacher=t),
        "real: labeled + unlabeled pool (28k)": lambda: FixedSource(
            np.concatenate([d.train.X, d.pool.X]), None, C, teacher=t),
        "real: 8k + online mixup": lambda: OnlineSource(d.train.X, d.train.y, C, t, aug="mixup"),
        "real: 8k + online jitter": lambda: OnlineSource(d.train.X, d.train.y, C, t, aug="noise", noise=0.5),
        "real: 500 only": lambda: FixedSource(few.X, None, C, teacher=t),
        "real: 500 + online mixup": lambda: OnlineSource(few.X, few.y, C, t, aug="mixup"),
        "real: 500 + online mixup+jitter": lambda: OnlineSource(few.X, few.y, C, t, aug="mixup+noise", noise=0.5),
        "fake: gaussian noise N(0,1)": lambda: FixedSource(
            rng.standard_normal((N, d.world.spec.d)).astype(np.float32), None, C, teacher=t),
        "fake: uniform noise [-3,3]": lambda: FixedSource(
            rng.uniform(-3, 3, (N, d.world.spec.d)).astype(np.float32), None, C, teacher=t),
        "fake: teacher inversion (data-free)": lambda: FixedSource(inv, None, C, teacher=t),
        "fake: inversion + 500 real": lambda: FixedSource(
            np.concatenate([inv, few.X]), None, C, teacher=t),
    }
    rows = []
    for s in SEEDS:
        rows.append(dict(source="(labels only, 8k scratch)", seed=s, **ev(scratch(STUDENT, s), t)))
        rows.append(dict(source="(labels only, 500 scratch)", seed=s,
                         **ev(scratch(STUDENT, s, d=data(n_train=500)), t)))
        for name, mk in srcs.items():
            m = train(net(STUDENT, s), mk(), [KD(T=4)], seed=s)
            rows.append(dict(source=name, seed=s, **ev(m, t)))
    return dict(rows=agg(rows, ["source"]), refs=refs(t), plots=[
        dict(x="source", y="acc", kind="bar", hlines=["teacher"]),
        dict(x="source", y="fidelity", kind="bar"),
    ])


@experiment("function_matching", G, "Patient & consistent teaching",
            "Beyer et al.: treat KD as *function matching* -- query the teacher on fresh "
            "augmented inputs every step and train for a long time. Does it keep paying?")
def function_matching():
    d = data()
    t = teacher()
    C = d.world.C
    rows = []
    for mult in [0.5, 2, 8] if not QUICK else [1]:
        steps = int(STEPS * mult)
        for s in SEEDS[:2]:
            arms = {
                "fixed teacher targets": FixedSource(d.train.X, None, C, teacher=t),
                "online mixup (consistent)": OnlineSource(d.train.X, d.train.y, C, t, aug="mixup"),
                "online mixup + jitter": OnlineSource(d.train.X, d.train.y, C, t, aug="mixup+noise"),
            }
            for name, src in arms.items():
                m = train(net(STUDENT, s), src, [KD(T=1)], steps=steps, seed=s)
                rows.append(dict(steps=steps, method=name, seed=s, **ev(m, t)))
            rows.append(dict(steps=steps, method="labels only (scratch)", seed=s,
                             **ev(scratch(STUDENT, s, steps=steps), t)))
    return dict(rows=agg(rows, ["steps", "method"]), refs=refs(t), plots=[
        dict(x="steps", y="acc", group="method", xlog=True, hlines=["teacher"]),
        dict(x="steps", y="fidelity", group="method", xlog=True),
    ])

