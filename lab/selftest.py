"""Finite-difference checks for the hand-written backprop and every loss term.

    python -m lab.selftest
"""

from __future__ import annotations

import numpy as np

from .nn import MLP
from .train import CE, KD, SP, Batch, Hint, onehot


def _f64(net: MLP) -> MLP:
    net.W = [w.astype(np.float64) for w in net.W]
    net.b = [b.astype(np.float64) for b in net.b]
    return net


def _check(name, net, term, b, extra_params=()):
    def loss_at():
        z, hid, _ = net.forward(b.X)
        return term(z, hid, b)[0]

    z, hid, cache = net.forward(b.X, cache=True)
    _, dz, dhid = term(z, hid, b)
    grads, dX = net.backward(cache, dz, dhid or None)
    params = net.params() + list(extra_params)
    if extra_params:
        grads = grads + [term.dP]
    rng = np.random.default_rng(0)
    worst = 0.0
    for p, g in zip(params, grads):
        for _ in range(6):
            i = tuple(rng.integers(0, s) for s in p.shape)
            old = p[i]
            p[i] = old + 1e-5; lp = loss_at()
            p[i] = old - 1e-5; lm = loss_at()
            p[i] = old
            num = (lp - lm) / 2e-5
            worst = max(worst, abs(num - g[i]) / max(1e-6, abs(num) + abs(g[i])))
    status = "ok " if worst < 1e-3 else "BAD"
    print(f"  {status} {name:<24} max rel err {worst:.2e}")
    return worst < 1e-3


def main():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((16, 6))
    y = rng.integers(0, 5, 16)
    y[:3] = -1  # some unlabeled rows
    teacher = _f64(MLP([6, 10, 10, 5], seed=3))
    t_out, t_hid, _ = teacher.forward(X)
    b = Batch(X, onehot(y, 5).astype(np.float64), y >= 0, t_out, t_hid)
    G = np.zeros((5, 2)); G[0, 0] = 1; G[1:, 1] = 1
    ok = True
    for res in (False, True):
        tag = "res" if res else "plain"
        mk = lambda: _f64(MLP([6, 8, 8, 5], seed=2, res=res))
        ok &= _check(f"CE+ls [{tag}]", mk(), CE(smoothing=0.1), b)
        for kind in ("kl", "rkl", "mse", "topk"):
            ok &= _check(f"KD {kind} [{tag}]", mk(), KD(T=3, kind=kind), b)
        ok &= _check(f"SP [{tag}]", mk(), SP(1, 1), b)
        h = Hint(0, 1, 8, 10, seed=4); h.P = h.P.astype(np.float64)
        ok &= _check(f"Hint [{tag}]", mk(), h, b, extra_params=[h.P])
    lin = lambda: _f64(MLP([6, 12, 8, 12, 5], seed=6, linear={0, 2}))
    ok &= _check("KD kl [linear layers]", lin(), KD(T=2), b)
    net = lin()
    small = net.collapse_linear()
    gap = np.abs(net.forward(X)[0] - small.forward(X)[0]).max()
    print(f"  {'ok ' if gap < 1e-9 else 'BAD'} collapse_linear exact      max |diff| {gap:.1e}  {net.sizes} -> {small.sizes}")
    ok &= gap < 1e-9
    spec = _f64(MLP([6, 8, 2], seed=5))
    ok &= _check("KD grouped (specialist)", spec, KD(T=2, group=G), b)
    print("all gradients OK" if ok else "GRADIENT CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
