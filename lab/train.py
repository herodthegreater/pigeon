"""Training loop, the distillation-loss zoo, and evaluation.

A student learns from a *batch source* (which inputs it sees, and what the
teacher says about them) through a list of *loss terms*. Every distillation
variant in the experiments is some combination of those two choices:

    source  : FixedSource (precomputed teacher outputs) | OnlineSource (teacher
              queried on freshly augmented inputs every step -- "function matching")
    terms   : CE  | KD(kind=kl|rkl|mse|hard|topk, T, group) | Hint | SP

All terms return ``(loss, dlogits, dhid)`` so the MLP's manual backprop can
consume them; Hint carries its own trainable projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nn import MLP, Adam
from .world import TARGET, Split

F = np.float32


# ---------------------------------------------------------------------------
# small numerics
# ---------------------------------------------------------------------------
def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def log_softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def onehot(y: np.ndarray, C: int) -> np.ndarray:
    out = np.zeros((len(y), C), F)
    ok = y >= 0
    out[np.nonzero(ok)[0], y[ok]] = 1
    return out


# ---------------------------------------------------------------------------
# batch sources
# ---------------------------------------------------------------------------
@dataclass
class Batch:
    X: np.ndarray
    Y: np.ndarray | None        # soft/one-hot ground-truth targets (N, C)
    lab: np.ndarray | None      # bool: which rows have a ground-truth label
    t_logits: np.ndarray | None = None
    t_hidden: list[np.ndarray] | None = None


class FixedSource:
    """Fixed inputs (+ optional precomputed teacher outputs), reshuffled per epoch.

    ``y`` uses -1 for unlabeled rows (e.g. the transfer pool)."""

    def __init__(self, X, y=None, C=None, teacher: MLP | None = None, t_logits=None,
                 hidden: bool = False):
        self.X = X.astype(F)
        self.y = y
        self.C = C
        self.t_logits, self.t_hidden = t_logits, None
        if teacher is not None:
            out, hid, _ = teacher.forward(self.X)
            self.t_logits = out
            self.t_hidden = hid if hidden else None

    def __len__(self):
        return len(self.X)

    def batches(self, rng: np.random.Generator, bs: int):
        idx = rng.permutation(len(self.X))
        for s in range(0, len(idx), bs):
            j = idx[s:s + bs]
            Y = lab = None
            if self.y is not None:
                yy = self.y[j]
                lab = yy >= 0
                Y = onehot(yy, self.C)
            yield Batch(self.X[j], Y, lab,
                        None if self.t_logits is None else self.t_logits[j],
                        None if self.t_hidden is None else [h[j] for h in self.t_hidden])


class OnlineSource:
    """Teacher is queried on *fresh* augmented inputs every step (Beyer et al.'s
    'consistent teaching'). ``aug``: none | mixup | noise | mixup+noise."""

    def __init__(self, X, y, C, teacher: MLP | None, aug: str = "mixup", noise: float = 0.3,
                 hidden: bool = False, extra_X=None):
        self.X = X if extra_X is None else np.concatenate([X, extra_X])
        self.y = y if extra_X is None else np.concatenate([y, -np.ones(len(extra_X), np.int64)])
        self.C, self.teacher, self.aug, self.noise, self.hidden = C, teacher, aug, noise, hidden

    def __len__(self):
        return len(self.X)

    def batches(self, rng: np.random.Generator, bs: int):
        idx = rng.permutation(len(self.X))
        for s in range(0, len(idx), bs):
            j = idx[s:s + bs]
            X = self.X[j]
            Y = onehot(self.y[j], self.C)
            lab = self.y[j] >= 0
            if "mixup" in self.aug:
                k = rng.permutation(len(j))
                lam = rng.beta(1.0, 1.0, size=(len(j), 1)).astype(F)
                X = lam * X + (1 - lam) * X[k]
                Y = lam * Y + (1 - lam) * Y[k]
                lab = lab & lab[k]
            if "noise" in self.aug:
                X = X + rng.standard_normal(X.shape).astype(F) * self.noise
            if self.teacher is None:  # plain augmentation, no teacher
                yield Batch(X, Y, lab)
                continue
            out, hid, _ = self.teacher.forward(X)
            yield Batch(X, Y, lab, out, hid if self.hidden else None)


# ---------------------------------------------------------------------------
# loss terms
# ---------------------------------------------------------------------------
class CE:
    """Cross-entropy on ground-truth labels (rows without a label are ignored)."""

    def __init__(self, weight: float = 1.0, smoothing: float = 0.0):
        self.w, self.ls = weight, smoothing

    def __call__(self, z, hid, b: Batch):
        N = len(z)
        if b.Y is None or not b.lab.any():
            return 0.0, np.zeros_like(z), {}
        Y = b.Y
        if self.ls:
            Y = Y * (1 - self.ls) + self.ls / Y.shape[1]
        m = b.lab.astype(F)[:, None]
        n = max(m.sum(), 1.0)
        loss = -(m * Y * log_softmax(z)).sum() / n
        dz = (softmax(z) - Y) * m / n
        return self.w * loss, self.w * dz, {}

    params = staticmethod(lambda: [])


class KD:
    """Match the teacher's outputs.

    kind:
      kl    -- Hinton KD: T^2 * KL(p_t^T || p_s^T)            (mode-covering)
      rkl   -- reverse KL: T^2 * KL(p_s^T || p_t^T)           (mode-seeking)
      mse   -- squared error on raw logits (T ignored)
      hard  -- CE on the teacher's argmax (pseudo-labels; no dark knowledge)
      topk  -- teacher only reveals its top-k probabilities (an API that returns
               top-k logprobs); the rest of the mass is spread uniformly
    group (C_t x C_s matrix): sum teacher probabilities into the student's
      coarser label space, e.g. 20 classes -> {bird, other} for a specialist.
    """

    def __init__(self, T: float = 4.0, weight: float = 1.0, kind: str = "kl", k: int = 3,
                 group: np.ndarray | None = None):
        self.T, self.w, self.kind, self.k, self.G = T, weight, kind, k, group

    def _teacher_probs(self, tl):
        T = 1.0 if self.kind == "hard" else self.T
        p = softmax(tl, T)
        if self.kind == "topk":
            thr = -np.sort(-p, 1)[:, self.k - 1:self.k]
            keep = p >= thr
            rest = (1 - (p * keep).sum(1, keepdims=True)) / (~keep).sum(1, keepdims=True).clip(1)
            p = np.where(keep, p, rest)
        if self.G is not None:
            p = p @ self.G
        if self.kind == "hard":
            p = onehot(p.argmax(1), p.shape[1])
        return p.astype(F)

    def __call__(self, z, hid, b: Batch):
        N, T = len(z), self.T
        if self.kind == "mse":
            t = b.t_logits if self.G is None else np.log(softmax(b.t_logits) @ self.G + 1e-8)
            r = z - t
            return self.w * (r ** 2).mean(), self.w * 2 * r / r.size, {}
        pt = self._teacher_probs(b.t_logits)
        if self.kind == "hard":
            loss = -(pt * log_softmax(z)).sum() / N
            return self.w * loss, self.w * (softmax(z) - pt) / N, {}
        ls = log_softmax(z, T)
        ps = np.exp(ls)
        lt = np.log(pt + 1e-8)
        if self.kind == "rkl":
            per = (ps * (ls - lt)).sum(1, keepdims=True)
            loss = T * T * per.sum() / N
            dz = T * ps * (ls - lt - per) / N
        else:  # kl / topk
            loss = T * T * (pt * (lt - ls)).sum() / N
            dz = T * (ps - pt) / N
        return self.w * loss, self.w * dz, {}

    params = staticmethod(lambda: [])


class Hint:
    """FitNets: regress a student hidden layer onto a teacher hidden layer
    through a learned linear projection (student width -> teacher width)."""

    def __init__(self, s_layer: int, t_layer: int, d_s: int, d_t: int, weight: float = 1.0,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.s, self.t, self.w = s_layer, t_layer, weight
        self.P = (rng.standard_normal((d_s, d_t)) / np.sqrt(d_s)).astype(F)
        self.dP = np.zeros_like(self.P)

    def params(self):
        return [self.P]

    def __call__(self, z, hid, b: Batch):
        hs, ht = hid[self.s], b.t_hidden[self.t]
        r = hs @ self.P - ht
        n = r.size
        self.dP = self.w * hs.T @ (2 * r / n)
        dh = self.w * (2 * r / n) @ self.P.T
        return self.w * (r ** 2).mean(), np.zeros_like(z), {self.s: dh}


class SP:
    """Similarity-preserving KD (Tung & Mori): match the batch's pairwise
    similarity structure, not the features themselves. Width-agnostic."""

    def __init__(self, s_layer: int, t_layer: int, weight: float = 1.0):
        self.s, self.t, self.w = s_layer, t_layer, weight

    @staticmethod
    def _gram(h):
        A = h @ h.T
        nrm = np.sqrt((A ** 2).sum(1, keepdims=True)) + 1e-8
        return A, nrm, A / nrm

    def __call__(self, z, hid, b: Batch):
        hs, ht = hid[self.s], b.t_hidden[self.t]
        N = len(hs)
        A, nrm, Gs = self._gram(hs)
        _, _, Gt = self._gram(ht)
        dG = 2 * (Gs - Gt) / (N * N)
        dA = (dG - Gs * (Gs * dG).sum(1, keepdims=True)) / nrm
        dh = (dA + dA.T) @ hs
        loss = ((Gs - Gt) ** 2).sum() / (N * N)
        return self.w * loss, np.zeros_like(z), {self.s: self.w * dh}

    params = staticmethod(lambda: [])


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
def fit(net: MLP, source, terms: list, epochs: int = 40, lr: float = 3e-3, wd: float = 1e-4,
        bs: int = 128, seed: int = 0, ema: float | None = None, callback=None,
        max_steps: int | None = None) -> MLP:
    """Train ``net`` in place (cosine LR, AdamW). Returns the net (EMA weights if
    ``ema`` is set). ``callback(epoch, net)`` runs after every epoch."""
    rng = np.random.default_rng(seed)
    extra = [p for t in terms for p in t.params()]
    opt = Adam(net.params() + extra, lr=lr, wd=wd)
    steps_per = int(np.ceil(len(source) / bs))
    total = max_steps or epochs * steps_per
    shadow = [p.copy() for p in net.params()] if ema else None
    step = 0
    for ep in range(epochs):
        for b in source.batches(rng, bs):
            opt.lr = lr * 0.5 * (1 + np.cos(np.pi * min(step / total, 1.0)))
            z, hid, cache = net.forward(b.X, cache=True)
            dz = np.zeros_like(z)
            dhid: dict[int, np.ndarray] = {}
            for t in terms:
                _, g, gh = t(z, hid, b)
                dz += g
                for k, v in gh.items():
                    dhid[k] = dhid.get(k, 0) + v
            grads, _ = net.backward(cache, dz, dhid or None)
            grads += [t.dP for t in terms if isinstance(t, Hint)]
            opt.step(grads)
            net.apply_masks()
            if shadow is not None:
                for s_, p in zip(shadow, net.params()):
                    s_ *= ema; s_ += (1 - ema) * p
            step += 1
            if step >= total:
                break
        if callback:
            callback(ep, net)
        if step >= total:
            break
    if shadow is not None:
        for s_, p in zip(shadow, net.params()):
            p[...] = s_
    return net


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def average_precision(scores: np.ndarray, positive: np.ndarray) -> float:
    """One-vs-rest AP (all-points), the same flavour as pigeon's bird mAP@50."""
    order = np.argsort(-scores, kind="stable")
    pos = positive[order]
    tp = np.cumsum(pos)
    prec = tp / np.arange(1, len(pos) + 1)
    return float((prec * pos).sum() / max(pos.sum(), 1))


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def evaluate(net: MLP, split: Split, teacher_logits: np.ndarray | None = None,
             target_out: int = TARGET, full_space: bool = True) -> dict:
    """Metrics on ``split``. For a specialist (``full_space=False``) only the
    target AP is meaningful; the student's ``target_out`` column is its score."""
    z = net.logits(split.X)
    p = softmax(z)
    tgt = split.y == TARGET
    r = {"target_ap": average_precision(p[:, target_out], tgt),
         "params": net.n_params(), "macs": net.macs()}
    if full_space:
        pred = p.argmax(1)
        eps = 1e-8
        r.update(
            acc=float((pred == split.y).mean()),
            nll=float(-np.log(p[np.arange(len(p)), split.y] + eps).mean()),
            ece=ece(p, split.y),
            kl_bayes=float((split.bayes * (np.log(split.bayes + eps) - np.log(p + eps))).sum(1).mean()),
        )
        if teacher_logits is not None:
            r["fidelity"] = float((pred == teacher_logits.argmax(1)).mean())
    return r


def bayes_metrics(split: Split) -> dict:
    p = split.bayes
    return {"acc": float((p.argmax(1) == split.y).mean()),
            "target_ap": average_precision(p[:, TARGET], split.y == TARGET),
            "nll": float(-np.log(p[np.arange(len(p)), split.y] + 1e-8).mean())}
