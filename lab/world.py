"""A synthetic 'generalist world' with a known right answer.

The real pigeon problem is: a detector trained on 80 COCO classes, of which we
keep one (bird). This module builds a miniature, fully-controlled version of
that setup so distillation and compression can be simulated thousands of times
on a CPU:

- **20 classes in 5 superclasses** (4 classes each). Class 0 is the *target*
  ("bird"); classes 1-3 are its siblings (the things a bird is confused with).
  Superclass structure is what gives a teacher's soft labels *dark knowledge*:
  a teacher that says "70% bird, 20% other-animal, 0% car" carries information
  a one-hot label does not.
- **Each class is a mixture of modes** (sub-clusters), so boundaries are
  multi-modal and small networks are genuinely capacity-limited.
- **A smooth, invertible warp** ``x = R1 z + 0.5 sin(R2 z)`` maps the Gaussian
  latent ``z`` to inputs ``x``. Because the warp is a bijection, the
  **Bayes-optimal posterior** ``p(y|x) = p(y|z)`` is computable exactly from
  the Gaussian mixture. That gives every experiment a ceiling (Bayes accuracy)
  and lets us ask how close a teacher's soft labels are to the *true*
  posterior -- the quantity distillation theory says matters.

Everything is deterministic given ``seed``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TARGET = 0  # the "bird": the one class pigeon wants to keep


@dataclass
class WorldSpec:
    # Calibrated so capacity matters: Bayes 97% acc, linear 52%, a 2x8 MLP
    # ~40%, 2x32 ~78%, 2x256 ~92%. A real ladder for compression to climb down.
    d: int = 12              # input dimensionality
    n_super: int = 5         # superclasses
    per_super: int = 4       # classes per superclass
    modes: int = 8           # Gaussian modes per class (multi-modal boundaries)
    s_super: float = 1.2     # spread of superclass centres
    s_class: float = 1.0     # spread of class centres within a superclass
    s_mode: float = 2.5      # spread of modes within a class
    noise: float = 1.5       # within-mode std
    warp: float = 0.5        # strength of the nonlinear (sin) part of the warp
    seed: int = 0

    @property
    def n_classes(self) -> int:
        return self.n_super * self.per_super


@dataclass
class Split:
    X: np.ndarray            # (N, d) standardized inputs, float32
    y: np.ndarray            # (N,) observed labels (may be noisy), int64
    y_clean: np.ndarray      # (N,) true generating class
    bayes: np.ndarray        # (N, C) exact Bayes posterior p(y|x)

    def __len__(self) -> int:
        return len(self.y)

    def take(self, n: int) -> "Split":
        return Split(self.X[:n], self.y[:n], self.y_clean[:n], self.bayes[:n])


@dataclass
class World:
    spec: WorldSpec
    centres: np.ndarray      # (C, modes, d) latent mode centres
    R1: np.ndarray
    R2: np.ndarray
    mu: np.ndarray = field(default=None)
    sd: np.ndarray = field(default=None)

    @property
    def C(self) -> int:
        return self.spec.n_classes

    def superclass(self, c):
        return np.asarray(c) // self.spec.per_super

    def siblings(self, c: int = TARGET) -> list[int]:
        s = c // self.spec.per_super
        return [k for k in range(s * self.spec.per_super, (s + 1) * self.spec.per_super) if k != c]

    # -- generative process -------------------------------------------------
    def _warp(self, z: np.ndarray) -> np.ndarray:
        return z @ self.R1.T + self.spec.warp * np.sin(z @ self.R2.T)

    def _posterior(self, z: np.ndarray) -> np.ndarray:
        """Exact p(y|z) for the equal-weight isotropic Gaussian mixture."""
        s = self.spec
        diff = z[:, None, None, :] - self.centres[None]          # N,C,M,d
        logp = -0.5 * (diff ** 2).sum(-1) / s.noise ** 2          # N,C,M
        m = logp.max(axis=(1, 2), keepdims=True)
        lik = np.exp(logp - m).sum(-1)                            # N,C
        return lik / lik.sum(1, keepdims=True)

    def sample(self, n: int, rng: np.random.Generator, label_noise: float = 0.0,
               class_probs: np.ndarray | None = None) -> Split:
        s = self.spec
        y = rng.choice(self.C, size=n, p=class_probs)
        mode = rng.integers(0, s.modes, size=n)
        z = self.centres[y, mode] + rng.standard_normal((n, s.d)) * s.noise
        x = self._warp(z)
        bayes = self._posterior(z)
        if class_probs is not None:  # prior shift changes the Bayes posterior too
            bayes = bayes * class_probs[None] * self.C
            bayes /= bayes.sum(1, keepdims=True)
        y_obs = y.copy()
        if label_noise > 0:
            flip = rng.random(n) < label_noise
            y_obs[flip] = rng.integers(0, self.C, size=flip.sum())
        if self.mu is None:
            self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        X = ((x - self.mu) / self.sd).astype(np.float32)
        return Split(X, y_obs.astype(np.int64), y.astype(np.int64), bayes.astype(np.float32))


def make_world(spec: WorldSpec | None = None) -> World:
    spec = spec or WorldSpec()
    rng = np.random.default_rng(spec.seed)
    C, d = spec.n_classes, spec.d
    supers = rng.standard_normal((spec.n_super, d)) * spec.s_super
    cls = np.repeat(supers, spec.per_super, axis=0) + rng.standard_normal((C, d)) * spec.s_class
    centres = cls[:, None, :] + rng.standard_normal((C, spec.modes, d)) * spec.s_mode
    R1, _ = np.linalg.qr(rng.standard_normal((d, d)))
    # spectral norm of R2 is 1.5, so warp*|R2| = 0.75 < 1 => the warp is a bijection
    U, _, Vt = np.linalg.svd(rng.standard_normal((d, d)))
    R2 = U @ Vt * 1.5
    w = World(spec, centres, R1, R2)
    # fix the standardization on a large reference draw
    w.sample(20000, np.random.default_rng(spec.seed + 1))
    return w


@dataclass
class Data:
    """The standard splits every experiment draws from."""
    world: World
    train: Split             # labeled data (teacher's and the default student's)
    pool: Split              # extra in-distribution inputs; labels unused ("unlabeled")
    test: Split


def make_data(n_train: int = 8000, n_pool: int = 20000, n_test: int = 10000,
              label_noise: float = 0.0, spec: WorldSpec | None = None, seed: int = 0) -> Data:
    w = make_world(spec)
    rng = np.random.default_rng(1000 + seed)
    return Data(w, w.sample(n_train, rng, label_noise), w.sample(n_pool, rng),
                w.sample(n_test, rng))
