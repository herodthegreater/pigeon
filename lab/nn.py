"""A tiny, dependency-free MLP with hand-written backprop.

Small enough to read in one sitting, fast enough on a CPU to train hundreds of
models per experiment. It supports exactly the knobs the compression and
distillation experiments need:

- **masks** (unstructured sparsity): weights are multiplied by a 0/1 mask in
  the forward pass and gradients are masked too, so pruned weights stay dead.
- **fake quantization** (``qbits``/``qgroup``): the forward pass uses weights
  rounded the same way ``harness/quant.py`` rounds a detector's convs; the
  backward pass is straight-through (STE), which is what quantization-aware
  training means.
- **residual hidden layers** (``res=True``): ``h <- h + relu(h W + b)``. Used to
  show why deleting a layer is graceful in residual nets and fatal in plain ones.
- **hidden-state gradients** (``dhid``): extra gradient injected at any hidden
  layer's output, which is how feature/hint/relational distillation losses
  plug in without an autograd engine.
- **input gradients**: ``backward`` returns dL/dX, used to invert a teacher
  into synthetic training data (data-free distillation).
"""

from __future__ import annotations

import copy

import numpy as np

F = np.float32


def fake_quantize(w: np.ndarray, bits: int, group_size: int | None = None) -> np.ndarray:
    """NumPy twin of ``harness.quant.fake_quantize`` (same semantics)."""
    if bits >= 16:
        return w
    flat = w.reshape(-1)
    n = flat.size
    if group_size is None or group_size <= 0 or group_size >= n:
        g, pad = flat[None, :], 0
    else:
        pad = (-n) % group_size
        g = np.concatenate([flat, np.zeros(pad, w.dtype)]).reshape(-1, group_size)
    if bits == 1:
        # mean |w| over the real weights: the last group's `pad` zeros don't count
        real = np.full((g.shape[0], 1), float(g.shape[1]))
        real[-1] = g.shape[1] - pad
        scale = np.maximum(np.abs(g).sum(1, keepdims=True) / real, 1e-12)
        q = np.where(g >= 0, 1.0, -1.0)
    else:
        qmax = (1 << (bits - 1)) - 1
        scale = np.maximum(np.abs(g).max(1, keepdims=True), 1e-12) / qmax
        q = np.clip(np.round(g / scale), -qmax, qmax)
    out = (q * scale).reshape(-1)
    if pad:
        out = out[:-pad]
    return out.reshape(w.shape).astype(w.dtype)


class MLP:
    def __init__(self, sizes: list[int], seed: int = 0, res: bool = False,
                 linear: set[int] | None = None):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            std = np.sqrt(2.0 / sizes[i])
            if res and 0 < i < len(sizes) - 2:
                std *= 0.5  # keep the residual stream from blowing up
            self.W.append((rng.standard_normal((sizes[i], sizes[i + 1])) * std).astype(F))
            self.b.append(np.zeros(sizes[i + 1], F))
        n = len(self.W)
        # residual only where in/out widths match and it isn't the first/last layer
        self.res = [res and 0 < i < n - 1 and sizes[i] == sizes[i + 1] for i in range(n)]
        # layers with no nonlinearity (low-rank factors, over-parameterized products)
        self.lin = [i in (linear or ()) for i in range(n)]
        self.masks: list[np.ndarray | None] = [None] * n
        self.qbits: int | None = None
        self.qgroup: int | None = None

    # -- bookkeeping ---------------------------------------------------------
    def save_npz(self, path: str) -> None:
        """Plain-array snapshot (portable across numpy versions, unlike pickle;
        the static website ships trained models this way)."""
        arrs = {f"W{i}": w for i, w in enumerate(self.W)}
        arrs.update({f"b{i}": b for i, b in enumerate(self.b)})
        arrs.update({f"m{i}": m for i, m in enumerate(self.masks) if m is not None})
        np.savez(path, sizes=np.array(self.sizes), res=np.array(self.res),
                 lin=np.array(self.lin), qbits=np.array(self.qbits or 0),
                 qgroup=np.array(self.qgroup or 0), **arrs)

    @classmethod
    def load_npz(cls, path: str) -> "MLP":
        z = np.load(path)
        net = cls.__new__(cls)
        net.sizes = [int(v) for v in z["sizes"]]
        n = len(net.sizes) - 1
        net.W = [z[f"W{i}"] for i in range(n)]
        net.b = [z[f"b{i}"] for i in range(n)]
        net.masks = [z[f"m{i}"] if f"m{i}" in z.files else None for i in range(n)]
        net.res = [bool(v) for v in z["res"]]
        net.lin = [bool(v) for v in z["lin"]]
        net.qbits = int(z["qbits"]) or None
        net.qgroup = int(z["qgroup"]) or None
        return net

    @property
    def n_layers(self) -> int:
        return len(self.W)

    def copy(self) -> "MLP":
        return copy.deepcopy(self)

    def params(self) -> list[np.ndarray]:
        return self.W + self.b

    def n_params(self, nonzero: bool = True) -> int:
        if nonzero:
            return int(sum((self.eff(i) != 0).sum() + b.size for i, b in enumerate(self.b)))
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    def macs(self) -> int:
        """Multiply-accumulates per example, counting only nonzero weights."""
        return int(sum((self.eff(i) != 0).sum() for i in range(self.n_layers)))

    def size_bits(self, bits: int | None = None) -> int:
        bits = bits or self.qbits or 32
        return self.n_params() * bits

    def eff(self, i: int) -> np.ndarray:
        w = self.W[i]
        if self.masks[i] is not None:
            w = w * self.masks[i]
        if self.qbits is not None:
            w = fake_quantize(w, self.qbits, self.qgroup)
            if self.masks[i] is not None:  # 1-bit sign() would revive pruned zeros
                w = w * self.masks[i]
        return w

    # -- forward / backward --------------------------------------------------
    def forward(self, X: np.ndarray, cache: bool = False):
        """Returns ``(logits, hidden, cache)``. ``hidden[i]`` is the output of
        hidden layer ``i`` (post-activation, post-residual)."""
        h = X
        hidden, ins, pre, effs = [], [], [], []
        last = self.n_layers - 1
        for i in range(self.n_layers):
            w = self.eff(i)
            z = h @ w + self.b[i]
            if cache:
                ins.append(h); pre.append(z); effs.append(w)
            if i == last:
                h = z
            else:
                a = z if self.lin[i] else np.maximum(z, 0)
                h = h + a if self.res[i] else a
                hidden.append(h)
        return h, hidden, ((ins, pre, effs) if cache else None)

    def logits(self, X: np.ndarray, bs: int = 8192) -> np.ndarray:
        return np.concatenate([self.forward(X[i:i + bs])[0] for i in range(0, len(X), bs)])

    def backward(self, cache, dlogits: np.ndarray, dhid: dict[int, np.ndarray] | None = None,
                 hidden_grads: list | None = None):
        """Returns ``(grads, dX)`` where grads align with ``self.params()``.
        If ``hidden_grads`` is a list, dL/d(hidden[i]) is stored at index i."""
        ins, pre, effs = cache
        dW = [None] * self.n_layers
        db = [None] * self.n_layers
        g = dlogits
        if hidden_grads is not None:
            hidden_grads[:] = [None] * (self.n_layers - 1)
        for i in reversed(range(self.n_layers)):
            if hidden_grads is not None and i < self.n_layers - 1:
                hidden_grads[i] = g
            last_or_lin = i == self.n_layers - 1 or self.lin[i]
            gz = g if last_or_lin else g * (pre[i] > 0)
            dW[i] = ins[i].T @ gz
            db[i] = gz.sum(0)
            if self.masks[i] is not None:
                dW[i] *= self.masks[i]
            g_in = gz @ effs[i].T
            if self.res[i]:
                g_in = g_in + g
            if dhid is not None and (i - 1) in dhid:
                g_in = g_in + dhid[i - 1]
            g = g_in
        return dW + db, g

    # -- surgery --------------------------------------------------------------
    def apply_masks(self) -> None:
        for i, m in enumerate(self.masks):
            if m is not None:
                self.W[i] *= m

    def keep_neurons(self, layer: int, idx: np.ndarray) -> None:
        """Really remove neurons: keep only ``idx`` outputs of hidden layer
        ``layer`` (and the matching inputs of the next layer). The network gets
        genuinely smaller -- the numpy equivalent of torch-pruning's surgery."""
        if self.res[layer] or self.res[layer + 1]:
            raise ValueError("neuron removal across a residual stream needs a "
                             "dependency group (the torch-pruning problem); not supported")
        idx = np.sort(idx)
        self.W[layer] = self.W[layer][:, idx]
        self.b[layer] = self.b[layer][idx]
        self.W[layer + 1] = self.W[layer + 1][idx, :]
        for j in (layer, layer + 1):
            if self.masks[j] is not None:
                self.masks[j] = self.masks[j][:, idx] if j == layer else self.masks[j][idx, :]
        self.sizes[layer + 1] = len(idx)

    def drop_layer(self, layer: int) -> None:
        """Delete hidden->hidden layer ``layer`` entirely (needs equal widths)."""
        assert 0 < layer < self.n_layers - 1 and self.sizes[layer] == self.sizes[layer + 1]
        for lst in (self.W, self.b, self.res, self.masks, self.lin):
            del lst[layer]
        del self.sizes[layer + 1]

    def collapse_linear(self) -> "MLP":
        """Merge every linear (activation-free) hidden layer into the layer
        after it: ``(h W1 + b1) W2 + b2 == h (W1 W2) + (b1 W2 + b2)``. Exact.
        Used to 'collapse' an over-parameterized net back to its small shape."""
        net = self.copy()
        i = 0
        while i < net.n_layers - 1:
            if net.lin[i]:
                assert not net.res[i] and net.masks[i] is None and net.masks[i + 1] is None
                W2 = net.W[i + 1]
                net.b[i + 1] = net.b[i] @ W2 + net.b[i + 1]
                net.W[i + 1] = net.W[i] @ W2
                for lst in (net.W, net.b, net.res, net.masks, net.lin):
                    del lst[i]
                del net.sizes[i + 1]
            else:
                i += 1
        return net


class Adam:
    """AdamW with a cosine schedule driven by the caller (``set_lr``)."""

    def __init__(self, params: list[np.ndarray], lr: float = 3e-3, wd: float = 0.0,
                 betas=(0.9, 0.999), eps: float = 1e-8):
        self.p, self.lr, self.wd, self.b1, self.b2, self.eps = params, lr, wd, *betas, eps
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            if self.wd and p.ndim > 1:
                p -= self.lr * self.wd * p
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
