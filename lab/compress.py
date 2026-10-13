"""Ways to make a trained network smaller. Every function returns a *new* net
(the input is never modified) so experiments can branch freely.

    unstructured  magnitude_prune          zero individual weights (sparse)
    structured    neuron_importance +      delete whole neurons (really smaller,
                  structured_prune         like torch-pruning on YOLO channels)
    precision     quantize                 fewer bits per weight (fake-quant)
                  kmeans_share             k shared values per layer (non-uniform)
    factorize     low_rank                 W ~= A B via truncated SVD
    depth         drop_layers              delete whole hidden layers
    size          storage_bits             what a net costs to store, in bits
"""

from __future__ import annotations

import numpy as np

from .nn import MLP
from .train import softmax

F = np.float32


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
def storage_bits(m: MLP, bits: int = 32, group: int | None = None, sparse: bool = False, *,
                 per_row: bool = False, codebook: int = 0, fp32_last: bool = False) -> int:
    """Bytes-on-disk proxy, in bits: weights x ``bits``, + fp16 scales, + fp32
    biases. The experiments' ``kbits`` columns and every method run use it.

    The format is passed in, not read off the weights: quantization rounds
    small weights to exactly zero, so counting nonzeros would make a quantized
    net look pruned. Dense storage counts every position. ``sparse`` (for
    pruned nets) counts only the nonzero weights, at ``bits`` each, plus a
    1-bit-per-position occupancy bitmap that says where they go.

    Scales (only when ``bits < 32``) are one per tensor (``group=None``), one
    per ``group`` consecutive weights of each tensor (``fake_quantize``'s
    layout, which pads the last group), or with ``per_row`` one per output
    row, or per row x ``group`` input columns (``layerwise.quant_rows``'s
    layout). ``codebook=k`` is k-means weight sharing instead: a k-entry fp32
    codebook per layer and no scales. ``fp32_last`` keeps the output layer in
    fp32, as the layer-wise quantizers do (``skip_last``)."""
    total = sum(b.size for b in m.b) * 32
    last = m.n_layers - 1
    for i in range(m.n_layers):
        w = m.eff(i)
        size = w.size
        lb = 32 if fp32_last and i == last else bits
        total += (int((w != 0).sum()) if sparse else size) * lb
        if sparse:
            total += size
        if lb >= 32:
            continue
        if codebook:
            total += codebook * 32
        elif per_row:
            din, dout = w.shape
            total += dout * (-(-din // group) if group else 1) * 16
        else:
            total += (-(-size // group) if group else 1) * 16
    return int(total)


def kbits(m: MLP, *args, **kw) -> float:
    """``storage_bits`` in kilobits, with the format the net itself declares by
    default: a net that still fake-quantizes (QAT) is stored at its ``qbits``
    and ``qgroup``, a net with pruning masks is stored sparse. A net whose
    quantization was baked in (``quantize``, ``kmeans_share``, the layer-wise
    quantizers) or that was pruned without masks (Wanda, SparseGPT) no longer
    says so, and its caller passes the format."""
    if not args and not kw:
        kw = dict(bits=m.qbits or 32, group=m.qgroup,
                  sparse=any(mk is not None for mk in m.masks))
    return storage_bits(m, *args, **kw) / 1000


# ---------------------------------------------------------------------------
# unstructured
# ---------------------------------------------------------------------------
def magnitude_prune(net: MLP, sparsity: float, global_: bool = True,
                    spare_last: bool = False) -> MLP:
    """Zero the smallest-|w| weights until ``sparsity`` of *all* weights (not of
    the ones still alive) are zero, so calling it repeatedly with a rising
    target (gradual pruning, lottery-ticket rounds) does not compound.
    ``global_`` ranks across all layers (lets big redundant layers absorb more
    of the cut); otherwise each layer gets the same sparsity."""
    net = net.copy()
    layers = [i for i in range(net.n_layers) if not (spare_last and i == net.n_layers - 1)]

    def mask_for(mags: np.ndarray, flat: np.ndarray) -> np.ndarray:
        k = int(round(sparsity * flat.size))
        if k == 0:
            return mags != 0 if (flat == 0).any() else np.ones_like(mags, bool)
        thr = np.partition(flat, k - 1)[k - 1]
        return mags > thr

    if global_:
        flat = np.concatenate([np.abs(net.eff(i)).ravel() for i in layers])
        for i in layers:
            net.masks[i] = mask_for(np.abs(net.eff(i)), flat).astype(F)
    else:
        for i in layers:
            w = np.abs(net.eff(i))
            net.masks[i] = mask_for(w, w.ravel()).astype(F)
    net.apply_masks()
    return net


def sparsity(net: MLP) -> float:
    tot = sum(w.size for w in net.W)
    nz = sum((net.eff(i) != 0).sum() for i in range(net.n_layers))
    return 1 - nz / tot


# ---------------------------------------------------------------------------
# structured (neurons)
# ---------------------------------------------------------------------------
def neuron_importance(net: MLP, X: np.ndarray, y: np.ndarray | None = None,
                      method: str = "l1", rng: np.random.Generator | None = None,
                      group: np.ndarray | None = None) -> list[np.ndarray]:
    """One score per hidden neuron, per hidden layer (higher = keep).

    l1        -- L1 norm of the neuron's incoming weights (torch-pruning's
                 GroupMagnitudeImportance, the criterion the YOLOv8n cockpit uses)
    act       -- mean |activation| on data
    taylor    -- |a * dL/da| summed over data: first-order estimate of the loss
                 change if the neuron is removed (data- and task-aware)
    random    -- control
    ``group`` (C x K 0/1 matrix) makes 'taylor' task-aware: the loss becomes CE
    in the grouped label space (e.g. bird-vs-rest), asking "which neurons matter
    for *this* task?" rather than for all 20 classes.
    """
    n_hidden = net.n_layers - 1
    if method == "random":
        rng = rng or np.random.default_rng(0)
        return [rng.random(net.sizes[i + 1]) for i in range(n_hidden)]
    if method == "l1":
        return [np.abs(net.W[i]).sum(0) for i in range(n_hidden)]
    z, hid, cache = net.forward(X, cache=True)
    if method == "act":
        return [np.abs(h).mean(0) for h in hid]
    if method == "taylor":
        p = softmax(z)
        if group is None:
            dz = (p - np.eye(z.shape[1], dtype=F)[y]) / len(z)
        else:  # d/dz of -log q[yg], q = p @ G:  p - p * G[:, yg] / q[yg]
            yg = group[y].argmax(1)
            Gy = group[:, yg].T
            qy = (p * Gy).sum(1, keepdims=True)
            dz = (p - p * Gy / qy) / len(z)
        hg: list = []
        net.backward(cache, dz, hidden_grads=hg)
        return [np.abs(h * g).sum(0) for h, g in zip(hid, hg)]
    raise ValueError(method)


def structured_prune(net: MLP, keep: float | list[int], scores: list[np.ndarray]) -> MLP:
    """Keep the top-scoring neurons in every hidden layer. ``keep`` is a
    fraction (same for every layer) or an explicit per-layer width."""
    net = net.copy()
    for i in range(net.n_layers - 1):
        k = keep[i] if isinstance(keep, (list, tuple)) else max(1, int(round(net.sizes[i + 1] * keep)))
        idx = np.argsort(-scores[i])[:k]
        net.keep_neurons(i, idx)
    return net


def collapse_head(net: MLP, G: np.ndarray) -> MLP:
    """Shrink the output layer to a coarser label space (C -> K): each new
    output starts as the mean of its member classes' weights. Exact for
    singleton groups (e.g. 'bird' stays bird); an init for merged ones."""
    net = net.copy()
    norm = G / G.sum(0, keepdims=True)
    net.W[-1] = (net.W[-1] @ norm).astype(F)
    net.b[-1] = (net.b[-1] @ norm).astype(F)
    net.masks[-1] = None
    net.sizes[-1] = G.shape[1]
    return net


# ---------------------------------------------------------------------------
# precision
# ---------------------------------------------------------------------------
def quantize(net: MLP, bits: int, group: int | None = None) -> MLP:
    """Post-training quantization: bake fake-quantized weights in."""
    net = net.copy()
    net.qbits, net.qgroup = bits, group
    net.W = [net.eff(i) for i in range(net.n_layers)]
    net.qbits = None
    return net


def kmeans_share(net: MLP, k: int, iters: int = 25, seed: int = 0) -> MLP:
    """Deep-Compression weight sharing: each layer's weights snap to one of
    ``k`` centroids (k-means on the 1-D weight values; linear init). Storage is
    log2(k) bits/weight + a k-entry codebook. Non-uniform, unlike ``quantize``."""
    net = net.copy()
    for i in range(net.n_layers):
        w = net.W[i].reshape(-1)
        nz = w != 0 if net.masks[i] is not None else np.ones_like(w, bool)
        v = w[nz]
        c = np.linspace(v.min(), v.max(), k)
        for _ in range(iters):
            a = np.abs(v[:, None] - c[None]).argmin(1)
            for j in range(k):
                if (a == j).any():
                    c[j] = v[a == j].mean()
        a = np.abs(v[:, None] - c[None]).argmin(1)
        out = w.copy()
        out[nz] = c[a]
        net.W[i] = out.reshape(net.W[i].shape).astype(F)
    return net


# ---------------------------------------------------------------------------
# factorization
# ---------------------------------------------------------------------------
def low_rank(net: MLP, rank: int, layers: list[int] | None = None) -> MLP:
    """Replace W (in x out) by A (in x r) @ B (r x out) from a truncated SVD,
    implemented as two layers with no nonlinearity in between. Only applied
    where it actually saves parameters: r * (in + out) < in * out."""
    src = net
    if any(src.res) or any(m is not None for m in src.masks) or src.qbits is not None:
        raise ValueError("low_rank expects a dense, unmasked, unquantized, non-residual net")
    layers = layers if layers is not None else list(range(src.n_layers))
    sizes, W, b, lin = [src.sizes[0]], [], [], []
    for i in range(src.n_layers):
        w = src.W[i]
        din, dout = w.shape
        if i in layers and rank * (din + dout) < din * dout:
            U, S, Vt = np.linalg.svd(w, full_matrices=False)
            r = rank
            A = (U[:, :r] * np.sqrt(S[:r])).astype(F)
            B = (np.sqrt(S[:r])[:, None] * Vt[:r]).astype(F)
            W += [A, B]; b += [np.zeros(r, F), src.b[i].copy()]
            sizes += [r, dout]; lin += [True, False]
        else:
            W.append(w.copy()); b.append(src.b[i].copy())
            sizes.append(dout); lin.append(False)
    out = MLP(sizes, seed=0)
    out.W, out.b = W, b
    out.lin = lin
    out.res = [False] * len(W)
    out.masks = [None] * len(W)
    return out


# ---------------------------------------------------------------------------
# depth
# ---------------------------------------------------------------------------
def drop_layers(net: MLP, layers: list[int]) -> MLP:
    net = net.copy()
    for i in sorted(layers, reverse=True):
        net.drop_layer(i)
    return net
