"""Compression methods invented for large language models, run on the toy MLP.

LLM compression papers mostly work **one linear layer at a time**, using a few
hundred "calibration" inputs and no gradient training. A linear layer is a
linear layer whether it sits in a transformer or in this MLP, so the algorithms
below are the real ones, just applied to a smaller network:

    rtn            round-to-nearest, per output row (the baseline every paper beats)
    gptq           quantize column by column, pushing each rounding error onto
                   the columns not yet quantized (Frantar et al. 2022)
    awq            protect the input channels that see big activations by
                   scaling them up before rounding (Lin et al. 2023)
    smoothquant    move activation outliers into the weights so both can be
                   quantized to 8 bits (Xiao et al. 2022)
    wanda          prune by |weight| x input-activation norm (Sun et al. 2023)
    sparsegpt      prune with the same error-compensation idea as GPTQ
                   (Frantar & Alistarh 2023)
    qlora          freeze a quantized model, train small low-rank adapters on
                   top (Dettmers et al. 2023)
    block_influence  rank residual layers by how little they change their input,
                   delete the laziest (ShortGPT, Men et al. 2024)

Convention: the papers write a layer as ``y = W x`` with ``W`` shaped
(out, in). Our MLP stores ``W`` as (in, out) and computes ``h @ W``. Every
function here takes and returns the papers' (out, in) orientation, called
``Wt``, so the code reads like the papers.
"""

from __future__ import annotations

import numpy as np

from .nn import MLP, Adam
from .train import KD, FixedSource

F = np.float32


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def qmax_for(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def quant_rows(Wt: np.ndarray, bits: int, group: int | None = None) -> np.ndarray:
    """Round-to-nearest with one scale per output row (or per row x group of
    ``group`` consecutive input columns). Symmetric, like harness/quant.py."""
    qmax = qmax_for(bits)
    out = np.empty_like(Wt)
    din = Wt.shape[1]
    g = group or din
    for c0 in range(0, din, g):
        blk = Wt[:, c0:c0 + g]
        scale = np.maximum(np.abs(blk).max(1, keepdims=True), 1e-12) / qmax
        out[:, c0:c0 + g] = np.clip(np.round(blk / scale), -qmax, qmax) * scale
    return out


def layer_forward(net: MLP, i: int, h: np.ndarray) -> np.ndarray:
    """Output of layer ``i`` alone, given its input ``h``."""
    z = h @ net.eff(i) + net.b[i]
    if i == net.n_layers - 1 or net.lin[i]:
        return z
    a = np.maximum(z, 0)
    return h + a if net.res[i] else a


def apply_layerwise(net: MLP, X: np.ndarray, fn, skip_last: bool = False) -> MLP:
    """Run ``fn(Wt, layer_input) -> new Wt`` on every layer in order.

    Each layer is calibrated on the inputs it will *actually* see: the outputs
    of the layers before it after they were already compressed. That is how
    GPTQ/SparseGPT process a transformer block by block."""
    net = net.copy()
    h = X.astype(np.float64)
    for i in range(net.n_layers):
        if not (skip_last and i == net.n_layers - 1):
            net.W[i] = fn(net.W[i].T.astype(np.float64), h).T.astype(F)
        h = layer_forward(net, i, h.astype(F)).astype(np.float64)
    return net


def _hessian_inverse_chol(X: np.ndarray, damp: float) -> tuple[np.ndarray, np.ndarray]:
    """H = 2 X^T X / n (the curvature of the layer's squared output error).
    Returns the upper Cholesky factor U of H^-1 (H^-1 = U^T U) and the mask of
    input channels that never activate ("dead"; their H diagonal is 0)."""
    H = 2 * X.T @ X / len(X)
    dead = np.diag(H) == 0
    H[dead, dead] = 1
    H += damp * np.mean(np.diag(H)) * np.eye(len(H))
    U = np.linalg.cholesky(np.linalg.inv(H)).T
    return U, dead


# ---------------------------------------------------------------------------
# quantization
# ---------------------------------------------------------------------------
def rtn(Wt: np.ndarray, X: np.ndarray, bits: int = 4, group: int | None = None) -> np.ndarray:
    """Baseline: round every weight to its nearest level. Ignores the data."""
    del X
    return quant_rows(Wt, bits, group)


def gptq(Wt: np.ndarray, X: np.ndarray, bits: int = 4, group: int | None = None,
         damp: float = 0.01, act_order: bool = False) -> np.ndarray:
    """GPTQ: quantize one input column at a time. The rounding error of column
    j is spread onto the remaining columns, weighted by the inverse Hessian, so
    the layer's *output* on calibration data changes as little as possible.

    This is the Optimal Brain Surgeon update restricted to a fixed column
    order, which is what makes it cheap enough for 175B-parameter models."""
    W = Wt.copy()
    dout, din = W.shape
    U, dead = _hessian_inverse_chol(X, damp)
    W[:, dead] = 0
    perm = np.arange(din)
    if act_order:  # quantize the columns with the biggest activations first
        perm = np.argsort(-np.diag(X.T @ X))
        W = W[:, perm]
        U, _ = _hessian_inverse_chol(X[:, perm], damp)
    qmax = qmax_for(bits)
    Q = np.zeros_like(W)
    scale = np.maximum(np.abs(W).max(1), 1e-12) / qmax  # per row, from the original W
    for j in range(din):
        if group and j % group == 0:  # scales for a group use the *updated* weights
            scale = np.maximum(np.abs(W[:, j:j + group]).max(1), 1e-12) / qmax
        w = W[:, j]
        q = np.clip(np.round(w / scale), -qmax, qmax) * scale
        Q[:, j] = q
        err = (w - q) / U[j, j]
        W[:, j + 1:] -= np.outer(err, U[j, j + 1:])
    return Q[:, np.argsort(perm)]


def awq(Wt: np.ndarray, X: np.ndarray, bits: int = 4, group: int | None = None,
        grid: int = 20) -> np.ndarray:
    """AWQ: a few input channels carry large activations; their weights matter
    most. Multiply those columns by s > 1 before rounding (so they get finer
    effective resolution) and divide the input by s (folded into the previous
    op in a real deployment). Search s = mean|x|^a over a in [0, 1]."""
    xmag = np.abs(X).mean(0) + 1e-8
    best, best_err = None, np.inf
    ref = X @ Wt.T
    for a in np.linspace(0, 1, grid + 1):
        s = xmag ** a
        s = s / np.sqrt(s.max() * s.min())
        Wq = quant_rows(Wt * s[None, :], bits, group) / s[None, :]
        err = ((X @ Wq.T - ref) ** 2).mean()
        if err < best_err:
            best, best_err = Wq, err
    return best


# ---------------------------------------------------------------------------
# pruning
# ---------------------------------------------------------------------------
def wanda(Wt: np.ndarray, X: np.ndarray, sparsity: float = 0.5) -> np.ndarray:
    """Wanda: score = |w_ij| * ||x_j||. A small weight on a loud input can
    matter more than a big weight on a quiet one. Prune per output row; no
    weight update, no retraining."""
    score = np.abs(Wt) * np.sqrt((X ** 2).sum(0))[None, :]
    k = int(round(Wt.shape[1] * sparsity))
    out = Wt.copy()
    if k:
        idx = np.argsort(score, 1)[:, :k]
        np.put_along_axis(out, idx, 0.0, 1)
    return out


def sparsegpt(Wt: np.ndarray, X: np.ndarray, sparsity: float = 0.5, block: int = 32,
              damp: float = 0.01) -> np.ndarray:
    """SparseGPT: like GPTQ, but the 'rounding' is setting a weight to zero.
    Within each block of columns choose which weights to drop by the OBS
    saliency w^2 / [H^-1]_jj^2, then compensate the remaining columns.

    Simplification: we drop the same number of weights in every row of a
    block (so each row ends at exactly ``sparsity``); the paper picks the
    lowest-saliency weights across all rows of the block jointly."""
    W = Wt.copy()
    dout, din = W.shape
    U, dead = _hessian_inverse_chol(X, damp)
    W[:, dead] = 0
    keep = np.ones_like(W, bool)
    for b0 in range(0, din, block):
        b1 = min(b0 + block, din)
        sal = W[:, b0:b1] ** 2 / np.diag(U)[b0:b1][None, :] ** 2
        # per-block counts that add up to exactly round(din * sparsity) per row
        k = int(round(b1 * sparsity)) - int(round(b0 * sparsity))
        if k:
            idx = np.argsort(sal, 1)[:, :k]
            blk = np.ones((dout, b1 - b0), bool)
            np.put_along_axis(blk, idx, False, 1)
            keep[:, b0:b1] = blk
        for j in range(b0, b1):
            w = W[:, j]
            q = w * keep[:, j]
            err = (w - q) / U[j, j]
            W[:, j] = q
            W[:, j + 1:] -= np.outer(err, U[j, j + 1:])
    return W * keep


# ---------------------------------------------------------------------------
# activation quantization + SmoothQuant
# ---------------------------------------------------------------------------
def quant_act(h: np.ndarray, bits: int) -> np.ndarray:
    """Dynamic per-token activation quantization: one scale per row (per
    example here, per token in an LLM), computed on the fly. A single huge
    channel still sets every row's scale, which is the problem SmoothQuant fixes."""
    if bits >= 16:
        return h
    qmax = qmax_for(bits)
    scale = np.maximum(np.abs(h).max(1, keepdims=True), 1e-12) / qmax
    return np.clip(np.round(h / scale), -qmax, qmax) * scale


def forward_wa(net: MLP, X: np.ndarray, abits: int) -> np.ndarray:
    """Forward pass that also quantizes the input of every layer after the
    first (W8A8-style). Weights are whatever ``net.eff`` gives; quantize them
    first (e.g. with ``apply_layerwise`` + ``rtn``) for the full W+A setting."""
    h = X
    for i in range(net.n_layers):
        if i > 0:
            h = quant_act(h, abits)
        h = layer_forward(net, i, h)
    return h


def inject_outliers(net: MLP, layer: int = 0, k: int = 3, factor: float = 30.0,
                    seed: int = 0) -> MLP:
    """Create LLM-style activation outlier channels *without changing the
    function*: multiply k neurons' outputs by ``factor`` and divide the next
    layer's matching weights by it. Because relu(c z) = c relu(z) for c > 0,
    the network computes exactly the same thing, but its activations now have
    a few huge channels, which is what breaks per-tensor 8-bit activations
    in real LLMs."""
    net = net.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(net.sizes[layer + 1], k, replace=False)
    net.W[layer][:, idx] *= factor
    net.b[layer][idx] *= factor
    net.W[layer + 1][idx, :] /= factor
    return net


def smoothquant(net: MLP, X: np.ndarray, alpha: float = 0.5) -> MLP:
    """SmoothQuant: per input channel j of layer i, s_j = max|x_j|^a / max|w_j|^(1-a).
    Divide the activation by s (fold into the previous layer: legal because
    relu is positively homogeneous) and multiply the weights by s. Same
    function; activation outliers shrink, weights absorb the difficulty."""
    net = net.copy()
    _, hid, _ = net.forward(X)   # scaling layer i's inputs never changes layer i's own inputs
    for i in range(1, net.n_layers):
        if net.res[i - 1] or net.res[i]:
            continue
        a = np.abs(hid[i - 1]).max(0)
        w = np.abs(net.W[i]).max(1) + 1e-8
        s = a ** alpha / w ** (1 - alpha)
        # a channel that never fired on calibration data has no range to fix;
        # scaling it by a tiny s would blow up its weights for unseen inputs
        silent = a < 1e-3 * a.max()
        s = np.where(silent, 1.0, np.clip(s, 1e-3, 1e3)).astype(F)
        net.W[i] = net.W[i] * s[:, None]
        net.W[i - 1] = net.W[i - 1] / s[None, :]
        net.b[i - 1] = net.b[i - 1] / s
        hid[i - 1] = hid[i - 1] / s
    return net


# ---------------------------------------------------------------------------
# QLoRA
# ---------------------------------------------------------------------------
def qlora(net: MLP, teacher: MLP, X: np.ndarray, bits: int = 3, group: int | None = 16,
          rank: int = 4, steps: int = 600, lr: float = 3e-3, bs: int = 128, seed: int = 0):
    """Freeze a quantized copy of ``net`` and learn W = Q(W0) + A B for every
    layer, with A (in x r), B (r x out) and B initialized to zero (so training
    starts exactly at the quantized model). Only A and B get gradients:
    dA = dW B^T, dB = A^T dW. Returns (model with adapters merged, #adapter params)."""
    from .compress import quantize
    base = quantize(net, bits, group)
    rng = np.random.default_rng(seed)
    A = [(rng.standard_normal((w.shape[0], rank)) / np.sqrt(w.shape[0])).astype(F) for w in base.W]
    B = [np.zeros((rank, w.shape[1]), F) for w in base.W]
    opt = Adam(A + B, lr=lr)
    work = base.copy()
    src = FixedSource(X, None, net.sizes[-1], teacher=teacher)
    kd = KD(T=2.0)
    step = 0
    while step < steps:
        for b in src.batches(rng, bs):
            for i in range(work.n_layers):
                work.W[i] = base.W[i] + A[i] @ B[i]
            z, hid, cache = work.forward(b.X, cache=True)
            _, dz, _ = kd(z, hid, b)
            grads, _ = work.backward(cache, dz)
            dW = grads[:work.n_layers]
            opt.step([g @ Bi.T for g, Bi in zip(dW, B)] + [Ai.T @ g for g, Ai in zip(dW, A)])
            step += 1
            if step >= steps:
                break
    for i in range(work.n_layers):
        work.W[i] = base.W[i] + A[i] @ B[i]
    return work, int(sum(a.size + b_.size for a, b_ in zip(A, B)))


# ---------------------------------------------------------------------------
# depth pruning (ShortGPT)
# ---------------------------------------------------------------------------
def block_influence(net: MLP, X: np.ndarray) -> dict[int, float]:
    """For every residual hidden layer i: BI = 1 - mean cos(input, output).
    A layer whose output points the same way as its input barely changes the
    residual stream, so it is the cheapest to delete."""
    out = {}
    h = X
    for i in range(net.n_layers - 1):
        nxt = layer_forward(net, i, h)
        if net.res[i]:
            cos = (h * nxt).sum(1) / (np.linalg.norm(h, axis=1) * np.linalg.norm(nxt, axis=1) + 1e-8)
            out[i] = float(1 - cos.mean())
        h = nxt
    return out


