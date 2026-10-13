"""The precision lab -- the doubt-dial with the knobs real quantizers expose.

`quant.py` is the minimal dial (bits + a flat group size), from the LLM
harness. This module adds what you need to *experiment* with
precision instead of just watching it collapse:

- **granularity** -- who shares a scale: the whole tensor, each output channel,
  or small groups inside a channel. Finer = less error, but every scale is
  metadata you pay for, so we report *effective* bits/weight honestly.
- **levels** -- ``int`` (evenly spaced, the classic) or ``fp`` (a tiny
  floating-point grid like FP4's E2M1: dense near zero, sparse at the tails,
  which matches bell-shaped weights).
- **clip** -- ``max`` (scale to the largest weight) or ``mse`` (shrink the
  range to minimise weight error, clipping outliers).
- **per-stage bits** -- mixed precision: keep load-bearing stages wide, crush
  the rest. This is what makes 4-bit survivable on a small detector.

All of it is fake-quantization: values are rounded to what the format could
store, shapes/dtypes never change, and accuracy measured here transfers to a
real export that uses the same scheme.
"""

from __future__ import annotations

import math

import torch

from .mask import top_stage
from .model import iter_weight_modules

SCALE_BITS = 16    # each shared scale is stored as fp16 -- this is the metadata tax
NATIVE_BITS = 16   # "native" = the fp16 checkpoint
ALWAYS_SKIP = ("dfl",)  # YOLO's DFL conv is a fixed arange(16) decoder, not a learned weight

# Minifloat (sign + exponent + mantissa) layouts for the `fp` level grid.
FP_FORMATS = {8: (4, 3), 7: (3, 3), 6: (3, 2), 5: (2, 2), 4: (2, 1), 3: (1, 1), 2: (1, 0)}


def _int_levels(bits: int) -> int:
    return (1 << (bits - 1)) - 1  # symmetric signed: 4 bits -> +/-7


def fp_grid(bits: int) -> torch.Tensor:
    """Every non-negative magnitude a sign+E+M minifloat can store, normalised so
    the largest is 1.0 (no inf/nan codes). 4 bits -> E2M1 = {0,.5,1,1.5,2,3,4,6}/6."""
    e, m = FP_FORMATS[bits]
    bias = (1 << (e - 1)) - 1 if e > 1 else 0
    vals = set()
    for ex in range(1 << e):
        for man in range(1 << m):
            if ex == 0:   # subnormal
                v = (man / (1 << m)) * 2.0 ** (1 - bias)
            else:
                v = (1 + man / (1 << m)) * 2.0 ** (ex - bias)
            vals.add(v)
    g = torch.tensor(sorted(vals))
    return g / g.max()


def _rows(w: torch.Tensor, granularity) -> tuple[torch.Tensor, int]:
    """Reshape `w` into rows that each share one scale. Returns (rows, pad).

    'tensor' -> 1 row; 'channel' -> one row per output channel; int g -> groups
    of g consecutive weights *within* each output channel (how GPTQ/AWQ/MX
    formats group), padding the tail group with zeros.
    """
    if granularity == "tensor":
        return w.reshape(1, -1), 0
    per_out = w.reshape(w.shape[0], -1)
    if granularity == "channel":
        return per_out, 0
    g = int(granularity)
    cols = per_out.shape[1]
    if g >= cols:
        return per_out, 0
    pad = (-cols) % g
    if pad:
        per_out = torch.cat([per_out, per_out.new_zeros(per_out.shape[0], pad)], 1)
    return per_out.reshape(-1, g), pad


def _unrows(q: torch.Tensor, w: torch.Tensor, granularity, pad: int) -> torch.Tensor:
    if granularity in ("tensor", "channel") or not pad:
        return q.reshape(w.shape)
    return q.reshape(w.shape[0], -1)[:, :-pad].reshape(w.shape)


def n_scales(w: torch.Tensor, granularity) -> int:
    if granularity == "tensor":
        return 1
    if granularity == "channel":
        return w.shape[0]
    cols = w[0].numel()
    return w.shape[0] * math.ceil(cols / int(granularity))


def _snap(rows: torch.Tensor, scale: torch.Tensor, bits: int, levels: str) -> torch.Tensor:
    if levels == "fp":
        grid = fp_grid(bits).to(rows)
        grid = torch.cat([-grid.flip(0)[:-1], grid])  # signed, one zero
        x = (rows / scale).clamp(-1, 1)
        idx = torch.bucketize(x, (grid[1:] + grid[:-1]) / 2)
        return grid[idx] * scale
    qmax = _int_levels(bits)
    return torch.clamp(torch.round(rows / (scale / qmax)), -qmax, qmax) * (scale / qmax)


def _quant_rows(rows: torch.Tensor, bits: int, levels: str, clip: str,
                real: torch.Tensor | None = None) -> torch.Tensor:
    """``real``: how many entries of each row are weights rather than the zero
    padding `_rows` adds to a channel's short tail group (None: all of them)."""
    if bits == 1:  # binary: sign * mean|w| per row (no zero, no range to clip)
        n = real if real is not None else rows.shape[1]
        s = rows.abs().sum(1, keepdim=True) / n
        return torch.where(rows >= 0, s, -s)
    amax = rows.abs().amax(1, keepdim=True).clamp_min(1e-12)
    if clip != "mse":
        return _snap(rows, amax, bits, levels)
    best = best_err = None
    for a in torch.linspace(0.3, 1.0, 29):
        dq = _snap(rows, amax * a, bits, levels)
        err = ((dq - rows) ** 2).sum(1, keepdim=True)
        if best is None:
            best, best_err = dq, err
        else:
            better = err < best_err
            best = torch.where(better, dq, best)
            best_err = torch.where(better, err, best_err)
    return best


@torch.no_grad()
def quantize_weight(w: torch.Tensor, bits: int, granularity="channel",
                    levels: str = "int", clip: str = "max") -> torch.Tensor:
    """Fake-quantize one weight tensor. bits >= 16 is a no-op."""
    if bits >= NATIVE_BITS:
        return w.clone()
    if levels == "fp" and bits not in FP_FORMATS:
        levels = "int"
    rows, pad = _rows(w.float(), granularity)
    real = None
    if pad:  # the last group of every output channel carries `pad` zeros
        g = rows.shape[1]
        real = torch.full((rows.shape[0], 1), float(g), device=rows.device)
        real.view(w.shape[0], -1)[:, -1] = g - pad
    q = _quant_rows(rows, bits, levels, clip, real)
    return _unrows(q, w, granularity, pad).to(w.dtype)


def weight_cost_bits(w: torch.Tensor, bits: int, granularity) -> float:
    """Storage cost in bits: the weights themselves plus their shared scales."""
    if bits >= NATIVE_BITS:
        return w.numel() * NATIVE_BITS
    return w.numel() * bits + n_scales(w, granularity) * SCALE_BITS


# --------------------------------------------------------------------------- #
# whole-model application


def precision_stages(model) -> list[dict]:
    """Every stage that holds quantizable weights, with its share of them.
    Unlike the pruning grid this INCLUDES the detection head -- it is often the
    most precision-sensitive part of the network."""
    count: dict[str, int] = {}
    for name, mod in iter_weight_modules(model):
        st = top_stage(name)
        if st is None or any(s in name for s in ALWAYS_SKIP):
            continue
        count[st] = count.get(st, 0) + mod.weight.numel()
    total = sum(count.values()) or 1
    return [{"stage": st, "weights": n, "weight_frac": n / total}
            for st, n in sorted(count.items(), key=lambda kv: int(kv[0].split(".")[1]))]


@torch.no_grad()
def apply_precision(model, bits: int = 16, stage_bits: dict[str, int] | None = None,
                    granularity="channel", levels: str = "int", clip: str = "max",
                    only: str | None = None) -> dict:
    """Quantize every weight in-place. `stage_bits` overrides `bits` per stage
    (mixed precision). `only` quantizes a single stage (sensitivity probing).
    Returns storage accounting: weight bits, scale bits, effective bits/weight."""
    stage_bits = stage_bits or {}
    n = weight_bits = scale_bits = 0
    for name, mod in iter_weight_modules(model):
        if any(s in name for s in ALWAYS_SKIP):
            continue
        st = top_stage(name)
        if only is not None and st != only:
            continue
        b = int(stage_bits.get(st, bits))
        w = mod.weight.data
        if b < NATIVE_BITS:
            mod.weight.data.copy_(quantize_weight(w, b, granularity, levels, clip))
            scale_bits += n_scales(w, granularity) * SCALE_BITS
        weight_bits += w.numel() * min(b, NATIVE_BITS)
        n += w.numel()
    return {"weights": n, "weight_bits": weight_bits, "scale_bits": scale_bits,
            "eff_bits": (weight_bits + scale_bits) / max(n, 1)}


@torch.no_grad()
def inspect_weights(w: torch.Tensor, bits: int, granularity="channel",
                    levels: str = "int", clip: str = "max", row: int = 0,
                    max_points: int = 144) -> dict:
    """The microscope: one scale-sharing row of a tensor, before and after, plus
    whole-tensor damage stats. Lets you *see* what N bits does to weights."""
    wq = quantize_weight(w, bits, granularity, levels, clip)
    rows, _ = _rows(w.float(), granularity)
    qrows, _ = _rows(wq.float(), granularity)
    row = int(row) % rows.shape[0]
    orig, snapped = rows[row][:max_points], qrows[row][:max_points]

    if bits >= NATIVE_BITS:
        grid = []
    elif bits == 1:
        s = float(rows[row].abs().mean())
        grid = [-s, s]
    else:
        grid = sorted({round(float(v), 7) for v in qrows[row].unique()})
        amax = float(rows[row].abs().max())
        if clip == "max" and amax > 0:  # the full representable grid for this row
            if levels == "fp" and bits in FP_FORMATS:
                g = fp_grid(bits) * amax
                grid = sorted({round(float(v), 7) for v in torch.cat([-g, g])})
            else:
                q = _int_levels(bits)
                grid = [amax * k / q for k in range(-q, q + 1)]

    wf, wqf = w.float().reshape(-1), wq.float().reshape(-1)
    nonzero = wf != 0
    err = ((wf - wqf) ** 2).mean()
    sig = (wf ** 2).mean()
    return {
        "orig": [round(float(v), 6) for v in orig],
        "quant": [round(float(v), 6) for v in snapped],
        "grid": grid,
        "row": row,
        "rows": rows.shape[0],
        "row_len": int(rows.shape[1]),
        "sqnr_db": round(float(10 * torch.log10(sig / err.clamp_min(1e-20))), 1) if bits < NATIVE_BITS else None,
        "zeroed_pct": round(100 * float(((wqf == 0) & nonzero).sum() / nonzero.sum().clamp_min(1)), 1),
        "distinct": int(wqf.unique().numel()),
        "numel": int(wf.numel()),
        "eff_bits": round(weight_cost_bits(w, bits, granularity) / w.numel(), 2),
    }


# --------------------------------------------------------------------------- #
# regrow with the quantizer in the loop (quantization-aware training)


class QATConv2d(torch.nn.Conv2d):
    """A Conv2d that fake-quantizes its own weight on every forward pass, with a
    straight-through gradient: the loss sees the quantized weights, the update
    lands on the full-precision ones. Fine-tuning through this teaches the
    network to live on the coarse grid -- the 'regrow' lever for precision.

    Top-level class so checkpoints containing it pickle and reload normally.
    """

    q_bits: int = 16
    q_granularity = "channel"
    q_levels: str = "int"
    q_clip: str = "max"   # class default keeps checkpoints saved before clip was stored loading

    def forward(self, x):
        w = self.weight
        if self.q_bits < NATIVE_BITS:
            # the same grid the recipe is scored on (cockpit_map), clip search included;
            # clip="mse" searches 29 clip ratios per row, about 3x the forward time on CPU
            wq = quantize_weight(w.detach(), self.q_bits, self.q_granularity, self.q_levels, self.q_clip)
            w = w + (wq - w).detach()
        return self._conv_forward(x, w, self.bias)


def enable_qat(model, bits: int = 16, stage_bits: dict[str, int] | None = None,
               granularity="channel", levels: str = "int", clip: str = "max") -> int:
    """Swap every quantizable Conv2d (in place, keeping its weights) for a
    QATConv2d at its recipe's bit-width. Use on the *unfused* training model.

    Per-tensor scales don't survive BN folding (BN rescales each channel), so
    QAT uses per-channel scales in that case; per-channel and in-channel groups
    fold exactly, so what is learned here is what the fused export will see.
    Returns the number of convs put in the loop."""
    stage_bits = stage_bits or {}
    if granularity == "tensor":
        granularity = "channel"
    n = 0
    for name, mod in iter_weight_modules(model):
        if type(mod) is not torch.nn.Conv2d or any(s in name for s in ALWAYS_SKIP):
            continue
        b = int(stage_bits.get(top_stage(name), bits))
        if b >= NATIVE_BITS:
            continue
        mod.__class__ = QATConv2d
        mod.q_bits, mod.q_granularity, mod.q_levels, mod.q_clip = b, granularity, levels, clip
        n += 1
    return n
