"""Core fake-quantization -- the 'doubt-dial'.

We never change a tensor's dtype or shape. We *round* each weight to the
nearest value an N-bit encoding could store, then leave it in its original
float dtype. A normal forward pass through these fake-quantized weights
produces numerically identical outputs to a real N-bit-store-then-dequantize
path, so task quality measured here transfers exactly to a real quantized
export -- provided the scheme matches.

Design note we surfaced while building: compression comes from *sharing a
scale* across a group of weights. A per-weight scale is a no-op -- a single
value rounds back to itself. So bit-width can vary per weight, but the scale
is computed per *group* (`group_size`) or per tensor. `group_size == 1` makes
quantization a no-op by construction; that is expected, not a bug.
"""

from __future__ import annotations

import torch


def _pos_levels(bits: int) -> int:
    """Number of positive levels for symmetric signed quant (e.g. 8 -> 127)."""
    return (1 << (bits - 1)) - 1


def fake_quantize(
    w: torch.Tensor,
    bits: int,
    group_size: int | None = None,
) -> torch.Tensor:
    """Return a fake-quantized copy of ``w`` at ``bits`` bits.

    - ``bits >= 16``: returned unchanged (top of the dial / native).
    - ``bits == 1``: binary sign scheme, ``w_hat = sign(w) * mean(|w|)`` per group.
    - ``2..15``: symmetric linear quant with a per-group scale.

    ``group_size`` is the number of consecutive elements (along the flattened
    tensor) that share one scale. ``None`` -> the whole tensor shares one scale.
    Smaller groups = finer scales = less error (but more scale metadata in a
    real format). ``group_size == 1`` is a deliberate no-op.
    """
    if bits < 1:
        raise ValueError("bits must be >= 1")
    if bits >= 16:
        return w.clone()

    orig_shape = w.shape
    orig_dtype = w.dtype
    flat = w.reshape(-1).float()
    n = flat.numel()

    if group_size is None or group_size <= 0 or group_size >= n:
        groups = flat.unsqueeze(0)  # 1 x N: one shared scale
        pad = 0
    else:
        pad = (-n) % group_size
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        groups = flat.reshape(-1, group_size)  # G x group_size

    if bits == 1:
        # mean |w| over the real weights: the last group's `pad` zeros don't count
        real = torch.full((groups.shape[0], 1), float(groups.shape[1]), device=groups.device)
        real[-1] = groups.shape[1] - pad
        scale = (groups.abs().sum(dim=1, keepdim=True) / real).clamp_min(1e-12)
        q = torch.sign(groups)
        q = torch.where(q == 0, torch.ones_like(q), q)  # map exact zeros to +1
        deq = q * scale
    else:
        qmax = _pos_levels(bits)
        scale = groups.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
        q = torch.clamp(torch.round(groups / scale), -qmax, qmax)
        deq = q * scale

    out = deq.reshape(-1)
    if pad:
        out = out[:-pad]
    return out.reshape(orig_shape).to(orig_dtype)


def quantization_error(
    w: torch.Tensor,
    bits: int,
    group_size: int | None = None,
) -> float:
    """Mean-squared error introduced by fake-quantizing ``w``.

    A cheap proxy for 'how much does this tensor care about precision' -- the
    first, dumbest importance signal for the reverse-training loop.
    """
    wq = fake_quantize(w, bits, group_size)
    return torch.mean((w.float() - wq.float()) ** 2).item()
