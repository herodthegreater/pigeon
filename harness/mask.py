"""Masking = zero the lowest-L1 output channels of a stage's convs. Shapes stay
fixed, so it is always safe and reversible (via snapshot/restore) -- this is the
live, interactive layer of the UI. It shows REAL accuracy impact; the size/speed
numbers it implies are ESTIMATES of what a real Export removal would yield.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def top_stage(name: str) -> str | None:
    """'model.5.cv1.conv' -> 'model.5'."""
    p = name.split(".")
    if len(p) >= 2 and p[0] == "model" and p[1].isdigit():
        return f"model.{p[1]}"
    return None


def stage_table(model, imgsz: int = 640, exclude=("model.23",)):
    """One forward with hooks -> per-stage {channels, flop_frac}, backbone+neck
    only (the head is excluded from masking to keep outputs sane)."""
    flops: dict[str, int] = {}
    hooks = []

    def mk(name):
        def h(m, inp, out):
            x = inp[0]
            k = m.kernel_size[0] * m.kernel_size[1]
            flops[name] = x.shape[-2] * x.shape[-1] * m.in_channels * m.out_channels * k // m.groups
        return h

    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(mk(n)))
    was = model.training
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        model(torch.randn(1, 3, imgsz, imgsz, device=dev))
    for h in hooks:
        h.remove()
    model.train(was)

    stage_f: dict[str, int] = {}
    stage_ch: dict[str, int] = {}
    for n, m in model.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        st = top_stage(n)
        if st is None or st in exclude:
            continue
        stage_f[st] = stage_f.get(st, 0) + flops.get(n, 0)
        stage_ch[st] = max(stage_ch.get(st, 0), m.out_channels)

    total = sum(stage_f.values()) or 1
    return [
        {"stage": st, "channels": stage_ch[st], "flop_frac": stage_f[st] / total}
        for st in sorted(stage_f, key=lambda s: int(s.split(".")[1]))
    ]


def _stage_convs(model, stage):
    return [m for n, m in model.named_modules()
            if isinstance(m, nn.Conv2d) and top_stage(n) == stage]


@torch.no_grad()
def mask_stage(model, stage: str, frac: float) -> None:
    """Zero the lowest-L1 output channels (by `frac`) of every conv in `stage`."""
    if frac <= 0:
        return
    for m in _stage_convs(model, stage):
        w = m.weight.data
        out = w.shape[0]
        k = min(int(round(frac * out)), out - 1)  # always keep >=1 channel
        if k <= 0:
            continue
        l1 = w.abs().flatten(1).sum(1)
        idx = torch.argsort(l1)[:k]
        w[idx] = 0.0


@torch.no_grad()
def apply_masks(model, masks: dict[str, float]) -> None:
    for stage, frac in masks.items():
        mask_stage(model, stage, float(frac))


def estimated_flop_saving(table, masks: dict[str, float]) -> float:
    """First-order estimate of the FLOP fraction a real removal would save
    (conservative: counts each stage's own compute, not downstream coupling)."""
    frac_by_stage = {t["stage"]: t["flop_frac"] for t in table}
    return sum(frac_by_stage.get(s, 0.0) * float(f) for s, f in masks.items())
