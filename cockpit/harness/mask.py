"""Masking = zero exactly the output channels a real Export would remove (the
plan comes from `prune.removal_plan`), weights AND bias, so a masked channel
outputs SiLU(0) = 0 as if it were gone. Shapes stay fixed, so it is always safe
and reversible (via snapshot/restore) -- this is the live, interactive layer of
the UI. Its accuracy is the exported model's accuracy before fine-tuning; the
size/speed it implies (`active_cost`) is what Export would leave.
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


def head_stage(model) -> str | None:
    """The detection head's stage name ('model.22' on YOLOv8n, 'model.23' on
    YOLO11n). It is always the last module of an Ultralytics model."""
    seq = getattr(model, "model", None)
    return f"model.{len(seq) - 1}" if seq is not None and len(seq) else None


def stage_table(model, imgsz: int = 640, exclude=None):
    """One forward with hooks -> per-stage {channels, flop_frac, param_frac},
    backbone+neck only. The head is excluded: masking it would zero class
    logits and DFL box bins, and real Export never prunes it either."""
    if exclude is None:
        exclude = (head_stage(model),)
    flops: dict[str, int] = {}
    hooks = []

    def mk(name):
        def h(m, inp, out):
            # MACs happen once per OUTPUT pixel (a stride-2 conv has 4x fewer than its input)
            k = m.kernel_size[0] * m.kernel_size[1]
            flops[name] = out.shape[-2] * out.shape[-1] * m.in_channels * m.out_channels * k // m.groups
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
    stage_p: dict[str, int] = {}
    stage_ch: dict[str, int] = {}
    all_f = all_p = 0
    for n, m in model.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        all_f += flops.get(n, 0)
        all_p += m.weight.numel()
        st = top_stage(n)
        if st is None or st in exclude:
            continue
        stage_f[st] = stage_f.get(st, 0) + flops.get(n, 0)
        stage_p[st] = stage_p.get(st, 0) + m.weight.numel()
        stage_ch[st] = max(stage_ch.get(st, 0), m.out_channels)

    # Fractions are of the WHOLE model (head included), so a saving estimate
    # never overstates what cutting the backbone+neck can buy.
    return [
        {"stage": st, "channels": stage_ch[st],
         "flop_frac": stage_f[st] / (all_f or 1), "param_frac": stage_p[st] / (all_p or 1)}
        for st in sorted(stage_f, key=lambda s: int(s.split(".")[1]))
    ]


@torch.no_grad()
def apply_plan(model, plan: dict[str, list[int]]) -> None:
    """Zero the weight AND bias of each listed output channel of a fused model.
    With the bias zeroed too the channel outputs SiLU(0) = 0, exactly what its
    consumers would see if it were removed. `plan` is `prune.removal_plan`."""
    mods = dict(model.named_modules())
    for name, idx in plan.items():
        m = mods[name]
        m.weight[idx] = 0.0
        if m.bias is not None:
            m.bias[idx] = 0.0


@torch.no_grad()
def active_cost(model, imgsz: int = 256) -> tuple[int, int]:
    """(MACs, conv weights) the model would have if every masked channel were
    physically removed. One forward with hooks: an input channel whose
    activation is exactly zero everywhere (a masked producer: zero weights, zero
    bias, SiLU(0) = 0) is dead, and so is an output channel with an all-zero
    weight row and bias. Counts downstream coupling -- concats, residual adds --
    for free, so it tracks a real Export far better than per-stage shares.
    MACs are at `imgsz`; compare ratios, not absolutes."""
    macs = params = 0
    hooks = []

    def hook(m, inp, out):
        nonlocal macs, params
        x = inp[0]
        live_in = int((x.abs().sum((0, 2, 3)) > 0).sum())
        w = m.weight
        row = w.flatten(1).abs().sum(1) > 0
        if m.bias is not None:
            row = row | (m.bias != 0)
        live_out = int(row.sum())
        k = m.kernel_size[0] * m.kernel_size[1]
        cin = live_in / m.in_channels * (m.in_channels // m.groups)
        params += int(live_out * cin * k)
        macs += int(out.shape[-2] * out.shape[-1] * live_out * cin * k)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(hook))
    was = model.training
    model.eval()
    dev = next(model.parameters()).device
    try:
        model(torch.rand(1, 3, imgsz, imgsz, device=dev))
    finally:
        for h in hooks:
            h.remove()
        model.train(was)
    return macs, params
