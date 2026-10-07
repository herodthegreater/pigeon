"""Masking = zero exactly the output channels a real Export would remove (the
plan comes from `prune.removal_plan`). Shapes stay fixed, so it is always safe
and reversible (via snapshot/restore) -- this is the live, interactive layer of
the UI. Its accuracy is the exported model's accuracy before fine-tuning; it
does not itself make the model smaller or faster.
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


def head_stage(model) -> str:
    """Name of the detection head: the last stage of an Ultralytics model
    ('model.22' on YOLOv8n, 'model.23' on YOLO11n)."""
    return f"model.{len(model.model) - 1}"


def stage_table(model, imgsz: int = 640, exclude=None):
    """One forward with hooks -> per-stage {channels, flop_frac} (MAC share),
    backbone+neck only. `exclude` defaults to the head, which Export never
    prunes either."""
    if exclude is None:
        exclude = (head_stage(model),)
    flops: dict[str, int] = {}
    hooks = []

    def mk(name):
        def h(m, inp, out):
            # one MAC per output pixel per kernel tap: stride-2 convs do 1/4 the work
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
    stage_ch: dict[str, int] = {}
    for n, m in model.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        st = top_stage(n)
        if st is None or st in exclude:
            continue
        stage_f[st] = stage_f.get(st, 0) + flops.get(n, 0)
        stage_ch[st] = max(stage_ch.get(st, 0), m.out_channels)

    # share of the WHOLE model's conv compute (head included), so a stage's
    # flop_frac times its cut fraction is a fraction of the native MACs
    total = sum(flops.values()) or 1
    return [
        {"stage": st, "channels": stage_ch[st], "flop_frac": stage_f[st] / total}
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

