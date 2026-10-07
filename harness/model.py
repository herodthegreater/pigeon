"""Apply the doubt-dial across a vision model, with snapshot/restore.

Same instrument as the LLM harness, one line different: Conv2d is now a
quantizable module, because a detector's weights live in convolutions.
Works on any nn.Module -- for an Ultralytics YOLO, pass ``yolo.model``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .quant import fake_quantize

# Conv2d is where a detector keeps almost all of its weight. Linear covers
# the detection head; Embedding is harmless to include. Norm layers and
# biases are tiny and precision-sensitive, so we leave them at native.
QUANTIZABLE = (nn.Conv2d, nn.Linear, nn.Embedding)


def iter_weight_modules(model):
    """Yield ``(name, module)`` for every quantizable weight-bearing module."""
    for name, module in model.named_modules():
        if isinstance(module, QUANTIZABLE) and getattr(module, "weight", None) is not None:
            yield name, module


def snapshot(model) -> dict[str, torch.Tensor]:
    """Clone every quantizable weight (and bias, which masking zeroes) so we
    can restore the native model."""
    snap = {}
    for name, mod in iter_weight_modules(model):
        snap[name] = mod.weight.detach().clone()
        if getattr(mod, "bias", None) is not None:
            snap[name + ".bias"] = mod.bias.detach().clone()
    return snap


def restore(model, snap: dict[str, torch.Tensor]) -> None:
    """Copy the native weights back in, undoing any quantization or masking."""
    for name, mod in iter_weight_modules(model):
        mod.weight.data.copy_(snap[name])
        if name + ".bias" in snap:
            mod.bias.data.copy_(snap[name + ".bias"])


@torch.no_grad()
def apply_uniform(
    model,
    bits: int,
    group_size: int | None = None,
    skip: tuple[str, ...] = (),
) -> int:
    """Fake-quantize every quantizable weight to ``bits``. Returns how many
    modules were touched. ``skip`` matches on substrings of the module name
    (e.g. ``(".dfl.",)`` to spare the head's fixed DFL bin positions, or
    ``("model.22",)`` to spare the whole YOLOv8n head).
    """
    touched = 0
    for name, mod in iter_weight_modules(model):
        if any(s in name for s in skip):
            continue
        mod.weight.data.copy_(fake_quantize(mod.weight.data, bits, group_size))
        touched += 1
    return touched
