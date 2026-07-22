"""Real structured channel removal for YOLOv8, via torch-pruning.

The catch we discovered the hard way: torch-pruning's tracer can't follow the
`chunk()` inside Ultralytics' C2f block, so a generic MetaPruner prunes *zero*
channels. The fix (from torch-pruning's official yolov8 recipe) is to swap each
C2f for a `C2f_v2` that splits that chunk into two explicit conv branches,
making the graph traceable. Then a normal structured prune works.

This is the Export/Bake engine: it produces a genuinely smaller, faster model.
Accuracy drops after a hard cut and is recovered by fine-tuning (bonsai regrow).

Base model is YOLOv8n: YOLO11's C3k2/C2PSA blocks have no equivalent recipe.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics.nn.modules import Bottleneck, C2f, Conv, Detect


class C2f_v2(nn.Module):
    """C2f with cv1's chunk() split into two explicit conv branches (cv0, cv1)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def infer_shortcut(bottleneck: Bottleneck) -> bool:
    return bool(getattr(bottleneck, "add", False))


def transfer_weights(c2f: C2f, c2f_v2: C2f_v2) -> None:
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    old_weight = state_dict["cv1.conv.weight"]
    half = old_weight.shape[0] // 2
    state_dict_v2["cv0.conv.weight"] = old_weight[:half]
    state_dict_v2["cv1.conv.weight"] = old_weight[half:]

    for bn_key in ["weight", "bias", "running_mean", "running_var"]:
        old_bn = state_dict[f"cv1.bn.{bn_key}"]
        state_dict_v2[f"cv0.bn.{bn_key}"] = old_bn[:half]
        state_dict_v2[f"cv1.bn.{bn_key}"] = old_bn[half:]

    for key in state_dict:
        if not key.startswith("cv1."):
            state_dict_v2[key] = state_dict[key]

    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and "_" not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)


def replace_c2f_with_c2f_v2(module: nn.Module) -> None:
    """In-place: swap every C2f in `module` for a traceable C2f_v2."""
    for name, child in module.named_children():
        if isinstance(child, C2f):
            shortcut = infer_shortcut(child.m[0])
            v2 = C2f_v2(
                child.cv1.conv.in_channels, child.cv2.conv.out_channels,
                n=len(child.m), shortcut=shortcut,
                g=child.m[0].cv2.conv.groups,
                e=child.c / child.cv2.conv.out_channels,
            )
            transfer_weights(child, v2)
            setattr(module, name, v2)
        else:
            replace_c2f_with_c2f_v2(child)


def make_prunable(yolo) -> None:
    """Prepare an Ultralytics YOLO (v8) model for structured pruning."""
    replace_c2f_with_c2f_v2(yolo.model)


def structured_prune(yolo, ratio: float, example: torch.Tensor | None = None) -> None:
    """Really remove `ratio` of channels (global L2), sparing the Detect head.
    Call `make_prunable(yolo)` once before the first prune.
    """
    model = yolo.model
    device = next(model.parameters()).device
    if example is None:
        example = torch.randn(1, 3, 640, 640)
    example = example.to(device)

    ignored = [m for m in model.modules() if isinstance(m, Detect)]
    was_training = model.training
    model.train()  # clean, traceable forward (head returns raw feature maps)
    Pruner = getattr(tp.pruner, "GroupNormPruner", tp.pruner.MetaPruner)
    pruner = Pruner(
        model, example,
        importance=tp.importance.GroupMagnitudeImportance(),
        iterative_steps=1, pruning_ratio=ratio, ignored_layers=ignored,
    )
    pruner.step()
    model.train(was_training)
