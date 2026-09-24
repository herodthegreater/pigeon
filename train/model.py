"""Pigeon: MobileNetV3-Small specialized to bird / no-bird, with a CAM head.

The general model ends in a 1000-way classifier (576 -> 1024 -> 1000), which is
about 1.6 M of its 2.54 M parameters. Pigeon replaces it with global-average-pool
-> one linear logit. That head makes the heatmap exact and gradient-free:

    cam[h, w] = w . A[:, h, w] + b        (A = final 576 x 7 x 7 feature map)
    logit     = mean over (h, w) of cam   (the heatmap literally averages to the answer)
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FEAT_CH = 576


def general_model() -> nn.Module:
    """The untouched 1000-class ImageNet model (the "before" of the story)."""
    return mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)


def imagenet_bird_classes() -> list[int]:
    """ImageNet-1k indices that are birds (7-24, 80-100, 127-146)."""
    return [*range(7, 25), *range(80, 101), *range(127, 147)]


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class Pigeon(nn.Module):
    """Input: RGB in [0, 1], shape (B, 3, 224, 224). Output: bird logit (B,)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        self.norm = Normalize()
        self.features = base.features
        self.head = nn.Linear(FEAT_CH, 1)

    def forward(self, x: torch.Tensor, return_cam: bool = False):
        a = self.features(self.norm(x))
        logit = self.head(a.mean((2, 3))).squeeze(1)
        if not return_cam:
            return logit
        cam = torch.einsum("bchw,c->bhw", a, self.head.weight[0]) + self.head.bias
        return logit, cam


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
