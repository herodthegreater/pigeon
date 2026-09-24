"""In-memory Pigeon dataset + on-device augmentation.

The whole set is ~7k images at 256 px (~1.4 GB as uint8), so it lives in RAM
and augmentation runs batched on the GPU; no JPEG decoding in the training loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/pigeon"
CACHE = ROOT / "data/cache"
INPUT = 224


@dataclass
class Split:
    images: torch.Tensor  # uint8 (N, 3, 256, 256)
    labels: torch.Tensor  # float (N,)
    hard: torch.Tensor  # bool (N,)
    boxes: list[list[list[float]]]  # normalized [x0, y0, x1, y1] per bird box
    ids: list[str]

    def __len__(self):
        return len(self.ids)


def load_split(name: str) -> Split:
    CACHE.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "manifest.csv")
    df = df[df.split == name].reset_index(drop=True)
    cache = CACHE / f"{name}.pt"
    if cache.exists():
        images = torch.load(cache)
    else:
        arr = np.stack(
            [np.asarray(Image.open(DATA / name / f"{i}.jpg").convert("RGB")) for i in tqdm(df.image_id, desc=f"load {name}")]
        )
        images = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        torch.save(images, cache)
    return Split(
        images=images,
        labels=torch.tensor(df.label.values, dtype=torch.float32),
        hard=torch.tensor(df.hard.values, dtype=torch.bool),
        boxes=[json.loads(b) for b in df.boxes],
        ids=list(df.image_id),
    )


def to_input(u8: torch.Tensor) -> torch.Tensor:
    """Eval preprocessing: uint8 256 px -> float [0, 1] at 224 px (whole image, squashed)."""
    x = u8.float() / 255.0
    return F.interpolate(x, size=(INPUT, INPUT), mode="bilinear", antialias=True, align_corners=False)


def augment(u8: torch.Tensor) -> torch.Tensor:
    """Train augmentation on device: random crop (70-100% side), flip, colour jitter."""
    b, _, s, _ = u8.shape
    x = u8.float() / 255.0
    side = (torch.rand(b, device=x.device) * 0.3 + 0.7) * s
    cx = torch.rand(b, device=x.device) * (s - side) + side / 2
    cy = torch.rand(b, device=x.device) * (s - side) + side / 2
    flip = torch.where(torch.rand(b, device=x.device) < 0.5, -1.0, 1.0)
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0] = side / s * flip
    theta[:, 1, 1] = side / s
    theta[:, 0, 2] = (cx / s) * 2 - 1
    theta[:, 1, 2] = (cy / s) * 2 - 1
    grid = F.affine_grid(theta, (b, 3, INPUT, INPUT), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", align_corners=False)
    bright = 1 + (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 0.4
    contrast = 1 + (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 0.4
    mean = x.mean((1, 2, 3), keepdim=True)
    x = ((x - mean) * contrast + mean) * bright
    return x.clamp(0, 1)
