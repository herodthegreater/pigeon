"""Pigeon's scoreboard, computed exactly the way the site computes it.

- accuracy        bird / no-bird correct (logit > 0 means "bird")
- pointing        on bird images: does the heatmap's hottest point land inside a real bird box?
- agreement       same answer as a reference model (the uncompressed Pigeon)
- hard_neg_acc    accuracy on the tricky non-birds (planes, kites, bats, butterflies...)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .dataset import INPUT, Split, to_input


@torch.no_grad()
def predict(model, split: Split, device, bs: int = 128):
    model.eval()
    logits, cams = [], []
    for i in range(0, len(split), bs):
        x = to_input(split.images[i : i + bs].to(device))
        lg, cam = model(x, return_cam=True)
        logits.append(lg.float().cpu())
        cams.append(cam.float().cpu())
    return torch.cat(logits), torch.cat(cams)


def pointing_hits(cams: torch.Tensor, split: Split) -> torch.Tensor:
    """For each bird image, 1 if the CAM peak (upsampled to input size) is inside any bird box."""
    idx = [i for i, y in enumerate(split.labels.tolist()) if y == 1]
    up = F.interpolate(cams[idx].unsqueeze(1), size=(INPUT, INPUT), mode="bilinear", align_corners=False)
    flat = up.flatten(1).argmax(1)
    ys = (flat // INPUT).float().add(0.5) / INPUT
    xs = (flat % INPUT).float().add(0.5) / INPUT
    hits = []
    for k, i in enumerate(idx):
        x, y = xs[k].item(), ys[k].item()
        hits.append(any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in split.boxes[i]))
    return torch.tensor(hits, dtype=torch.float32)


def scoreboard(logits: torch.Tensor, cams: torch.Tensor, split: Split, ref_pred: torch.Tensor | None = None) -> dict:
    pred = (logits > 0).float()
    y = split.labels
    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    out = {
        "accuracy": (pred == y).float().mean().item(),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "hard_neg_acc": (pred[split.hard] == 0).float().mean().item() if split.hard.any() else None,
        "pointing": pointing_hits(cams, split).mean().item(),
    }
    if ref_pred is not None:
        out["agreement"] = (pred == ref_pred).float().mean().item()
    return out
