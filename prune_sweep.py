"""Structural confession curve: really remove channels at several ratios and
measure bird mAP@50 + params + MACs at each. The twin of carve.py (which swept
bit-width). No fine-tuning here -- this is the raw 'cut without regrow' curve.
"""

from __future__ import annotations

import torch
import torch_pruning as tp
from ultralytics import YOLO

from harness.evaluate import BirdmAPEvaluator
from harness.prune import make_prunable, structured_prune

RATIOS = [0.0, 0.05, 0.10, 0.20, 0.30]


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ev = BirdmAPEvaluator("eval_set", device=device)
    ex = torch.randn(1, 3, 640, 640)

    print(f"{'prune':>6}  {'params(M)':>9}  {'MACs(G)':>8}  {'bird mAP@50':>11}")
    print("-" * 44)
    for r in RATIOS:
        y = YOLO("yolov8n.pt")           # fresh model each time (pruning is destructive)
        make_prunable(y)
        if r > 0:
            structured_prune(y, r)
        y.model.to(device).eval()
        y.predictor = None
        macs, params = tp.utils.count_ops_and_params(y.model, ex.to(device))
        m = ev.score(y)
        print(f"{int(r*100):>5}%  {params/1e6:>9.3f}  {macs/1e9:>8.3f}  {m:>11.4f}")
    print("-" * 44)
    print("(one-shot magnitude prune, no fine-tune -- recovery comes next)")


if __name__ == "__main__":
    main()
