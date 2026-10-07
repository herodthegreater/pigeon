"""First real accuracy curve: sweep the doubt-dial (bit-width) on the detector
and measure bird mAP@50 on the fixed COCO bird set at each level.

    python carve.py                          # YOLOv8n, the cockpit's model
    python carve.py --weights yolo11n.pt     # the original YOLO11n curve

This uses fake-quantization (accuracy changes, size/speed don't yet). Channel
pruning -- which also moves size/speed -- lands next, behind the same seam.
"""

from __future__ import annotations

import argparse

import torch
from ultralytics import YOLO

from harness.evaluate import BirdmAPEvaluator
from harness.model import apply_uniform, restore, snapshot

LADDER = [16, 8, 6, 4, 3, 2, 1]
EVAL_ROOT = "eval_set"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--group-size", type=int, default=None,
                    help="weights sharing one scale (None = whole tensor)")
    # the DFL conv's weights are the fixed bin positions 0..15 used to decode
    # box edges, not learned weights; a real quantized export keeps them as is
    ap.add_argument("--skip", nargs="*", default=[".dfl."],
                    help="module-name substrings to keep at native precision")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    yolo = YOLO(args.weights)
    yolo.model.to(device).eval()
    # Warm up so conv+bn fuse before we snapshot/carve.
    yolo.predict(source=EVAL_ROOT + "/images/val", classes=[14],
                 conf=0.25, device=device, verbose=False, stream=True).__next__()
    snap = snapshot(yolo.model)

    ev = BirdmAPEvaluator(EVAL_ROOT, device=device)
    print(f"eval: {ev.name} on {len(ev.images)} images / {ev.n_gt} birds")
    print(f"config: {args.weights}  group_size={args.group_size}  skip={args.skip or 'none'}\n")

    print(f"{'bits':>4}  {'bird mAP@50':>11}  {'vs native':>10}")
    print("-" * 32)
    native = None
    for bits in LADDER:
        restore(yolo.model, snap)
        apply_uniform(yolo.model, bits, group_size=args.group_size,
                      skip=tuple(args.skip))
        m = ev.score(yolo)
        if native is None:
            native = m
        delta = "" if bits >= 16 else f"{100 * (m - native) / max(native, 1e-9):+.0f}%"
        tag = "  (native)" if bits >= 16 else ""
        print(f"{bits:>4}  {m:>11.4f}  {delta:>10}{tag}")
    print("-" * 32)


if __name__ == "__main__":
    main()
