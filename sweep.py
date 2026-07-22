"""Carve-and-check: sweep the doubt-dial on the detector, measuring how bird
detection survives as precision drops. The vision twin of the LLM smoke test.

    python sweep.py --source samples/birds.jpg
    python sweep.py --source samples/birds.jpg --group-size 64 --skip model.23

We warm up (which fuses conv+bn) *before* snapshotting, so the weights we
carve are exactly the ones that run at inference.
"""

from __future__ import annotations

import argparse

import torch
from ultralytics import YOLO

from harness.model import apply_uniform, restore, snapshot

BIRD = 14  # COCO 'bird'
LADDER = [16, 8, 6, 4, 3, 2, 1]


def detect_stats(yolo, source, conf, device):
    """Return (num_birds, mean_confidence) for one source at current weights."""
    res = yolo.predict(source=source, classes=[BIRD], conf=conf,
                       device=device, verbose=False)
    n, confs = 0, []
    for r in res:
        if r.boxes is not None and len(r.boxes):
            n += len(r.boxes)
            confs += r.boxes.conf.tolist()
    return n, (sum(confs) / len(confs) if confs else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--source", required=True, help="image or video with birds")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--group-size", type=int, default=None,
                    help="weights sharing one scale (None = whole tensor)")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="substrings of module names to keep at native precision")
    ap.add_argument("--device",
                    default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    yolo = YOLO(args.model)
    yolo.model.to(args.device).eval()
    # Warm up first: this fuses conv+bn in place, so snapshot/carve act on the
    # exact weights inference uses.
    yolo.predict(source=args.source, classes=[BIRD], conf=args.conf,
                 device=args.device, verbose=False)
    snap = snapshot(yolo.model)

    print(f"\n{'bits':>4}  {'birds':>5}  {'mean_conf':>9}   carved modules")
    print("-" * 44)
    for bits in LADDER:
        restore(yolo.model, snap)
        n_mod = apply_uniform(yolo.model, bits,
                              group_size=args.group_size, skip=tuple(args.skip))
        birds, mconf = detect_stats(yolo, args.source, args.conf, args.device)
        tag = "  (native)" if bits >= 16 else ""
        print(f"{bits:>4}  {birds:>5}  {mconf:>9.3f}   {n_mod}{tag}")
    print("-" * 44)
    print(f"group_size={args.group_size}  skip={args.skip or 'none'}")


if __name__ == "__main__":
    main()
