"""Baseline yardstick for pigeon: size, params, and latency on this machine.

Every shrink/carve experiment is judged against these numbers. Run it on the
native model first, then re-run after quantizing/pruning to see what you paid
and what you bought.

    python benchmark.py                      # native yolo11n on mps
    python benchmark.py --model yolo11n.pt --imgsz 640 --runs 100
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from ultralytics import YOLO


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device",
                    default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    yolo = YOLO(args.model)
    net = yolo.model.to(args.device).eval()

    params = count_params(net)
    size_mb = os.path.getsize(args.model) / 1e6 if os.path.exists(args.model) else float("nan")

    x = torch.randn(1, 3, args.imgsz, args.imgsz, device=args.device)
    sync = torch.mps.synchronize if args.device == "mps" else (lambda: None)

    with torch.no_grad():
        for _ in range(args.warmup):
            net(x)
        sync()
        t0 = time.time()
        for _ in range(args.runs):
            net(x)
        sync()
        per = (time.time() - t0) / args.runs

    print(f"model    : {args.model}")
    print(f"device   : {args.device}")
    print(f"params   : {params / 1e6:.3f} M")
    print(f"weights  : {size_mb:.2f} MB  (.pt on disk)")
    print(f"latency  : {per * 1000:.2f} ms/frame   ({1 / per:.1f} FPS)   @ {args.imgsz}px")


if __name__ == "__main__":
    main()
