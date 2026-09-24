"""Prove real structured removal: prune 30% of YOLOv8n's channels and measure
the actual params / MACs / latency / bird-mAP change. Accuracy drop here is
expected and recovered by fine-tuning; the point is that size & speed really move.
"""

from __future__ import annotations

import time

import torch
import torch_pruning as tp
from ultralytics import YOLO

from harness.evaluate import BirdmAPEvaluator
from harness.prune import make_prunable, structured_prune

RATIO = 0.30


def measure(model, device, imgsz=640, runs=50, warmup=10):
    model.to(device).eval()
    ex = torch.randn(1, 3, imgsz, imgsz, device=device)
    macs, params = tp.utils.count_ops_and_params(model, ex)
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)
    with torch.no_grad():
        for _ in range(warmup):
            model(ex)
        sync()
        t = time.time()
        for _ in range(runs):
            model(ex)
        sync()
        dt = (time.time() - t) / runs
    return params, macs, dt


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ev = BirdmAPEvaluator("eval_set", device=device)

    native = YOLO("yolov8n.pt")
    p0, m0, dt0 = measure(native.model, device)
    map0 = ev.score(native)

    pruned = YOLO("yolov8n.pt")        # keep on CPU for the graph surgery
    make_prunable(pruned)              # new C2f_v2 branches init on CPU -> uniform
    structured_prune(pruned, RATIO)    # trace + prune on CPU
    pruned.model.to(device).eval()     # now move the whole pruned graph to device
    pruned.predictor = None            # force predictor rebuild on the pruned graph
    p1, m1, dt1 = measure(pruned.model, device)
    map1 = ev.score(pruned)

    def row(label, a, b, unit, better_low=True):
        pct = 100 * (b - a) / a
        print(f"{label:<14} {a:>9.3f} -> {b:>8.3f} {unit:<5} ({pct:+.0f}%)")

    print(f"\n=== YOLOv8n: native vs {int(RATIO*100)}% channel prune (no fine-tune yet) ===")
    row("params (M)", p0 / 1e6, p1 / 1e6, "M")
    row("MACs (G)", m0 / 1e9, m1 / 1e9, "G")
    row("latency (ms)", dt0 * 1e3, dt1 * 1e3, "ms")
    print(f"{'FPS':<14} {1/dt0:>9.1f} -> {1/dt1:>8.1f}")
    row("bird mAP@50", map0, map1, "")
    print("\n(accuracy is recoverable by fine-tuning; size & speed are the win)")


if __name__ == "__main__":
    main()
