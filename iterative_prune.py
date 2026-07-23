"""Iterative prune + fine-tune: cut a little, regrow, repeat. This is the method
the cliff finding demands (a hard one-shot cut destroys the model; small cuts
recovered between steps keep it alive).

Apples-to-apples with the one-shot baseline: same 400 images, ~20% total prune
(3 x ~7.2%), similar total epochs. If iterative clearly beats one-shot's 0.07,
the method is validated and more data/epochs push it further.

    python iterative_prune.py --steps 3 --step-ratio 0.072 --epochs 15
"""

from __future__ import annotations

import argparse

import torch
import torch_pruning as tp
from ultralytics import YOLO

from harness.evaluate import BirdmAPEvaluator
from harness.prune import make_prunable, structured_prune


def macs_params_cpu(model):
    model.cpu()
    ex = torch.randn(1, 3, 640, 640)
    macs, params = tp.utils.count_ops_and_params(model, ex)
    return params, macs


def finetune(yolo, device, epochs, name):
    for p in yolo.model.parameters():
        p.requires_grad_(True)
    overrides = dict(
        model="yolov8n.pt", data="finetune_data/data.yaml",
        epochs=epochs, imgsz=640, batch=16, device=device,
        workers=0, amp=False, verbose=False, plots=False, cache=False,
        project="runs", name=name, exist_ok=True,
    )
    trainer = yolo._smart_load("trainer")(overrides=overrides)
    trainer.model = yolo.model.to(device)
    trainer.train()
    return str(trainer.best)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--step-ratio", type=float, default=0.072)  # 3x -> ~20%
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ev = BirdmAPEvaluator("eval_set", device=device)

    # Native mAP on a THROWAWAY (ev.score fuses conv+bn; we must keep the model
    # we prune unfused, or transfer_weights can't find cv1.bn).
    native = YOLO("yolov8n.pt")
    p0, m0 = macs_params_cpu(native.model)
    rows = [("native", p0, m0, ev.score(native))]

    yolo = YOLO("yolov8n.pt")
    make_prunable(yolo)                        # before any fusion

    for step in range(1, args.steps + 1):
        yolo.model.cpu()                       # surgery on CPU
        structured_prune(yolo, args.step_ratio)
        best = finetune(yolo, device, args.epochs, f"iter_step{step}")
        yolo = YOLO(best)                      # unfused reload, ready for next prune
        p, m = macs_params_cpu(yolo.model)
        mp = ev.score(YOLO(best))              # throwaway eval keeps `yolo` unfused
        pct = int((1 - (1 - args.step_ratio) ** step) * 100)
        rows.append((f"step{step} (~{pct}% pruned)", p, m, mp))
        print(f"  [step {step}] params={p/1e6:.3f}M  mAP@50={mp:.4f}", flush=True)

    print("\n=== iterative prune + fine-tune (400 imgs, YOLOv8n) ===")
    print(f"{'stage':<26}{'params(M)':>10}{'MACs(G)':>9}{'mAP@50':>9}")
    for name, p, m, mp in rows:
        print(f"{name:<26}{p/1e6:>10.3f}{m/1e9:>9.3f}{mp:>9.4f}")
    print("\ncompare: one-shot 20% + 40ep = 0.0706")


if __name__ == "__main__":
    main()
