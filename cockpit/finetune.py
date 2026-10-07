"""Fine-tune recovery: prune YOLOv8n, then retrain the *pruned* model and watch
bird mAP@50 climb back off zero. This closes the cut-and-regrow loop.

We drive Ultralytics' DetectionTrainer directly and hand it the pruned model.
In 8.4, `setup_model()` returns early when `trainer.model` is already an
nn.Module, so the pruned architecture survives (no rebuild from YAML), and
`_setup_train` still attaches nc/names/args for us.
"""

from __future__ import annotations

import argparse

import torch
import torch_pruning as tp
from ultralytics import YOLO

from harness.evaluate import BirdmAPEvaluator
from harness.prune import make_prunable, structured_prune


def params_macs(model, device):
    ex = torch.randn(1, 3, 640, 640, device=device)
    macs, params = tp.utils.count_ops_and_params(model.to(device), ex)
    return params, macs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.20)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ev = BirdmAPEvaluator("eval_set", device=device)

    native = YOLO("yolov8n.pt")
    p0, m0 = params_macs(native.model, device)
    map0 = ev.score(native)

    yolo = YOLO("yolov8n.pt")           # CPU surgery
    make_prunable(yolo)
    structured_prune(yolo, args.ratio)
    for p in yolo.model.parameters():
        p.requires_grad_(True)
    yolo.model.to(device)
    yolo.predictor = None
    p1, m1 = params_macs(yolo.model, device)
    map1 = ev.score(yolo)               # pruned, no fine-tune (expect ~0)

    overrides = dict(
        model="yolov8n.pt", data="finetune_data/data.yaml",
        epochs=args.epochs, imgsz=640, batch=args.batch, device=device,
        workers=0, amp=False, verbose=False, plots=False, cache=False,
        project="runs", name="finetune", exist_ok=True,
    )
    Trainer = yolo._smart_load("trainer")
    trainer = Trainer(overrides=overrides)
    trainer.model = yolo.model.to(device)   # hand it the pruned model
    trainer.train()

    tuned = YOLO(str(trainer.best))
    map2 = ev.score(tuned)

    print(f"\n=== fine-tune recovery (YOLOv8n, prune {int(args.ratio*100)}%, "
          f"{args.epochs} epochs) ===")
    print(f"{'stage':<22}{'params(M)':>10}{'MACs(G)':>9}{'bird mAP@50':>13}")
    print(f"{'native':<22}{p0/1e6:>10.3f}{m0/1e9:>9.3f}{map0:>13.4f}")
    print(f"{'pruned (no ft)':<22}{p1/1e6:>10.3f}{m1/1e9:>9.3f}{map1:>13.4f}")
    print(f"{'pruned + finetuned':<22}{p1/1e6:>10.3f}{m1/1e9:>9.3f}{map2:>13.4f}")


if __name__ == "__main__":
    main()
