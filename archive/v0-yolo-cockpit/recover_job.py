"""Recovery job (runs as a subprocess so training never blocks the server).

Prunes a fresh YOLOv8n per the UI masks, then fine-tunes the pruned model,
writing progress (epoch, mAP) to a JSON file after every epoch. The server
polls that file; the UI watches mAP climb off the cliff.

    python recover_job.py --masks '{"model.9":0.3}' --epochs 12 --progress runs/recover_progress.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch_pruning as tp
from ultralytics import YOLO

from harness.prune import make_prunable, structured_prune_per_stage


def write(path, d):
    with open(path, "w") as f:
        json.dump(d, f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--progress", required=True)
    a = ap.parse_args()
    masks = json.loads(a.masks)
    pf = a.progress
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    try:
        write(pf, {"status": "pruning", "epoch": 0, "epochs": a.epochs})
        y = YOLO(os.path.join(HERE, "yolov8n.pt"))
        make_prunable(y)
        if any(v > 0 for v in masks.values()):
            structured_prune_per_stage(y, masks)
        for p in y.model.parameters():
            p.requires_grad_(True)
        y.model.to(device)
        macs, params = tp.utils.count_ops_and_params(
            y.model, torch.randn(1, 3, 640, 640, device=device))

        state = {"status": "training", "epoch": 0, "epochs": a.epochs, "map": 0.0,
                 "params_m": round(params / 1e6, 3), "flops_g": round(macs / 1e9, 3),
                 "curve": [0.0]}
        write(pf, state)

        trainer = y._smart_load("trainer")(overrides=dict(
            model=os.path.join(HERE, "yolov8n.pt"),
            data=os.path.join(HERE, "finetune_data", "data.yaml"),
            epochs=a.epochs, imgsz=640, batch=16, device=device,
            workers=0, amp=False, verbose=False, plots=False, cache=False,
            project=os.path.join(HERE, "runs"), name="recover", exist_ok=True,
        ))
        trainer.model = y.model.to(device)

        def on_epoch(tr):
            m = 0.0
            try:
                m = float(tr.metrics.get("metrics/mAP50(B)", 0.0))
            except Exception:
                pass
            state["epoch"] = tr.epoch + 1
            state["map"] = round(m, 4)
            state["curve"].append(round(m, 4))
            write(pf, state)

        trainer.add_callback("on_fit_epoch_end", on_epoch)
        trainer.train()

        state["status"] = "done"
        state["best"] = os.path.relpath(str(trainer.best), HERE)
        write(pf, state)
    except Exception as e:
        write(pf, {"status": "error", "msg": str(e)})
        raise


if __name__ == "__main__":
    main()
