"""Recovery job (runs as a subprocess so training never blocks the server).

Prunes a fresh YOLOv8n per the UI masks, then fine-tunes the pruned model,
writing progress (epoch, mAP) to a JSON file after every epoch. The server
polls that file; the UI watches mAP climb off the cliff.

If the recipe also quantizes, the quantizer is put *in the training loop*
(QAT): the network fine-tunes while seeing its own low-bit weights, so it
learns to live on the coarse grid instead of just being rounded onto it.

Before and after training, the model is scored with the cockpit's own metric
(bird mAP@50 on the live eval subset, recipe applied to the fused model) so the
"before -> after" numbers are directly comparable to the scoreboard.

    python recover_job.py --masks '{"model.9":0.3}' --quant '{"bits":4}' \
        --epochs 12 --progress runs/recover_progress.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch_pruning as tp
from ultralytics import YOLO

from harness.evaluate import BirdmAPEvaluator
from harness.model import fuse
from harness.precision import apply_precision, enable_qat
from harness.prune import make_prunable, structured_prune_per_stage

LIVE_EVAL_IMAGES = 40  # must match server.py so numbers line up with the scoreboard


def write(path, d):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, path)  # atomic: the poller never sees half a file


def _gran(g):
    return g if g in ("tensor", "channel") else int(g)


def cockpit_map(yolo, quant: dict, ev: BirdmAPEvaluator) -> float:
    """Score a copy the way the cockpit does: fuse, apply the recipe's
    precision to the fused weights, measure bird mAP."""
    y = YOLO(os.path.join(HERE, "yolov8n.pt"))
    y.model = copy.deepcopy(yolo.model)
    for m in y.model.modules():  # QAT convs -> plain convs (fusion rebuilds them anyway)
        if type(m).__name__ == "QATConv2d":
            m.__class__ = torch.nn.Conv2d
    y.model.to(ev.device).eval()
    fuse(y)
    if quant.get("bits", 16) < 16 or quant.get("stage_bits"):
        apply_precision(y.model, quant.get("bits", 16), quant.get("stage_bits"),
                        _gran(quant.get("granularity", "channel")),
                        quant.get("levels", "int"), quant.get("clip", "max"))
    y.predictor = None
    return ev.score(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--quant", default="{}")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--progress", required=True)
    ap.add_argument("--native-map", type=float, default=None)
    a = ap.parse_args()
    masks = json.loads(a.masks)
    quant = json.loads(a.quant)
    pf = a.progress
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    qat = quant.get("bits", 16) < 16 or any(b < 16 for b in quant.get("stage_bits", {}).values())

    try:
        write(pf, {"status": "pruning", "epoch": 0, "epochs": a.epochs})
        ev = BirdmAPEvaluator(os.path.join(HERE, "eval_set"), device=device,
                              max_images=LIVE_EVAL_IMAGES)
        y = YOLO(os.path.join(HERE, "yolov8n.pt"))
        make_prunable(y)
        if any(v > 0 for v in masks.values()):
            structured_prune_per_stage(y, masks)
        for p in y.model.parameters():
            p.requires_grad_(True)
        y.model.to(device)
        macs, params = tp.utils.count_ops_and_params(
            y.model, torch.randn(1, 3, 640, 640, device=device))
        before = cockpit_map(y, quant, ev)
        n_qat = 0
        if qat:
            n_qat = enable_qat(y.model, quant.get("bits", 16), quant.get("stage_bits"),
                               _gran(quant.get("granularity", "channel")),
                               quant.get("levels", "int"), quant.get("clip", "max"))

        state = {"status": "training", "epoch": 0, "epochs": a.epochs, "map": 0.0,
                 "params_m": round(params / 1e6, 3), "flops_g": round(macs / 1e9, 3),
                 "qat": qat, "qat_convs": n_qat, "before": round(before, 4),
                 "native": a.native_map, "curve": []}
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
            # Ultralytics fires this once more in final_eval() on a *fused* best.pt,
            # which drops the fake-quant; the cockpit-metric "after" covers that.
            if len(state["curve"]) >= a.epochs:
                return
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

        state["status"] = "scoring"
        write(pf, state)
        best = YOLO(str(trainer.best))
        best.model.to(device)
        state["after"] = round(cockpit_map(best, quant, ev), 4)
        state["status"] = "done"
        state["best"] = os.path.relpath(str(trainer.best), HERE)
        write(pf, state)
    except Exception as e:
        write(pf, {"status": "error", "msg": str(e)})
        raise


if __name__ == "__main__":
    main()
