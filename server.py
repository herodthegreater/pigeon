"""pigeon carving cockpit -- FastAPI backend.

Loads YOLOv8n once, fuses it (so masking acts on the exact inference weights),
snapshots the native weights, and serves the node-grid UI. Masking is the live,
reversible interactive layer: click channels -> zero them -> measure REAL bird
mAP + ESTIMATED compute saved. Real structural removal is the (separate) Export.

    uvicorn server:app --port 8000     # run from anywhere; paths are absolute
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # make `harness` importable regardless of cwd

import torch
import torch_pruning as tp
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from ultralytics import YOLO

from harness.evaluate import BirdmAPEvaluator
from harness.mask import apply_masks, estimated_flop_saving, stage_table
from harness.model import restore, snapshot
from harness.prune import make_prunable, structured_prune_per_stage

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LIVE_EVAL_IMAGES = 40  # fast subset for interactive mAP

app = FastAPI()
_lock = threading.Lock()

ev = BirdmAPEvaluator(os.path.join(HERE, "eval_set"), device=DEVICE,
                      max_images=LIVE_EVAL_IMAGES)
yolo = YOLO(os.path.join(HERE, "yolov8n.pt"))
yolo.model.to(DEVICE).eval()
# Warm/fuse so snapshot + masking act on the exact weights inference uses.
yolo.predict(source=ev.images[0], device=DEVICE, verbose=False)
SNAP = snapshot(yolo.model)
TABLE = stage_table(yolo.model)
_ex = torch.randn(1, 3, 640, 640, device=DEVICE)
NATIVE_MACS, NATIVE_PARAMS = tp.utils.count_ops_and_params(yolo.model, _ex)
NATIVE_MAP = ev.score(yolo)


class ApplyReq(BaseModel):
    masks: dict[str, float] = {}


class RecoverReq(BaseModel):
    masks: dict[str, float] = {}
    epochs: int = 12


RECOVER_PROGRESS = os.path.join(HERE, "runs", "recover_progress.json")
_recover_proc: subprocess.Popen | None = None


def _measure(masks: dict[str, float]) -> dict:
    restore(yolo.model, SNAP)
    apply_masks(yolo.model, masks)
    yolo.predictor = None
    m = ev.score(yolo)
    saving = estimated_flop_saving(TABLE, masks)
    return {
        "map": round(m, 4),
        "map_pct": round(100 * m / max(NATIVE_MAP, 1e-9)),
        "est_flops_g": round(NATIVE_MACS / 1e9 * (1 - saving), 3),
        "est_saving_pct": round(100 * saving),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "static", "index.html")) as f:
        return f.read()


@app.get("/api/structure")
def structure():
    return {
        "node_height": 20,
        "stages": TABLE,
        "native": {
            "map": round(NATIVE_MAP, 4),
            "params_m": round(NATIVE_PARAMS / 1e6, 3),
            "flops_g": round(NATIVE_MACS / 1e9, 3),
        },
        "eval_images": len(ev.images),
        "device": DEVICE,
    }


@app.post("/api/apply")
def apply(req: ApplyReq):
    with _lock:
        return _measure(req.masks)


@app.post("/api/reset")
def reset():
    with _lock:
        restore(yolo.model, SNAP)
        yolo.predictor = None
    return {"ok": True, "map": round(NATIVE_MAP, 4)}


@app.post("/api/export")
def export(req: ApplyReq):
    """Real structural removal honoring the per-stage masks -> a genuinely
    smaller/faster model, measured and saved. Accuracy needs fine-tuning to
    recover (the cliff); this reports the size/speed win."""
    with _lock:
        fresh = YOLO(os.path.join(HERE, "yolov8n.pt"))
        make_prunable(fresh)
        if any(v > 0 for v in req.masks.values()):
            structured_prune_per_stage(fresh, req.masks)
        fresh.model.to(DEVICE).eval()
        ex = torch.randn(1, 3, 640, 640, device=DEVICE)
        macs, params = tp.utils.count_ops_and_params(fresh.model, ex)
        sync = torch.mps.synchronize if DEVICE == "mps" else (lambda: None)
        with torch.no_grad():
            for _ in range(8):
                fresh.model(ex)
            sync()
            t = time.time()
            for _ in range(30):
                fresh.model(ex)
            sync()
            dt = (time.time() - t) / 30
        path = os.path.join(HERE, "runs", "pigeon_pruned.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(fresh.model, path)
        return {
            "params_m": round(params / 1e6, 3),
            "params_pct": round(100 * params / NATIVE_PARAMS),
            "flops_g": round(macs / 1e9, 3),
            "flops_pct": round(100 * macs / NATIVE_MACS),
            "latency_ms": round(dt * 1000, 2),
            "path": os.path.relpath(path, HERE),
        }


@app.post("/api/recover")
def recover(req: RecoverReq):
    """Start a background subprocess: prune per masks, then fine-tune the pruned
    model. Progress streams to a JSON file that /api/recover_status polls."""
    global _recover_proc
    if _recover_proc is not None and _recover_proc.poll() is None:
        return {"started": False, "reason": "a recovery job is already running"}
    os.makedirs(os.path.dirname(RECOVER_PROGRESS), exist_ok=True)
    with open(RECOVER_PROGRESS, "w") as f:
        json.dump({"status": "starting", "epochs": req.epochs}, f)
    _recover_proc = subprocess.Popen([
        sys.executable, os.path.join(HERE, "recover_job.py"),
        "--masks", json.dumps(req.masks),
        "--epochs", str(req.epochs),
        "--progress", RECOVER_PROGRESS,
    ])
    return {"started": True, "epochs": req.epochs}


@app.get("/api/recover_status")
def recover_status():
    if not os.path.exists(RECOVER_PROGRESS):
        return {"status": "idle"}
    try:
        with open(RECOVER_PROGRESS) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"status": "starting"}  # mid-write; try again next poll
