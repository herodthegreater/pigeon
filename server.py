"""pigeon carving cockpit -- FastAPI backend.

Loads YOLOv8n once (in its prunable C2f_v2 form), fuses it so masking acts on
the exact inference weights, snapshots the native weights, and serves the
node-grid UI. Masking is the live, reversible interactive layer: click channels
-> zero exactly the channels Export would remove -> measure REAL bird mAP (the
same mAP the exported model gets before fine-tuning) + the exact MACs/params
Export would give. Real structural removal is the (separate) Export.

    uvicorn server:app --port 8000     # run from anywhere; paths are absolute
"""

from __future__ import annotations

import copy
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
from harness.mask import apply_plan, stage_table
from harness.model import apply_uniform, iter_weight_modules, restore, snapshot
from harness.prune import make_prunable, removal_plan, structured_prune_per_stage

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LIVE_EVAL_IMAGES = 40  # fast subset for interactive mAP

app = FastAPI()
_lock = threading.Lock()

ev = BirdmAPEvaluator(os.path.join(HERE, "eval_set"), device=DEVICE,
                      max_images=LIVE_EVAL_IMAGES)
ev_scan = BirdmAPEvaluator(os.path.join(HERE, "eval_set"), device=DEVICE,
                           max_images=16)  # tiny subset -> fast per-stage probing
yolo = YOLO(os.path.join(HERE, "yolov8n.pt"))
make_prunable(yolo)  # same module names as the planning copy below
yolo.model.to(DEVICE).eval()
# Fuse explicitly so snapshot + masking act on the exact weights inference
# uses: depending on the Ultralytics version, predict() fuses yolo.model in
# place or only a private copy.
yolo.model.fuse(verbose=False)
yolo.predict(source=ev.images[0], device=DEVICE, verbose=False)  # warm up
SNAP = snapshot(yolo.model)
TABLE = stage_table(yolo.model)
# Unfused CPU copy that only plans which channels Export would remove.
planner = YOLO(os.path.join(HERE, "yolov8n.pt"))
make_prunable(planner)
# Compute is always counted the same way: on the unfused model, on CPU, so the
# native numbers and every pruned/exported model are directly comparable.
_EX_CPU = torch.randn(1, 3, 640, 640)
NATIVE_MACS, NATIVE_PARAMS = tp.utils.count_ops_and_params(planner.model, _EX_CPU)
# What ships is the fused model (BN folded into conv bias), so size uses that.
FUSED_PARAMS = sum(p.numel() for p in yolo.model.parameters())
# The DFL conv holds the fixed bin positions 0..15 that decode box edges; it is
# not a learned weight, so a quantized export keeps it at native precision.
QUANT_SKIP = (".dfl.",)
NATIVE_MAP = ev.score(yolo)
NATIVE_MAP_SCAN = ev_scan.score(yolo)


class ApplyReq(BaseModel):
    masks: dict[str, float] = {}


class RecoverReq(BaseModel):
    masks: dict[str, float] = {}
    epochs: int = 12


class QuantReq(BaseModel):
    bits: int = 8
    group_size: int | None = 128


RECOVER_PROGRESS = os.path.join(HERE, "runs", "recover_progress.json")
_recover_proc: subprocess.Popen | None = None


def _mask(masks: dict[str, float]) -> None:
    """Restore native weights, then zero the channels Export would remove."""
    restore(yolo.model, SNAP)
    if any(v > 0 for v in masks.values()):
        apply_plan(yolo.model, removal_plan(planner, masks))
    yolo.predictor = None


def _pruned_cost(masks: dict[str, float]) -> tuple[int, int]:
    """(MACs, params) of the model Export would produce. Pruning a CPU copy of
    the planner takes ~0.3 s and counts the channels removed downstream of a
    cut too, which a per-stage FLOP-share estimate misses."""
    if not any(v > 0 for v in masks.values()):
        return NATIVE_MACS, NATIVE_PARAMS
    c = copy.deepcopy(planner)
    structured_prune_per_stage(c, masks)
    return tp.utils.count_ops_and_params(c.model, _EX_CPU)


def _measure(masks: dict[str, float]) -> dict:
    _mask(masks)
    m = ev.score(yolo)
    macs, params = _pruned_cost(masks)
    return {
        "map": round(m, 4),
        "map_pct": round(100 * m / max(NATIVE_MAP, 1e-9)),
        "macs_g": round(macs / 1e9, 3),
        "params_m": round(params / 1e6, 3),
        "saving_pct": round(100 * (1 - macs / NATIVE_MACS)),
    }


def _quant_size_bytes(bits: int, group_size: int | None, native_bits: int = 16) -> int:
    """On-disk estimate: quantized weights at `bits`, plus one fp16 scale per
    group (or per tensor), plus everything left at native precision (biases,
    the DFL, any skipped module)."""
    q_elems = scales = 0
    for name, mod in iter_weight_modules(yolo.model):
        if any(sk in name for sk in QUANT_SKIP):
            continue
        n = mod.weight.numel()
        q_elems += n
        scales += 1 if not group_size or group_size >= n else -(-n // group_size)
    rest = FUSED_PARAMS - q_elems
    return (q_elems * bits + scales * 16 + rest * native_bits) // 8


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
            "macs_g": round(NATIVE_MACS / 1e9, 3),
            "size_mb": round(FUSED_PARAMS * 2 / 1e6, 2),  # fused, fp16
        },
        "eval_images": len(ev.images),
        "device": DEVICE,
    }


@app.get("/api/sensitivity")
def sensitivity(probe: float = 0.25):
    """Probe every stage: cut `probe` of its channels, measure the mAP drop.
    High drop = load-bearing (avoid); low drop = redundant (cut here). Combined
    with FLOP share this teaches cut-by-value = FLOP / sensitivity."""
    with _lock:
        out = []
        for t in TABLE:
            _mask({t["stage"]: probe})
            m = ev_scan.score(yolo)
            drop = max(0.0, (NATIVE_MAP_SCAN - m) / max(NATIVE_MAP_SCAN, 1e-9))
            value = t["flop_frac"] / (drop + 0.05)  # compute saved per accuracy lost
            out.append({
                "stage": t["stage"], "flop_frac": t["flop_frac"],
                "channels": t["channels"], "sensitivity": round(drop, 3),
                "value": round(value, 3),
            })
        restore(yolo.model, SNAP)
        yolo.predictor = None
        return {"probe": probe, "native_map": round(NATIVE_MAP_SCAN, 4), "stages": out}


@app.post("/api/quantize")
def quantize(req: QuantReq):
    """Precision technique: fake-quantize every weight to `bits` and measure the
    real mAP. Reports estimated on-disk size, scale overhead included."""
    with _lock:
        restore(yolo.model, SNAP)
        if req.bits < 16:
            apply_uniform(yolo.model, req.bits, group_size=req.group_size,
                          skip=QUANT_SKIP)
        yolo.predictor = None
        m = ev.score(yolo)
        native = FUSED_PARAMS * 2
        size = native if req.bits >= 16 else _quant_size_bytes(req.bits, req.group_size)
        return {
            "bits": req.bits,
            "map": round(m, 4),
            "map_pct": round(100 * m / max(NATIVE_MAP, 1e-9)),
            "size_mb": round(size / 1e6, 2),
            "native_mb": round(native / 1e6, 2),
            "size_pct": round(100 * size / native),
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
            "macs_g": round(macs / 1e9, 3),
            "macs_pct": round(100 * macs / NATIVE_MACS),
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
