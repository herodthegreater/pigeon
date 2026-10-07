"""pigeon reverse-training cockpit -- FastAPI backend.

Loads YOLOv8n once (in its prunable C2f_v2 form), fuses it (so every edit acts
on the exact inference weights), snapshots the native weights, and serves the
cockpit UI.

Everything the UI does is one **recipe** -- {masks, quant} -- so the levers
stack: channel masks (structure) and fake-quantization (precision) are applied
together, and every measurement is the REAL bird mAP of that combined model.
Masks zero exactly the channels Export would remove, so a masked model scores
what the exported model scores before any regrow.
Size and compute are ESTIMATES of what a real export would weigh; Export and
Recover produce the real thing.

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

import numpy as np
import torch
from PIL import Image
import torch_pruning as tp
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field
from ultralytics import YOLO

from harness.evaluate import BIRD, BirdmAPEvaluator, _iou_matrix
from harness.mask import active_cost, apply_plan, head_stage, stage_table
from harness.model import fuse, iter_weight_modules, restore, snapshot
from harness.precision import (ALWAYS_SKIP, apply_precision, inspect_weights,
                               precision_stages)
from harness.prune import make_prunable, removal_plan, structured_prune_per_stage

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LIVE_EVAL_IMAGES = 40  # fast subset for interactive mAP
SCAN_IMAGES = 16       # tinier subset for per-stage probing
PREVIEW_IMAGES = 12    # photos for the "through the model's eyes" panel
PREVIEW_CONF = 0.25

app = FastAPI()
_lock = threading.Lock()

EVAL_ROOT = os.path.join(HERE, "eval_set")
ev = BirdmAPEvaluator(EVAL_ROOT, device=DEVICE, max_images=LIVE_EVAL_IMAGES)
ev_scan = BirdmAPEvaluator(EVAL_ROOT, device=DEVICE, max_images=SCAN_IMAGES)
yolo = YOLO(os.path.join(HERE, "yolov8n.pt"))
make_prunable(yolo)  # same module names as the planning copy below
yolo.model.to(DEVICE).eval()
fuse(yolo)  # BN folded in: masks zero a channel for real, quant sees deploy weights
yolo.predict(source=ev.images[0], device=DEVICE, verbose=False)  # warm up
SNAP = snapshot(yolo.model)
TABLE = stage_table(yolo.model)          # prunable stages (head excluded)
PSTAGES = precision_stages(yolo.model)   # quantizable stages (head included)
HEAD = head_stage(yolo.model)
# Unfused CPU copy that only plans which channels Export would remove.
planner = YOLO(os.path.join(HERE, "yolov8n.pt"))
make_prunable(planner)
# Compute is counted the way Export counts it: unfused model, on CPU, so native
# and exported numbers are directly comparable. Size uses the fused model,
# which is what ships (BN folded into the conv bias).
NATIVE_MACS, NATIVE_PARAMS = tp.utils.count_ops_and_params(
    planner.model, torch.randn(1, 3, 640, 640))
FUSED_PARAMS = sum(p.numel() for p in yolo.model.parameters())
NATIVE_MAP = ev.score(yolo)
NATIVE_MAP_SCAN = ev_scan.score(yolo)
NATIVE_BYTES = FUSED_PARAMS * 2  # fp16 checkpoint
NATIVE_ACTIVE = active_cost(yolo.model)  # (MACs, conv weights) with nothing masked


# --------------------------------------------------------------------------- #
# the recipe


class Quant(BaseModel):
    bits: int = Field(16, ge=1, le=16)
    granularity: str = "channel"   # "tensor" | "channel" | group size, e.g. "32"
    levels: str = "int"            # "int" (evenly spaced) | "fp" (minifloat grid)
    clip: str = "max"              # "max" | "mse"
    stage_bits: dict[str, int] = {}  # mixed precision overrides


class Recipe(BaseModel):
    masks: dict[str, float] = {}
    quant: Quant = Quant()


class MeasureReq(Recipe):
    commit: bool = True   # make this the live model (preview follows it)
    fast: bool = False    # score on the tiny scan subset instead


class ScanReq(BaseModel):
    bits: int = 4
    granularity: str = "channel"
    levels: str = "int"
    clip: str = "max"


class RecoverReq(Recipe):
    epochs: int = 12


def _gran(g: str):
    if g in ("tensor", "channel"):
        return g
    try:
        return max(2, int(g))
    except ValueError:
        raise HTTPException(400, f"bad granularity {g!r}")


_current = Recipe()


def _mask(masks: dict[str, float]) -> None:
    """restore -> zero the channels Export would remove."""
    restore(yolo.model, SNAP)
    if any(v > 0 for v in masks.values()):
        apply_plan(yolo.model, removal_plan(planner, masks))


def _apply(r: Recipe) -> dict:
    """restore -> mask -> quantize. Returns the precision storage accounting."""
    _mask(r.masks)
    q = r.quant
    stats = apply_precision(yolo.model, q.bits, q.stage_bits, _gran(q.granularity),
                            q.levels, q.clip)
    yolo.predictor = None
    return stats


def _report(r: Recipe, stats: dict, m: float, native: float) -> dict:
    # what a real export would keep: count live channels in the masked model
    macs, conv_w = active_cost(yolo.model) if r.masks else NATIVE_ACTIVE
    flop_keep = macs / NATIVE_ACTIVE[0]
    param_keep = conv_w / NATIVE_ACTIVE[1]
    other = FUSED_PARAMS - stats["weights"]  # biases etc. stay fp16
    size_bits = (stats["weight_bits"] + stats["scale_bits"]) * param_keep + other * 16
    size_b = size_bits / 8
    return {
        "map": round(m, 4),
        "map_pct": round(100 * m / max(native, 1e-9), 1),
        "size_mb": round(size_b / 1e6, 2),
        "size_pct": round(100 * size_b / NATIVE_BYTES, 1),
        "eff_bits": round(stats["eff_bits"], 2),
        "flops_g": round(NATIVE_MACS / 1e9 * flop_keep, 3),
        "flops_pct": round(100 * flop_keep, 1),
    }


# --------------------------------------------------------------------------- #
# preview photos: pick birds the native model actually sees, cache its boxes


def _gt_boxes(evaluator, img):
    return evaluator.gt[img].round(1).tolist()


def _predict_boxes(img) -> list[list[float]]:
    r = yolo.predict(source=img, classes=[BIRD], conf=PREVIEW_CONF,
                     device=DEVICE, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    return [[*(round(float(v), 1) for v in b), round(float(c), 3)] for b, c in zip(xyxy, conf)]


def _pick_previews():
    full = BirdmAPEvaluator(EVAL_ROOT, device=DEVICE)
    picks = []
    for img in full.images:
        gt = full.gt[img]
        if not 1 <= len(gt) <= 5:
            continue
        w, h = Image.open(img).size
        biggest = ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).max() / (w * h)
        if biggest < 0.02:
            continue
        boxes = _predict_boxes(img)
        if boxes and max(b[4] for b in boxes) > 0.5:
            picks.append({"path": img, "w": w, "h": h, "gt": _gt_boxes(full, img),
                          "native": boxes})
        if len(picks) >= PREVIEW_IMAGES:
            break
    return picks


PREVIEWS = _pick_previews()


def _score_boxes(boxes, gt) -> dict:
    """Found / missed / false-alarm counts at IoU 0.5 (greedy by confidence)."""
    if not boxes:
        return {"found": 0, "missed": len(gt), "false": 0}
    pb = np.array([b[:4] for b in boxes])
    g = np.array(gt) if gt else np.zeros((0, 4))
    ious = _iou_matrix(pb, g)
    matched, found = set(), 0
    for i in np.argsort([-b[4] for b in boxes]):
        if g.shape[0]:
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= 0.5 and j not in matched:
                matched.add(j)
                found += 1
    return {"found": found, "missed": len(gt) - found, "false": len(boxes) - found}


# --------------------------------------------------------------------------- #
# routes


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "static", "index.html")) as f:
        return f.read()


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/api/structure")
def structure():
    return {
        "node_height": 20,
        "stages": TABLE,
        "precision_stages": PSTAGES,
        "head": HEAD,
        "native": {
            "map": round(NATIVE_MAP, 4),
            "params_m": round(NATIVE_PARAMS / 1e6, 3),
            "flops_g": round(NATIVE_MACS / 1e9, 3),
            "size_mb": round(NATIVE_BYTES / 1e6, 2),
        },
        "eval_images": len(ev.images),
        "previews": len(PREVIEWS),
        "device": DEVICE,
    }


@app.post("/api/measure")
def measure(req: MeasureReq):
    """Apply a recipe and measure its REAL bird mAP (+ estimated size/compute).
    commit=False scores a what-if (sweeps, auto-mix) and puts the live model back."""
    global _current
    r = Recipe(masks=req.masks, quant=req.quant)
    with _lock:
        stats = _apply(r)
        e, native = (ev_scan, NATIVE_MAP_SCAN) if req.fast else (ev, NATIVE_MAP)
        out = _report(r, stats, e.score(yolo), native)
        if req.commit:
            _current = r
        else:
            _apply(_current)
        return out


@app.get("/api/sensitivity")
def sensitivity(probe: float = 0.10):
    """Structure scan: cut `probe` of each stage's channels, measure the mAP
    drop. High drop = load-bearing (avoid); low drop = redundant (cut here).
    Combined with FLOP share this teaches cut-by-value = FLOP / sensitivity."""
    with _lock:
        out = []
        for t in TABLE:
            _mask({t["stage"]: probe})
            yolo.predictor = None
            m = ev_scan.score(yolo)
            drop = max(0.0, (NATIVE_MAP_SCAN - m) / max(NATIVE_MAP_SCAN, 1e-9))
            value = t["flop_frac"] / (drop + 0.05)  # compute saved per accuracy lost
            out.append({
                "stage": t["stage"], "flop_frac": t["flop_frac"],
                "channels": t["channels"], "sensitivity": round(drop, 3),
                "value": round(value, 3),
            })
        _apply(_current)
        return {"probe": probe, "native_map": round(NATIVE_MAP_SCAN, 4), "stages": out}


@app.post("/api/precision_scan")
def precision_scan(req: ScanReq):
    """Precision scan: quantize ONE stage at a time to `bits` (everything else
    native) and measure the mAP drop. Reveals where precision is load-bearing --
    usually the tiny early stages, not the big ones."""
    with _lock:
        out = []
        for s in PSTAGES:
            restore(yolo.model, SNAP)
            apply_precision(yolo.model, req.bits, None, _gran(req.granularity),
                            req.levels, req.clip, only=s["stage"])
            yolo.predictor = None
            m = ev_scan.score(yolo)
            drop = max(0.0, (NATIVE_MAP_SCAN - m) / max(NATIVE_MAP_SCAN, 1e-9))
            out.append({**s, "sensitivity": round(drop, 3),
                        "value": round(s["weight_frac"] / (drop + 0.05), 3)})
        _apply(_current)
        return {"bits": req.bits, "native_map": round(NATIVE_MAP_SCAN, 4), "stages": out}


@app.get("/api/inspect")
def inspect(stage: str, bits: int = 4, granularity: str = "channel", levels: str = "int",
            clip: str = "max", row: int = 0):
    """The microscope: the biggest conv in `stage`, one scale-sharing row of its
    native weights, and where each weight snaps at this precision."""
    best = None
    for name, mod in iter_weight_modules(yolo.model):
        if name.startswith(stage + ".") and not any(s in name for s in ALWAYS_SKIP):
            if best is None or mod.weight.numel() > SNAP[best[0]].numel():
                best = (name, mod)
    if best is None:
        raise HTTPException(404, f"no weights in {stage}")
    w = SNAP[best[0]].detach().cpu()
    out = inspect_weights(w, bits, _gran(granularity), levels, clip, row)
    return {"layer": best[0], "shape": list(w.shape), **out}


@app.get("/api/preview")
def preview(idx: int = 0):
    """Run the LIVE model (current recipe) on one photo: its boxes, the native
    model's boxes, and the ground truth."""
    if not PREVIEWS:
        return {"n": 0}
    p = PREVIEWS[idx % len(PREVIEWS)]
    with _lock:
        boxes = _predict_boxes(p["path"])
    return {"n": len(PREVIEWS), "idx": idx % len(PREVIEWS), "w": p["w"], "h": p["h"],
            "gt": p["gt"], "native": p["native"], "boxes": boxes,
            "score": _score_boxes(boxes, p["gt"]),
            "native_score": _score_boxes(p["native"], p["gt"])}


@app.get("/api/preview_img/{idx}")
def preview_img(idx: int):
    if not PREVIEWS:
        raise HTTPException(404)
    return FileResponse(PREVIEWS[idx % len(PREVIEWS)]["path"])


@app.post("/api/reset")
def reset():
    global _current
    with _lock:
        _current = Recipe()
        _apply(_current)
    return {"ok": True, "map": round(NATIVE_MAP, 4)}


@app.post("/api/export")
def export(req: Recipe):
    """Real structural removal honoring the per-stage masks -> a genuinely
    smaller/faster model, measured (params/FLOPs/latency AND real mAP before any
    regrow) and saved."""
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
        m = ev.score(fresh)  # the really-pruned model, no regrow yet
        return {
            "params_m": round(params / 1e6, 3),
            "params_pct": round(100 * params / NATIVE_PARAMS),
            "flops_g": round(macs / 1e9, 3),
            "flops_pct": round(100 * macs / NATIVE_MACS),
            "latency_ms": round(dt * 1000, 2),
            "map": round(m, 4),
            "map_pct": round(100 * m / max(NATIVE_MAP, 1e-9), 1),
            "path": os.path.relpath(path, HERE),
        }


RECOVER_PROGRESS = os.path.join(HERE, "runs", "recover_progress.json")
_recover_proc: subprocess.Popen | None = None


@app.post("/api/recover")
def recover(req: RecoverReq):
    """Start a background subprocess: prune per masks, then fine-tune -- with the
    recipe's quantization in the loop (QAT) when it has any. Progress streams to
    a JSON file that /api/recover_status polls."""
    global _recover_proc
    if _recover_proc is not None and _recover_proc.poll() is None:
        return {"started": False, "reason": "a recovery job is already running"}
    os.makedirs(os.path.dirname(RECOVER_PROGRESS), exist_ok=True)
    with open(RECOVER_PROGRESS, "w") as f:
        json.dump({"status": "starting", "epochs": req.epochs}, f)
    _recover_proc = subprocess.Popen([
        sys.executable, os.path.join(HERE, "recover_job.py"),
        "--masks", json.dumps(req.masks),
        "--quant", req.quant.model_dump_json(),
        "--epochs", str(req.epochs),
        "--progress", RECOVER_PROGRESS,
        "--native-map", str(NATIVE_MAP),
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
