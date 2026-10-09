"""Build what the site's Coop needs to run YOLOv8n in the browser.

The Coop's second cage (site/src/pages/coop/yolo.astro) runs the real YOLOv8n
in TF.js and scores it on the site's own 500 Open Images test photos
(site/public/testset), so this script ships the model and nothing else:

  site/public/coop/yolov8n/weights.bin
        every fused conv weight + bias (BatchNorm folded in), fp16 little-endian,
        PyTorch [O, I, kh, kw] layout; the DFL decoder's fixed conv is left out
  site/public/coop/yolov8n/model.json
        the layer graph, the conv manifest (offsets into weights.bin), the stage
        tables, the native compute at 224 px, the preview photos, and the
        PyTorch reference scores recorded here
  site/src/lib/detector/__fixtures__/torch_parity.json
        small parity cases for the browser port's unit tests: the precision lab
        on real weight slices, the AP@50 scorer and NMS on synthetic boxes

No photos are written: the browser reads the test set's sprite sheets.

The reference scores run the cockpit's own `BirdmAPEvaluator` (Ultralytics
predict at 224 px, conf 0.001, COCO class 14) on the same 500 photos, decoded
from the same sprite sheets, with the same fp16-rounded weights the browser
gets. They are recorded, not live; the page measures its own.

    .venv/bin/python cockpit/build_coop.py [--weights cockpit/yolov8n.pt]

Needs torch and ultralytics (the project venv), so CI can't run it: commit
what it writes.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np
import torch
import ultralytics
from PIL import Image
from ultralytics import YOLO

from harness.evaluate import BIRD, BirdmAPEvaluator
from harness.mask import head_stage, stage_table
from harness.model import fuse, iter_weight_modules, restore, snapshot
from harness.precision import ALWAYS_SKIP, apply_precision, precision_stages, quantize_weight

IMGSZ = 224                       # the test photos' size, a multiple of YOLOv8's 32 px stride
QUICK = 100                       # the page's quick estimate scores the first 100 photos
N_PREVIEW = 12                    # bird photos the page draws boxes on
TESTSET = os.path.join(ROOT, "site", "public", "testset")
OUT = os.path.join(ROOT, "site", "public", "coop", "yolov8n")
FIXTURE = os.path.join(ROOT, "site", "src", "lib", "detector", "__fixtures__", "torch_parity.json")


def load_testset():
    """The 500 photos exactly as the browser sees them: cut from the sprite sheets."""
    meta = json.load(open(os.path.join(TESTSET, "testset.json")))
    size, grid = meta["size"], meta["grid"]
    sheets = {}
    photos = []
    for img in meta["images"]:
        s = img["sheet"]
        if s not in sheets:
            sheets[s] = Image.open(os.path.join(TESTSET, f"sheet-{s}.webp")).convert("RGB")
        x, y = (img["cell"] % grid) * size, (img["cell"] // grid) * size
        photos.append(sheets[s].crop((x, y, x + size, y + size)))
    return meta, photos


def write_eval_root(root: str, meta, photos) -> None:
    """The photos and boxes in the YOLO layout BirdmAPEvaluator reads. Lossless
    PNG, numbered so glob order is test-set order."""
    os.makedirs(os.path.join(root, "images", "val"))
    os.makedirs(os.path.join(root, "labels", "val"))
    for i, (img, im) in enumerate(zip(meta["images"], photos)):
        im.save(os.path.join(root, "images", "val", f"{i:03d}.png"))
        with open(os.path.join(root, "labels", "val", f"{i:03d}.txt"), "w") as f:
            for x0, y0, x1, y1 in img["boxes"]:
                f.write(f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}\n")


def score(yolo, root: str, max_images: int | None = None) -> float:
    yolo.predictor = None  # predict() caches a private copy of the weights; make it take the current ones
    ev = BirdmAPEvaluator(root, imgsz=IMGSZ, device="cpu", max_images=max_images)
    return round(ev.score(yolo), 4)


def native_macs(model) -> int:
    """Multiply-accumulates of every exported conv for one 224 px photo."""
    macs = 0
    hooks = []

    def hook(m, inp, out):
        nonlocal macs
        k = m.kernel_size[0] * m.kernel_size[1]
        macs += out.shape[-2] * out.shape[-1] * m.in_channels * m.out_channels * k // m.groups

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) and "dfl" not in name:
            hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        model(torch.zeros(1, 3, IMGSZ, IMGSZ))
    for h in hooks:
        h.remove()
    return int(macs)


def pick_previews(yolo, meta, photos) -> list[int]:
    """Bird photos where the native model clearly finds a bird (one to five
    boxes, the biggest at least 2% of the frame), in test-set order."""
    yolo.predictor = None
    out = []
    for i, (img, im) in enumerate(zip(meta["images"], photos)):
        b = img["boxes"]
        if not img["label"] or not 1 <= len(b) <= 5:
            continue
        if max((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in b) < 0.02:
            continue
        r = yolo.predict(source=np.asarray(im)[:, :, ::-1], classes=[BIRD], conf=0.25, imgsz=IMGSZ,
                         device="cpu", verbose=False)[0]
        if r.boxes is not None and len(r.boxes) and float(r.boxes.conf.max()) > 0.5:
            out.append(i)
        if len(out) == N_PREVIEW:
            break
    return out


# ------------------------------------------------------------- parity fixture

def f32(a) -> list[float]:
    """float32 values written with 9 significant digits: enough to recover each one exactly."""
    return [float(f"{v:.9g}") for v in np.asarray(a, dtype=np.float32).ravel()]


def quant_cases(model) -> list[dict]:
    mods = dict(model.named_modules())
    tensors = {
        "model.0.conv [16,3,3,3]": mods["model.0.conv"].weight.detach(),
        "model.2.m.0.cv1.conv first 2 of 16 channels": mods["model.2.m.0.cv1.conv"].weight.detach()[:2],
        "model.22.cv3.2.2 first 4 of 80 channels": mods["model.22.cv3.2.2"].weight.detach()[:4],
    }
    configs = {
        "model.0.conv [16,3,3,3]": [
            (8, "channel", "int", "max"), (4, "channel", "int", "max"), (4, "tensor", "int", "max"),
            (4, 16, "int", "max"), (4, "channel", "fp", "max"), (3, "tensor", "fp", "max"),
            (2, "channel", "int", "max"), (1, "channel", "int", "max"), (4, "channel", "int", "mse"),
        ],
        "model.2.m.0.cv1.conv first 2 of 16 channels": [
            (8, "tensor", "int", "max"), (5, 32, "int", "max"), (4, 64, "fp", "max"),
            (6, "channel", "fp", "max"), (2, 128, "fp", "max"), (3, "channel", "fp", "mse"),
        ],
        "model.22.cv3.2.2 first 4 of 80 channels": [
            (8, "channel", "int", "max"), (4, 32, "int", "max"), (4, "channel", "fp", "max"),
            (7, "tensor", "fp", "max"), (1, "tensor", "int", "max"),
        ],
    }
    out = []
    for name, w in tensors.items():
        cases = []
        for bits, gran, levels, clip in configs[name]:
            q = quantize_weight(w, bits, gran, levels, clip)
            cases.append({"bits": bits, "granularity": gran, "levels": levels, "clip": clip, "out": f32(q)})
        out.append({"name": name, "shape": list(w.shape), "w": f32(w), "cases": cases})
    return out


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy, self.conf = torch.tensor(xyxy, dtype=torch.float64).reshape(-1, 4), torch.tensor(conf, dtype=torch.float64)

    def __len__(self):
        return len(self.conf)


class _Result:
    def __init__(self, preds):
        self.boxes = _Boxes([p[:4] for p in preds], [p[4] for p in preds]) if preds else None


class _FakeYolo:
    """Stands in for Ultralytics so the harness scorer runs on synthetic detections."""

    def __init__(self, preds):
        self.preds = preds

    def predict(self, source, **kw):
        return (_Result(self.preds[i]) for i in range(len(source)))


def ap_cases() -> list[dict]:
    rng = np.random.default_rng(14)
    cases = []
    for n_img in (1, 6, 30):
        gt, preds = [], []
        for _ in range(n_img):
            g = []
            for _ in range(int(rng.integers(0, 4))):
                x0, y0 = rng.uniform(0, 160, 2)
                g.append([x0, y0, x0 + rng.uniform(12, 64), y0 + rng.uniform(12, 64)])
            p = []
            for b in g:  # a near hit for most boxes, sometimes two (a duplicate is a false positive)
                for _ in range(int(rng.integers(0, 3))):
                    j = rng.normal(0, 5, 4)
                    p.append([*(np.array(b) + j), rng.uniform(0.05, 1)])
            for _ in range(int(rng.integers(0, 4))):  # strays
                x0, y0 = rng.uniform(0, 180, 2)
                p.append([x0, y0, x0 + rng.uniform(8, 40), y0 + rng.uniform(8, 40), rng.uniform(0.001, 0.6)])
            gt.append([[round(float(v), 3) for v in b] for b in g])
            preds.append([[round(float(v), 3) for v in b[:4]] + [round(float(b[4]), 6)] for b in p])
        ev = object.__new__(BirdmAPEvaluator)
        ev.images = [str(i) for i in range(n_img)]
        ev.gt = {str(i): np.array(g, dtype=float) if g else np.zeros((0, 4)) for i, g in enumerate(gt)}
        ev.n_gt = sum(len(g) for g in gt)
        ev.imgsz, ev.conf, ev.iou, ev.device = IMGSZ, 0.001, 0.5, "cpu"
        cases.append({"gt": gt, "preds": preds, "ap": ev.score(_FakeYolo(preds))})
    return cases


def nms_cases() -> list[dict]:
    """torchvision's NMS (what Ultralytics calls) at IoU 0.7, on clustered boxes."""
    import torchvision

    rng = np.random.default_rng(22)
    cases = []
    for n in (5, 40):
        centers = rng.uniform(20, 200, (4, 2))
        boxes = []
        for _ in range(n):
            cx, cy = centers[rng.integers(0, 4)] + rng.normal(0, 4, 2)
            w, h = rng.uniform(20, 50, 2)
            boxes.append([round(float(v), 3) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)])
        scores = [round(float(s), 6) for s in rng.uniform(0, 1, n)]
        keep = torchvision.ops.nms(torch.tensor(boxes, dtype=torch.float64), torch.tensor(scores, dtype=torch.float64), 0.7)
        cases.append({"boxes": boxes, "scores": scores, "keep": keep.tolist()})
    return cases


# ---------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", default=os.path.join(HERE, "yolov8n.pt"),
                    help="the Ultralytics YOLOv8n checkpoint (git-ignored; default cockpit/yolov8n.pt)")
    args = ap.parse_args()
    if not os.path.exists(args.weights):  # YOLO() would quietly download a copy instead
        sys.exit(f"no checkpoint at {args.weights}; pass --weights /path/to/yolov8n.pt")

    torch.manual_seed(0)
    yolo = YOLO(args.weights)
    yolo.model.eval()
    fuse(yolo)
    model = yolo.model

    # The browser gets fp16 weights, so every number recorded here uses exactly those.
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d) and "dfl" not in name:
                m.weight.copy_(m.weight.half().float())
                if m.bias is not None:
                    m.bias.copy_(m.bias.half().float())

    # ---- weights.bin + conv manifest (fp16, PyTorch [O, I, kh, kw] layout) ----
    convs, chunks, off = [], [], 0
    for name, m in model.named_modules():
        if not isinstance(m, torch.nn.Conv2d) or "dfl" in name:
            continue
        assert m.groups == 1 and m.bias is not None, name
        w = m.weight.detach().numpy().astype(np.float16).ravel()
        b = m.bias.detach().numpy().astype(np.float16)
        convs.append({"name": name, "shape": list(m.weight.shape), "stride": m.stride[0], "pad": m.padding[0],
                      "w": off, "b": off + w.size})
        chunks += [w, b]
        off += w.size + b.size
    raw = np.concatenate(chunks).astype("<f2").tobytes()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "weights.bin"), "wb") as f:
        f.write(raw)

    # ---- the module graph the browser's forward pass walks ----
    graph = []
    for i, m in enumerate(model.model):
        t = type(m).__name__
        node = {"i": i, "type": t, "f": m.f}
        if t == "C2f":
            node.update(n=len(m.m), c=m.c, add=[bool(b.add) for b in m.m])
        elif t == "SPPF":
            node.update(k=m.m.kernel_size)
        elif t == "Detect":
            node.update(nc=m.nc, reg_max=m.reg_max, strides=m.stride.tolist())
        graph.append(node)

    params = int(sum(p.numel() for p in model.parameters()))
    conv_params = sum(c["shape"][0] * int(np.prod(c["shape"][1:])) + c["shape"][0] for c in convs)
    quantizable = sum(mod.weight.numel() for n, mod in iter_weight_modules(model) if not any(s in n for s in ALWAYS_SKIP))

    # ---- reference scores: the harness evaluator on the same photos and weights ----
    meta, photos = load_testset()
    snap = snapshot(model)
    with tempfile.TemporaryDirectory() as root:
        write_eval_root(root, meta, photos)
        ref = {"native": {"all": score(yolo, root), "quick": score(yolo, root, QUICK)}}
        for label, bits in (("int8_channel", 8), ("int4_channel", 4)):
            apply_precision(model, bits=bits, granularity="channel", levels="int", clip="max")
            ref[label] = {"all": score(yolo, root), "quick": score(yolo, root, QUICK)}
            restore(model, snap)
    previews = pick_previews(yolo, meta, photos)

    manifest = {
        "imgsz": IMGSZ,
        "bird_class": BIRD,
        "weights_file": "weights.bin",
        "weights_bytes": len(raw),
        "convs": convs,
        "graph": graph,
        "head": head_stage(model),
        "prune_stages": stage_table(model, imgsz=IMGSZ),
        "precision_stages": precision_stages(model),
        "params": params,
        "other_params": params - conv_params,  # the DFL decoder's fixed 16 bin positions
        "quantizable_weights": quantizable,
        "macs": native_macs(model),
        "preview": [meta["images"][i]["id"] for i in previews],
        "recorded": {
            "what": "bird AP@50 from the cockpit's BirdmAPEvaluator (Ultralytics predict, PyTorch, CPU) on the same "
                    "photos at 224 px with these fp16 weights; 'quick' is the first 100 photos",
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "date": datetime.date.today().isoformat(),
            "ap": ref,
        },
    }
    with open(os.path.join(OUT, "model.json"), "w") as f:
        json.dump(manifest, f, separators=(",", ":"))

    # ---- parity fixture for the unit tests ----
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    with open(FIXTURE, "w") as f:
        json.dump({"source": "cockpit/build_coop.py (harness/precision.py, harness/evaluate.py, torchvision NMS)",
                   "quant": quant_cases(model), "ap": ap_cases(), "nms": nms_cases()}, f, separators=(",", ":"))

    print(f"{os.path.relpath(OUT, ROOT)}: {len(convs)} convs, {params:,} params, weights.bin {len(raw) / 1e6:.2f} MB, "
          f"{manifest['macs'] / 1e9:.3f} GMACs at {IMGSZ} px, {len(previews)} preview photos")
    for k, v in ref.items():
        print(f"  {k:13s} bird AP@50 {v['all']:.4f} on 500 photos, {v['quick']:.4f} on the first {QUICK}")
    print(f"{os.path.relpath(FIXTURE, ROOT)}: {os.path.getsize(FIXTURE) / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
