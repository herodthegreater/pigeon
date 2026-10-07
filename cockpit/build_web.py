"""Build the serverless web edition of the cockpit into web/.

The browser runs the real YOLOv8n itself (TensorFlow.js, WebGL), so this
script only has to ship what the server would otherwise hold in memory:

  web/weights.txt   every fused conv weight + bias, fp16, PyTorch layout, as
                    base64 text (published pages serve text, not raw binary)
  web/model.json    the layer graph, conv manifest, stage shares, eval labels,
                    and the recorded Regrow/Export results the browser can't run
  web/photos/*.jpg  the eval photos (long side 512) the browser measures on

  web/index.html    the publishable page: web/cockpit.html with web/engine.js
                    inlined (a published page may only run inline scripts)

`web/cockpit.html` and `web/engine.js` are the hand-written sources (committed);
everything above is generated (git-ignored, like the datasets).

    python build_web.py              # after build_eval_set.py
"""

from __future__ import annotations

import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from harness.evaluate import BIRD, BirdmAPEvaluator
from harness.mask import head_stage, stage_table
from harness.model import fuse
from harness.precision import precision_stages

OUT = os.path.join(HERE, "web")
EVAL_ROOT = os.path.join(HERE, "eval_set")
N_EVAL = 32        # photos the browser scores for mAP (the live set)
N_PREVIEW = 12     # photos for "through the model's eyes"
LONG_SIDE = 512    # stored photo size; the browser letterboxes to 416
IMGSZ = 416

# What the browser cannot do (training, real channel removal) -- recorded from
# runs of the local server (640 px, 40-photo set) so the page can replay them.
RECORDED = {
    "regrow": {
        "recipe": "3-bit FP levels, per-channel, L0-L4 kept at 8-bit (3.16 bits/weight)",
        "epochs": 3,
        "before_pct": 0.3,
        "after_pct": 69.8,
        "val_curve": [0.2369, 0.4259, 0.4781],
    },
    "export": {
        "cut": "L15, L16, L19 by 20%",
        "params_pct": 97,
        "flops_pct": 95,
        "preview_pct": 25.6,
        "real_pct": 12.6,
    },
}


def main() -> None:
    os.makedirs(os.path.join(OUT, "photos"), exist_ok=True)
    yolo = YOLO(os.path.join(HERE, "yolov8n.pt"))
    yolo.model.eval()
    fuse(yolo)
    model = yolo.model

    # ---- conv manifest + weights (fp16, PyTorch [O, I, kh, kw] layout) ----
    convs, chunks, off = [], [], 0
    for name, m in model.named_modules():
        if not isinstance(m, torch.nn.Conv2d) or "dfl" in name:
            continue
        w = m.weight.detach().float().numpy().astype(np.float16).ravel()
        b = m.bias.detach().float().numpy().astype(np.float16)
        convs.append({"name": name, "shape": list(m.weight.shape), "stride": m.stride[0],
                      "pad": m.padding[0], "w": off, "b": off + w.size})
        chunks += [w, b]
        off += w.size + b.size
    raw = np.concatenate(chunks).astype("<f2").tobytes()
    with open(os.path.join(OUT, "weights.txt"), "w") as f:
        f.write(base64.b64encode(raw).decode("ascii"))

    # ---- module graph (what the JS forward walks) ----
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

    # ---- photos: the live eval set + preview picks ----
    full = BirdmAPEvaluator(EVAL_ROOT)
    photos, keep = [], set(full.images[:N_EVAL])
    previews = []
    for img in full.images:  # same picking rule as server.py, at the web size
        gt = full.gt[img]
        if len(previews) >= N_PREVIEW or not 1 <= len(gt) <= 5:
            continue
        w, h = Image.open(img).size
        if ((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])).max() / (w * h) < 0.02:
            continue
        r = yolo.predict(source=img, classes=[BIRD], conf=0.25, imgsz=IMGSZ, verbose=False)[0]
        if r.boxes is not None and len(r.boxes) and float(r.boxes.conf.max()) > 0.5:
            previews.append(img)
            keep.add(img)

    for img in full.images:
        if img not in keep:
            continue
        im = Image.open(img).convert("RGB")
        s = LONG_SIDE / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
        fn = os.path.splitext(os.path.basename(img))[0] + ".jpg"
        im.save(os.path.join(OUT, "photos", fn), quality=84)
        photos.append({"file": "photos/" + fn, "w": im.width, "h": im.height,
                       "gt": (full.gt[img] * s).round(1).tolist(),
                       "eval": img in full.images[:N_EVAL], "preview": img in previews})

    manifest = {
        "imgsz": IMGSZ,
        "convs": convs,
        "graph": graph,
        "head": head_stage(model),
        "prune_stages": stage_table(model),
        "precision_stages": precision_stages(model),
        "params": int(sum(p.numel() for p in model.parameters())),
        "photos": photos,
        "recorded": RECORDED,
    }
    with open(os.path.join(OUT, "model.json"), "w") as f:
        json.dump(manifest, f, separators=(",", ":"))

    with open(os.path.join(OUT, "cockpit.html")) as f:
        page = f.read()
    with open(os.path.join(OUT, "engine.js")) as f:
        page = page.replace("/*__ENGINE__*/", f.read())
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(page)

    mb = len(raw) / 1e6
    pmb = sum(os.path.getsize(os.path.join(OUT, p["file"])) for p in photos) / 1e6
    print(f"web/: {len(convs)} convs, weights {mb:.1f} MB, {len(photos)} photos "
          f"({sum(p['eval'] for p in photos)} eval, {len(previews)} preview) {pmb:.1f} MB")


if __name__ == "__main__":
    main()
