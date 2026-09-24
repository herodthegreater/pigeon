"""Confession curves: how each training-free method degrades Pigeon on the eval set.

Same algorithms as the browser (site/src/lib/compress), applied to the
BatchNorm-folded model. Tells us where the interesting cliffs are.

Run:  python -m train.sweep       -> prints tables, writes runs/sweep.json
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from .dataset import load_sheets
from .export import fold_batchnorm
from .metrics import predict, scoreboard
from .model import Pigeon
from .train import device

ROOT = Path(__file__).resolve().parent.parent


def fq_int(w: torch.Tensor, bits: int, gran: str, group: int = 32) -> torch.Tensor:
    out_ch = w.shape[0]
    flat = w.reshape(out_ch, -1) if gran != "tensor" else w.reshape(1, -1)
    if gran == "group" and flat.shape[1] % group == 0:
        flat = flat.reshape(-1, group)
    qmax = 2 ** (bits - 1) - 1
    s = flat.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax
    return (torch.clamp(torch.round(flat / s), -qmax, qmax) * s).reshape(w.shape)


def weights(model):
    return [p for n, p in model.named_parameters() if n.endswith("weight")]


def run(tag, model, ev, dev, ref):
    lg, cam = predict(model, ev, dev)
    m = scoreboard(lg, cam, ev, ref_pred=ref)
    print(f"  {tag:34s} acc {m['accuracy']:.3f}  pointing {m['pointing']:.3f}  agree {m['agreement']:.3f}")
    return {"tag": tag, **m}


def main() -> None:
    dev = device()
    ev = load_sheets()  # the exact pixels the site evaluates
    base = Pigeon(pretrained=False)
    base.load_state_dict(torch.load(ROOT / "runs/pigeon_fp32.pt", map_location="cpu"))
    base = fold_batchnorm(base.eval()).to(dev)
    ref = (predict(base, ev, dev)[0] > 0).float()
    results = [run("fp32", base, ev, dev, ref)]

    print("int weight quantization (symmetric, round-to-nearest)")
    for gran in ["tensor", "channel", "group"]:
        for bits in [8, 6, 5, 4, 3, 2]:
            m = copy.deepcopy(base)
            with torch.no_grad():
                for p in weights(m):
                    p.copy_(fq_int(p, bits, gran))
            results.append(run(f"int{bits} per-{gran}", m, ev, dev, ref))

    print("unstructured magnitude pruning (per tensor)")
    for frac in [0.3, 0.5, 0.7, 0.8, 0.9]:
        m = copy.deepcopy(base)
        with torch.no_grad():
            for p in weights(m):
                t = p.abs().flatten().kthvalue(max(1, int(frac * p.numel()))).values
                p.mul_((p.abs() > t).float())
        results.append(run(f"prune {int(frac*100)}% unstructured", m, ev, dev, ref))

    print("2:4 semi-structured")
    m = copy.deepcopy(base)
    with torch.no_grad():
        for p in weights(m):
            flat = p.reshape(p.shape[0], -1)
            if flat.shape[1] % 4:
                continue
            g = flat.reshape(-1, 4)
            idx = g.abs().topk(2, dim=1).indices
            mask = torch.zeros_like(g).scatter_(1, idx, 1.0)
            p.copy_((g * mask).reshape(p.shape))
    results.append(run("2:4 sparsity", m, ev, dev, ref))

    print("structured channel pruning (lowest-L1 output channels, every conv)")
    for frac in [0.05, 0.1, 0.2, 0.3]:
        m = copy.deepcopy(base)
        with torch.no_grad():
            for name, mod in m.named_modules():
                if isinstance(mod, torch.nn.Conv2d) and mod.out_channels > 8:
                    norms = mod.weight.abs().reshape(mod.out_channels, -1).mean(1)
                    drop = norms.argsort()[: int(frac * mod.out_channels)]
                    mod.weight[drop] = 0
                    if mod.bias is not None:
                        mod.bias[drop] = 0
        results.append(run(f"prune {int(frac*100)}% channels", m, ev, dev, ref))

    (ROOT / "runs/sweep.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
