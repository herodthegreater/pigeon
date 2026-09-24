"""Specialize: fine-tune MobileNetV3-Small into Pigeon (bird / no-bird + CAM).

Run:  python -m train.train [--epochs 12] [--lr 5e-4]
Saves the best-on-val checkpoint to runs/pigeon_fp32.pt.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .dataset import augment, load_split
from .metrics import predict, scoreboard
from .model import Pigeon

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = device()
    RUNS.mkdir(exist_ok=True)

    tr, va = load_split("train"), load_split("val")
    tr_x, tr_y = tr.images.to(dev), tr.labels.to(dev)
    print(f"train {len(tr)} (birds {int(tr.labels.sum())})  val {len(va)}  device {dev}")

    model = Pigeon().to(dev)
    opt = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": args.lr * 0.3},
            {"params": model.head.parameters(), "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    steps_per_epoch = math.ceil(len(tr) / args.bs)
    total = args.epochs * steps_per_epoch
    warm = steps_per_epoch
    base_lrs = [g["lr"] for g in opt.param_groups]

    best, history, step = -1.0, [], 0
    for epoch in range(args.epochs):
        model.train()
        t0, run_loss = time.time(), 0.0
        perm = torch.randperm(len(tr), device=dev)
        for i in range(0, len(tr), args.bs):
            f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * f
            idx = perm[i : i + args.bs]
            loss = F.binary_cross_entropy_with_logits(model(augment(tr_x[idx])), tr_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(idx)
            step += 1
        logits, cams = predict(model, va, dev)
        m = scoreboard(logits, cams, va)
        m.update(epoch=epoch + 1, loss=run_loss / len(tr), secs=round(time.time() - t0, 1))
        history.append(m)
        score = m["accuracy"] + 0.25 * m["pointing"]
        tag = ""
        if score > best:
            best = score
            torch.save(model.state_dict(), RUNS / "pigeon_fp32.pt")
            tag = "  * saved"
        print(
            f"ep {m['epoch']:2d}  loss {m['loss']:.4f}  val acc {m['accuracy']:.3f}  "
            f"pointing {m['pointing']:.3f}  hard-neg {m['hard_neg_acc']:.3f}  ({m['secs']}s){tag}"
        )
    (RUNS / "train_history.json").write_text(json.dumps(history, indent=1))


if __name__ == "__main__":
    main()
