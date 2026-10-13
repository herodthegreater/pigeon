"""Check that the cockpit's live masks are Export, exactly.

For several per-stage recipes this builds the model three ways, the way
server.py does: plan the channels on an unfused copy (`removal_plan`), zero
them in the fused model (`apply_plan`), and separately prune a fresh copy for
real (`structured_prune_per_stage`, what Export runs). The masked and the
pruned model must give the same outputs on the same input. It also reports
whether the plan differs from reading torch-pruning's groups *without* pruning
them, the shortcut `removal_plan` used before it replayed Export's loop.

    python check_masking.py          # downloads yolov8n.pt on first run; CPU, ~1 min
"""

from __future__ import annotations

import os

import torch
from ultralytics import YOLO

from harness.mask import apply_plan
from harness.model import fuse
from harness.prune import _stage_pruner, make_prunable, removal_plan, structured_prune_per_stage

HERE = os.path.dirname(os.path.abspath(__file__))
PT = os.path.join(HERE, "yolov8n.pt") if os.path.exists(os.path.join(HERE, "yolov8n.pt")) else "yolov8n.pt"

RECIPES = [
    {"model.9": 0.3},
    {"model.0": 0.25},
    {"model.2": 0.3, "model.4": 0.3, "model.6": 0.3},
    {"model.12": 0.5, "model.15": 0.5, "model.18": 0.5, "model.21": 0.5},
    {f"model.{i}": 0.2 for i in range(0, 22)},
]


def shortcut_plan(yolo, fracs):
    """The old way: read every group's indices without pruning any group."""
    names = {m: n for n, m in yolo.model.named_modules()}
    pruner = _stage_pruner(yolo.model, fracs)
    plan: dict[str, set[int]] = {}
    for group in pruner.step(interactive=True):
        for dep, idxs in group:
            m = dep.target.module
            if isinstance(m, torch.nn.Conv2d) and pruner.DG.is_out_channel_pruning_fn(dep.handler):
                plan.setdefault(names[m], set()).update(int(i) for i in idxs)
    return {n: sorted(ix) for n, ix in plan.items()}


def built(fracs, how):
    y = YOLO(PT)
    make_prunable(y)
    if how == "pruned":
        structured_prune_per_stage(y, fracs)
    y.model.eval()
    fuse(y)
    return y


def main():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 320, 320)
    ok = True
    for fracs in RECIPES:
        planner = YOLO(PT)
        make_prunable(planner)
        plan = removal_plan(planner, fracs)
        old = shortcut_plan(planner, fracs)
        masked = built(fracs, "masked")
        apply_plan(masked.model, plan)
        pruned = built(fracs, "pruned")
        with torch.no_grad():
            a, b = masked.model(x)[0], pruned.model(x)[0]
        diff = (a - b).abs().max().item()
        same = diff < 1e-3 * b.abs().max().item()
        ok &= same
        n = sum(len(v) for v in plan.values())
        moved = sum(len(set(plan.get(k, ())) ^ set(old.get(k, ()))) for k in set(plan) | set(old))
        print(f"{'ok ' if same else 'BAD'} {len(fracs):2d} stage(s): {n:5d} channels, "
              f"masked vs pruned max |diff| {diff:.2e}; shortcut plan differs in {moved} channel(s)")
    print("masking matches Export" if ok else "MASKING DOES NOT MATCH EXPORT")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
