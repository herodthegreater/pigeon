"""De-risk: can torch-pruning cut channels out of YOLO's branchy graph without
breaking the forward pass? Prunes 30% globally by L1, sparing the Detect head,
then verifies forward + measures the real size/FLOPs/latency change.
"""

from __future__ import annotations

import time

import torch
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

RATIO = 0.30

yolo = YOLO("yolo11n.pt")
model = yolo.model.cpu().eval()
example = torch.randn(1, 3, 640, 640)

macs0, params0 = tp.utils.count_ops_and_params(model, example)

# Ignore only the head's LEAF conv layers (their output channels must stay
# fixed = reg/cls outputs). Ignoring the whole Detect container over-constrains
# the graph and prunes nothing.
ignored = []
for m in model.modules():
    if isinstance(m, Detect):
        ignored += [c for c in m.modules() if isinstance(c, torch.nn.Conv2d)]
print(f"ignored {len(ignored)} head conv layers")

probe = model.model[1].conv  # a backbone conv, out=32 natively
print(f"probe backbone conv out_channels BEFORE: {probe.out_channels}")

imp = tp.importance.MagnitudeImportance(p=2)
pruner = tp.pruner.MetaPruner(
    model, example, importance=imp,
    pruning_ratio=RATIO, ignored_layers=ignored, global_pruning=True,
)
pruner.step()
print(f"probe backbone conv out_channels AFTER : {probe.out_channels}")

with torch.no_grad():
    out = model(example)  # <- the moment of truth: does it still run?

macs1, params1 = tp.utils.count_ops_and_params(model, example)
print(f"\nforward OK after pruning {int(RATIO*100)}% of channels (head spared)")
print(f"params : {params0/1e6:6.3f}M -> {params1/1e6:6.3f}M  ({100*(params1-params0)/params0:+.0f}%)")
print(f"MACs   : {macs0/1e9:6.3f}G -> {macs1/1e9:6.3f}G  ({100*(macs1-macs0)/macs0:+.0f}%)")

dev = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(dev)
x = example.to(dev)
sync = torch.mps.synchronize if dev == "mps" else (lambda: None)
with torch.no_grad():
    for _ in range(10):
        model(x)
    sync()
    t = time.time()
    for _ in range(50):
        model(x)
    sync()
    dt = (time.time() - t) / 50
print(f"latency: {dt*1000:.2f} ms/frame ({1/dt:.1f} FPS) on {dev}  [native was 8.86ms / 113 FPS]")
