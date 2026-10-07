"""Export Pigeon for the browser: fold BatchNorm, then ONNX with weights as graph inputs.

Outputs (site/public/model/):
  pigeon.onnx     graph only; inputs = image + every weight tensor
  weights.bin     all weights, float32 little-endian, concatenated
  weights.json    [{name, shape, offset, numel, kind, layer}] in graph-input order

Why weights-as-inputs: the page keeps each tensor as a Float32Array, applies a
compression method to it (quantize, prune, cluster, SVD...) and simply feeds
the edited arrays on the next run — no model re-serialization per slider move.

Why fold BatchNorm first: deployment runtimes fold it too, so live quantization
acts on the same conv weights a real int8 / int4 export would quantize.

Run:  python -m train.export
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from .model import Pigeon

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "site/public/model"


def fold_batchnorm(model: nn.Module) -> nn.Module:
    """Replace every Conv2d -> BatchNorm2d pair with a single biased Conv2d."""
    for parent in model.modules():
        kids = list(parent.named_children())
        for (n1, m1), (n2, m2) in zip(kids, kids[1:]):
            if isinstance(m1, nn.Conv2d) and isinstance(m2, nn.BatchNorm2d):
                setattr(parent, n1, fuse_conv_bn_eval(m1, m2))
                setattr(parent, n2, nn.Identity())
    return model


def kind_of(name: str, p: torch.Tensor, mod: nn.Module) -> str:
    if name.startswith("head"):
        return "head"
    if isinstance(mod, nn.Conv2d):
        if ".fc1" in name or ".fc2" in name:
            return "squeeze_excite"
        if mod.groups > 1 and mod.groups == mod.in_channels:
            return "depthwise"
        if mod.kernel_size == (1, 1):
            return "pointwise"
        return "conv"
    return "other"


class WithCam(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, return_cam=True)


def initializers_to_inputs(path: Path, names: list[str]) -> None:
    """Turn each named weight initializer into a required graph input (same name, shape, dtype)."""
    m = onnx.load(path)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    missing = [n for n in names if n not in inits]
    if missing:
        raise RuntimeError(f"weights not found as initializers: {missing[:5]}")
    existing = {i.name for i in g.input}
    for n in names:
        init = inits[n]
        g.initializer.remove(init)
        if n not in existing:
            g.input.append(onnx.helper.make_tensor_value_info(n, init.data_type, list(init.dims)))
    onnx.checker.check_model(m)
    onnx.save(m, path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = Pigeon(pretrained=False)
    model.load_state_dict(torch.load(RUNS / "pigeon_fp32.pt", map_location="cpu"))
    model.eval()
    ref_x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        ref_logit, ref_cam = model(ref_x, return_cam=True)
    fold_batchnorm(model)

    modules = dict(model.named_modules())
    params = [(n, p.detach()) for n, p in model.named_parameters()]
    names = [n for n, _ in params]
    onnx_names = [f"model.{n}" for n in names]
    torch.onnx.export(
        WithCam(model).eval(),
        (ref_x,),
        OUT / "pigeon.onnx",
        input_names=["image"],
        output_names=["logit", "cam"],
        dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}, "cam": {0: "batch"}},
        opset_version=17,
        do_constant_folding=False,
        dynamo=False,
    )
    initializers_to_inputs(OUT / "pigeon.onnx", onnx_names)

    manifest, offset, blobs = [], 0, []
    for n, p in params:
        arr = p.float().contiguous().numpy()
        mod = modules[n.rsplit(".", 1)[0]]
        manifest.append(
            {
                "name": f"model.{n}",
                "shape": list(arr.shape),
                "offset": offset,
                "numel": int(arr.size),
                "kind": kind_of(n, p, mod),
                "role": "bias" if n.endswith("bias") else "weight",
                "layer": n.rsplit(".", 1)[0],
            }
        )
        blobs.append(arr.astype("<f4").tobytes())
        offset += arr.nbytes
    (OUT / "weights.bin").write_bytes(b"".join(blobs))
    (OUT / "weights.json").write_text(json.dumps(manifest))

    # parity check: folded ONNX in onnxruntime vs the original PyTorch model
    sess = ort.InferenceSession(str(OUT / "pigeon.onnx"), providers=["CPUExecutionProvider"])
    feeds = {"image": ref_x.numpy(), **{f"model.{n}": p.numpy() for n, p in params}}
    logit, cam = sess.run(["logit", "cam"], feeds)
    dl = np.abs(logit - ref_logit.numpy()).max()
    dc = np.abs(cam - ref_cam.numpy()).max()
    total = sum(m["numel"] for m in manifest)
    print(f"exported {len(manifest)} tensors, {total:,} params, {offset/1e6:.2f} MB")
    print(f"onnx {Path(OUT/'pigeon.onnx').stat().st_size/1e3:.0f} KB graph   parity: max |dlogit| {dl:.2e}  max |dcam| {dc:.2e}")
    assert [i.name for i in sess.get_inputs()] == ["image", *onnx_names], "graph inputs out of order"


if __name__ == "__main__":
    main()
