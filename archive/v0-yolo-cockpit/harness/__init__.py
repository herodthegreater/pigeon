"""pigeon.harness -- the doubt-dial, retargeted to vision.

`quant.py` is byte-identical to the LLM harness (fake-quantization is
domain-agnostic). `model.py` adds Conv2d so it can carve a detector.
"""

from .quant import fake_quantize, quantization_error
from .model import snapshot, restore, apply_uniform, iter_weight_modules

__all__ = [
    "fake_quantize",
    "quantization_error",
    "snapshot",
    "restore",
    "apply_uniform",
    "iter_weight_modules",
]
