"""pigeon.harness -- the doubt-dial, retargeted to vision.

`quant.py` comes from the LLM harness (fake-quantization is domain-agnostic);
the one change since is that a 1-bit scale averages only the real weights of a
short tail group, not its zero padding. `model.py` adds Conv2d so it can carve
a detector.
"""

from .quant import fake_quantize, quantization_error
from .model import fuse, snapshot, restore, apply_uniform, iter_weight_modules

__all__ = [
    "fake_quantize",
    "quantization_error",
    "fuse",
    "snapshot",
    "restore",
    "apply_uniform",
    "iter_weight_modules",
]
