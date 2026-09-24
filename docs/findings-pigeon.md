# Pigeon v1 — measured findings

Model: MobileNetV3-Small, specialized to bird / no-bird with a CAM head (`train/`).
Eval set: 500 Open Images validation images (250 birds, 100 hard negatives). Apple M4.
Numbers regenerate with `python -m train.evaluate` and `python -m train.sweep`.

## Specialization (general → Pigeon)

| Model | Params | FP32 size | Accuracy | Hard-negative acc | Pointing |
|---|---|---|---|---|---|
| General ImageNet model ("top-1 is a bird class") | 2,542,856 | 10.17 MB | 93.4 % | 100 % | — |
| Pigeon (fine-tuned 7 epochs, ~3 min) | 927,585 | 3.71 MB | 94.8 % | 96.0 % | 98.0 % |

- Dropping the 1000-way classifier removes **63 %** of the parameters.
- Accuracy rises only ~1.4 pts: the general model is already a decent bird-spotter. **Specialization is mostly a size win** (plus the "where" heatmap).
- After BatchNorm folding: 921,529 params, 3.69 MB. **Half of them (462 K) are squeeze-and-excitation layers**; depthwise convs are only 61 K (7 %).

## Training-free compression (BN-folded weights, round-to-nearest, symmetric)

| Method | Accuracy | Pointing |
|---|---|---|
| FP32 | 94.8 % | 98.0 % |
| INT8 per-tensor | **50.0 %** (collapse) | 38 % |
| INT8 per-channel | 95.0 % | 98.0 % |
| INT6 per-channel / per-group(32) | 91.0 / 91.8 % | 96.4 / 97.6 % |
| INT5 per-channel / per-group(32) | 78.2 / 79.8 % | 93.2 / 91.2 % |
| INT4 per-channel / per-group(32) | 51.0 / 50.8 % | 59 / 50 % |
| Unstructured prune 30 % / 50 % | 59.2 / 50.0 % | 79 / 32 % |
| 2:4 sparsity | 50.0 % | 29 % |
| Channel prune 5 % | 50.0 % | 37 % |

- **Per-tensor INT8 collapses.** Folding BatchNorm into depthwise convs gives channels wildly different ranges; one shared scale crushes the small ones (the classic MobileNet result behind per-channel quantization and cross-layer equalization).
- Collapses are real damage, not just a shifted threshold: ranking quality (AUC) falls from 0.988 to ~0.62–0.66 at INT4.

## Where the fragility lives (INT4 per-channel applied to one part only)

| Quantized part | Share of weights | Accuracy |
|---|---|---|
| Squeeze-excite (fc1/fc2) | 50 % | 94.6 % |
| Head / first conv / last block | < 1 % each | 94.6–94.8 % |
| Pointwise convs | 43 % | 75.8 % |
| Depthwise convs | 7 % | 58.0 % |

Pruning 30 % of only the SE layers: 95.2 % (free). Pruning 30 % of everything else: 58.8 %.

**Lesson:** uniform settings fail fast; sensitivity-aware mixed precision (SE low-bit, depthwise high-bit) is how you actually shrink this model — then QAT / distillation for the rest.
