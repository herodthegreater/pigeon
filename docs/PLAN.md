# Pigeon v1 — plan

Decisions from the 2026-09-23 design grill. This is the source of truth; change it here when a decision changes.

## Goal

A free, static, in-browser site that teaches people how model compression works — and how to do it to their own models, LLMs included — by letting them compress one real model with their own hands and see exactly what each method costs.

The model: a general ImageNet classifier, specialized to one job — **is there a bird, and where?** — then shrunk to the smallest version that still does that job.

## Locked decisions

| Area | Decision |
|---|---|
| Task | Binary bird / no-bird classifier + class-activation map (CAM) showing where it looks. No bounding-box detector. |
| Base model | torchvision MobileNetV3-Small (ImageNet-1k, 2.54 M params, ~10 MB FP32). |
| Story arc | general (1000 classes) → specialized (bird head) → compressed → distilled. |
| Distillation student | Tiny custom CNN, used only in the distillation demos. |
| Runtime | Hybrid static site. Training-free methods run **live** in the browser; training-based methods (QAT, KD, prune+fine-tune…) are **precomputed** on a grid of settings. No server. |
| Data | Open Images V7 (CC-BY 4.0 annotations). Train on the `test` split, evaluate on the `validation` split. Bird boxes give the pointing score. |
| Test set | Fixed ~500 images shipped with the site (≈50% birds, ≈20% hard negatives: kites, planes, bats, butterflies, balloons…). Per-image attribution shipped alongside. |
| Play images | User upload (file input). Not counted in metrics. (Camera is not available inside the Artifact viewer; works on the GitHub Pages build.) |
| Metrics | Accuracy + Δ vs original · pointing score (CAM peak inside a true bird box) · agreement with original · true size in bytes · speed = estimated + measured offline on real hardware, never fake browser timing. |
| Scoreboard extras | Pareto chart (size vs accuracy) of every setting tried, a dashed "still works" line (accuracy ≥ 90 %, pointing ≥ 75 %), and a "smallest working Pigeon so far" badge. No points or leaderboard. |
| Site structure | **Flight School** (guided lessons = Tier 1 in order) · **The Coop** (stackable sandbox) · **The Field Guide** (every method card). |
| Depth | Every card has three layers: the gist · how it works · under the hood — plus "Do it yourself" code for real tools. |
| Look | Field-guide aesthetic. Mascot bird whose **species tracks size** and **condition tracks accuracy**. |
| LLM coverage | LLM-only methods get toy simulations in an **LLM wing** plus real commands (llama.cpp, bitsandbytes, AutoAWQ, GPTQModel, MLX…). |
| Repo | v0 YOLO cockpit tagged `v0-yolo-cockpit` and moved to `archive/v0-yolo-cockpit/` (since un-archived as `cockpit/`, where it kept evolving to v3); its findings are lesson material (`docs/findings-yolo.md`). |
| Hosting | GitHub Pages (pending the owner's go-ahead to make the repo public). |

## The mascot

**Species ← compression ratio vs the specialized FP32 Pigeon (3.69 MB)**, so the Coop starts with a fat bird. Thresholds sit just under the round numbers so that INT8 (3.8× once scales and FP32 biases are counted) reaches the swift:

| Ratio | Species | Real mass |
|---|---|---|
| 1× | Rock Pigeon, extra chonky | ~350 g |
| ≥1.8× | Mourning Dove | ~120 g |
| ≥3.5× | Common Swift | ~40 g |
| ≥7× | Barn Swallow | ~18 g |
| ≥14× | Ruby-throated Hummingbird | ~3 g |

**Condition ← metrics:** healthy (meets the "still works" line) · emaciated (below it, above chance; worsens continuously) · skeleton (accuracy ≲ 60 % or pointing collapses). Easter egg: breaking the uncompressed pigeon shows a dodo skeleton.

Art: stylized field-guide SVG, 5 species × 3 conditions, with interpolation for the emaciated range.

## Method scope (importance-ranked)

Ranking weights: building block ×3 · real-world use in 2026 ×2 · field impact ×1.

**Tier 1 — building blocks (Flight School syllabus, in order)**
1. What a weight is: FP32 → FP16 / BF16
2. Scale and zero-point (symmetric vs asymmetric)
3. Bit-width + round-to-nearest (INT8 → INT4 → INT2 → binary)
4. Granularity: per-tensor → per-channel → per-group
5. Outliers
6. Activations and calibration (weight-only vs W8A8, static vs dynamic, min-max vs percentile)
7. Sensitivity and mixed precision
8. Codebooks and clustering (non-uniform levels)
9. Pruning (unstructured vs structured; the structural cliff)
10. Recovery: fine-tuning and QAT (fake quant + straight-through estimator)
11. Knowledge distillation
12. Low-rank factorization (SVD)

**Tier 2 — used in practice (2026)**
GGUF k-quants + importance matrix · AWQ · GPTQ · FP8 (E4M3 / E5M2) · FP4 microscaling (NVFP4 / MXFP4) · bitsandbytes NF4 + QLoRA · SmoothQuant · rotations (QuaRot / SpinQuant / ParoQuant) · KV-cache quantization · MLX · 2:4 sparsity · prune + distill (Minitron-style) and LLM distillation · quantization-aware distillation · deployment runtimes (TorchAO, LiteRT, Core ML)

**Tier 3 — field-shaping (full cards, shared interactive)**
LLM.int8() · SparseGPT · Wanda · BitNet b1.58 · QuIP# / AQLM / QTIP · HQQ · AdaRound / BRECQ · OBS / OBQ · HAWQ · Deep Compression · LSQ / PACT · lottery ticket · EXL2 / EXL3 · speculative decoding · MoE · GQA / MLA · layer dropping · BatchNorm folding · depthwise-separable convolutions

**Tier 4 — index entries**
DoReFa · stochastic rounding · Tucker / CP · movement pruning · attention transfer · self-distillation · token merging · early exit · vocabulary trimming · LoftQ · EfficientQAT · OmniQuant · SpQR · SqueezeLLM · VPTQ · OCGQuant · FlashAttention (myth-buster) · NAS · ONNX QDQ · TensorRT internals

v1 = Tiers 1 + 2 fully built; Tier 3 complete cards with a shared interactive; Tier 4 index.

## Architecture

```
train/  (PyTorch, Apple MPS)                site/  (Astro + TypeScript, static)
  data.py       Open Images → 256px set       engine/   ONNX Runtime Web inference
  model.py      MobileNetV3-S + CAM head                weights are graph *inputs* →
  train.py      specialize (fine-tune)                  JS edits Float32Arrays live
  evaluate.py   acc / pointing / agreement    ops/      quantize · prune · cluster ·
  export.py     fold BN → ONNX (weights as              SVD · GPTQ/AWQ-lite … in JS
                inputs) + weights.bin         metrics/  scoreboard · Pareto · mascot
  variants/     QAT · KD · prune+FT grids     content/  Flight School · Field Guide
  bench/        real-hardware latency                   (MDX cards, 3 layers each)
```

Key technical choices:
- **CAM head, not Grad-CAM.** The 1000-way classifier (≈1.6 M of the 2.54 M params — "63 % of Pigeon was for the other 999 things") is replaced by global-average-pool → one linear logit. The heatmap is then exactly `w · A` over the final 576×7×7 feature map — no gradients needed in the browser.
- **BatchNorm folded before export**, so live quantization acts on the same conv weights a real deployment would quantize.
- **Weights as ONNX inputs.** The graph takes every weight tensor as an input; the page holds them as typed arrays, applies a method, and re-runs. No model re-serialization per slider move.
- **Activation quantization** via gated fake-quant nodes in the exported graph (`x + g·(fq(x; s, z) − x)`), so W8A8 / calibration lessons also run live.
- **Metrics honesty:** size is computed from the real encoding (bits + scale/zero-point/codebook overhead + sparsity metadata); speed is never timed in the browser.

## Phases

0. Archive v0, scaffold v1 — *done*
1. Training pipeline: data, specialized Pigeon, fixed test set, baseline metrics — *done*
2. Browser engine: live quantize / prune, scoreboard, Pareto chart, mascot — *core done* (Coop live; speed estimates move to Phase 5; photo upload pending)
3. Flight School (12 lessons)
4. Field Guide Tier 2 + LLM wing + DIY code
5. Precomputed variants (QAT, KD, prune + fine-tune) + real-hardware benchmarks
6. Tier 3 / 4 cards, polish, deploy
