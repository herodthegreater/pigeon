# Pigeon

**Learn model compression by doing it.** Pigeon takes a real image classifier, teaches it one job — *is there a bird, and where?* — and lets you shrink it with every major compression technique, right in your browser, while it shows you exactly what each one costs.

> **Status:** v1 in development. The plan and every design decision live in [`docs/PLAN.md`](docs/PLAN.md).
> The original YOLOv8n "reverse-training cockpit" (v0) is preserved in [`archive/v0-yolo-cockpit/`](archive/v0-yolo-cockpit/) and tagged `v0-yolo-cockpit`; its measured findings are in [`docs/findings-yolo.md`](docs/findings-yolo.md).

## What it will do

- **Flight School** — a guided course through the 12 building blocks of compression: number formats, scales, bit-widths, granularity, outliers, calibration, mixed precision, codebooks, pruning, QAT, distillation, low-rank.
- **The Coop** — a sandbox: stack methods, watch accuracy, pointing score, agreement and true size update live, and chart every attempt on a size-vs-accuracy Pareto plot.
- **The Field Guide** — a card for every technique (GGUF k-quants, AWQ, GPTQ, FP8, FP4, NF4/QLoRA, SmoothQuant, rotations, KV-cache quantization, 2:4 sparsity, distillation, and more), each at three depths with copy-paste code for real tools.

The mascot is the scoreboard: shrink the model and the chonky pigeon slims into a dove, a swift, a swallow, a hummingbird. Push too far and it starves — or turns to bones.

## Layout

```
train/     PyTorch pipeline: data, specialization, evaluation, ONNX export, precomputed variants
site/      the static web app (coming in Phase 2)
docs/      plan and findings
archive/   v0 YOLOv8n cockpit
```

## Quickstart (training pipeline)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r train/requirements.txt
python -m train.data        # download + prepare the Open Images bird / no-bird set
python -m train.train       # specialize MobileNetV3-Small into Pigeon
python -m train.evaluate    # accuracy, pointing score, size
```

## Data

Images and box annotations come from [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) (annotations CC-BY 4.0; images listed as CC-BY 2.0 by their authors). Per-image attribution ships with the site's test set.
