# pigeon

Carve a general optical detector down until it does **one thing** — find and
track pigeons — in the smallest, fastest form possible.

Prototype target: this Mac (Apple M4, MPS). Final edge target: TBD.

## Approach

1. **First-light (bird-as-proxy):** run a pretrained YOLO detector, keep only
   COCO's `bird` class. Working pigeon-tracker today, zero training data.
2. **Specialize:** fine-tune on real pigeons so it ignores crows/gulls, and
   collapse the head to a single class.
3. **Carve:** use `harness/` (the doubt-dial) to fake-quantize and prune
   weights, re-checking pigeon accuracy after each cut — "reverse training."
4. **Export:** ship the smallest/fastest artifact for the chosen edge target.

## Why the harness transfers from the LLM version

`harness/quant.py` is byte-identical to the language-model harness —
fake-quantization doesn't care what the weights mean. `harness/model.py` adds
`Conv2d` so it can carve a detector. Pass `yolo.model` to it.

## Files

- `pigeon.py` — detect / track pigeons (bird-proxy) with YOLO
- `benchmark.py` — baseline size / params / latency (the yardstick)
- `harness/` — fake-quantization + prune instrument, retargeted to vision
- `samples/` — drop test images/videos here
- `runs/` — annotated outputs

## The carving cockpit (web UI)

A node-grid where each column is a real YOLOv8n stage (20 nodes = 5% of its
channels, FLOP-weighted). Click nodes to mask channels, **Measure** real bird
mAP@50, and **Export** a genuinely smaller model that honors where you cut.

```bash
uvicorn server:app --port 8000 --app-dir .    # then open http://localhost:8000
```

- Masking is the live, reversible layer (real mAP, estimated compute).
- Export does real dependency-aware channel removal (torch-pruning) and saves
  `runs/pigeon_pruned.pt`. Fine-tune after Export to recover accuracy (structural
  cuts hit the cliff — see STATUS.md).

## Experiments / scripts

```bash
# uses the shared venv at ../weight-harness/.venv
python benchmark.py            # baseline numbers
python carve.py                # quant curve (mAP vs bit-width)
python prune_sweep.py          # structural cliff (mAP vs prune %)
python finetune.py             # prune 20% + fine-tune -> recovery
python pigeon.py --source 0 --track   # live webcam bird tracking
```
