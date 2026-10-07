## The idea

A one-shot structural cut can take YOLOv8n's bird mAP@50 (average precision for birds, a box counting when it overlaps the true box by at least 50%) to zero: STATUS.md records 0.000 after a uniform 5% cut. The surviving weights still hold most of what the model knew, so fine-tuning the pruned model on bird images lets the remaining channels relearn to cover for the missing ones. README.md records a 30% cut of stage 9 going from 0.126 to 0.395 validation mAP@50 in 2 epochs. The cockpit previews a cut by *masking*: zeroing exactly the channels the real cut would remove.

## How it works

1. Choose a cut fraction per stage, for example `{"model.9": 0.3}`.
2. Plan: ask the pruner which output channels it would remove, whole dependency groups included (see channel_pruning), without removing anything.
3. Preview: in the live model, whose batch norms are folded into the convs, zero those channels' weights and biases, then measure mAP. Shapes don't change, so restoring saved weights undoes it.
4. Commit: load a fresh model, rewrite its `C2f` blocks, and remove the planned channels for real.
5. Make every parameter trainable. If the recipe also quantizes, put the quantizer inside the convs' forward pass (quantization-aware training, QAT).
6. Train with the normal detection loss on 400 bird images, validating on 125 after each epoch and keeping the best checkpoint.
7. Score the model before and after training with the cockpit's own metric. To shrink further, cut again and retrain.

## The code, line by line

`removal_plan(yolo, stage_fracs)` builds the same pruner Export uses, `_stage_pruner(model, stage_fracs, example)`. `pruner.step(interactive=True)` yields each dependency group instead of pruning it. A group is a list of `(dep, idxs)` pairs: `dep.target.module` is the layer touched, `idxs` the channel indices. `pruner.DG.is_out_channel_pruning_fn(dep.handler)` keeps only the entries that remove a conv's *output* channels, the producers. Consumers' input slices are skipped, since a zeroed channel already feeds them nothing. The result is `{conv name: sorted channel indices}`; the model is unchanged. server.py plans on `planner`, an unfused CPU copy also passed through `make_prunable`, so conv names match the live fused model.

`apply_plan(model, plan)` runs under `@torch.no_grad()`. For each listed conv it sets `m.weight[idx] = 0.0` and, when there is a bias, `m.bias[idx] = 0.0`. The bias matters because folding batch norm moves its shift into the conv bias; with both zeroed the channel outputs `SiLU(0) = 0`, exactly what its consumers would see if it were gone. `restore(yolo.model, SNAP)` undoes it, since `snapshot` saves biases too. STATUS.md shows why both halves matter: for stages 15 and 21 at 25%, the old mask (each conv's lowest-L1 filters, weights only) scored 0.1811, Export 0.0088, and the plan-based mask 0.0088.

`recover_job.py:main` is the regrow, run as a subprocess so training never blocks the web server.

- `argparse` reads `--masks` (JSON, stage to fraction), `--quant` (the recipe's precision settings), `--epochs` (default 12) and `--progress`. `qat` is true if `bits` or any `stage_bits` entry is below 16. `write` saves via a temporary file and `os.replace`, so the poller never reads half a file.
- A fresh `YOLO(... "yolov8n.pt")` gets `make_prunable(y)`, then `structured_prune_per_stage(y, masks)` if any fraction is above zero. `p.requires_grad_(True)` on every parameter makes the whole network trainable.
- `tp.utils.count_ops_and_params` measures the pruned model (MACs, stored as `flops_g`).
- `before = cockpit_map(y, quant, ev)` scores a deep copy the way the cockpit does: demote any `QATConv2d` to `Conv2d`, `fuse`, `apply_precision` with the recipe, then score 40 images.
- If `qat`, `enable_qat` swaps each quantized conv's class to `QATConv2d`. Its forward computes `wq` with `quantize_weight` and uses `w + (wq - w).detach()`: the loss sees the rounded weights, the gradient reaches the full-precision ones (a straight-through estimator).
- `y._smart_load("trainer")` returns Ultralytics' `DetectionTrainer`, built with `data=finetune_data/data.yaml`, `imgsz=640`, `batch=16` and `amp=False`. Its `model=yolov8n.pt` override only satisfies the constructor; `trainer.model = y.model.to(device)` swaps in the pruned network. STATUS.md explains why it survives: `setup_model()` returns early when `self.model` is already an `nn.Module`.
- `on_epoch(tr)` runs on `on_fit_epoch_end`. It reads `tr.metrics["metrics/mAP50(B)"]`, Ultralytics' box mAP@50 on the 125 validation images, and appends it to `state["curve"]`. The guard `len(state["curve"]) >= a.epochs` drops the extra call Ultralytics makes when it re-validates `best.pt`.
- After `trainer.train()`, `state["after"]` is `cockpit_map` of `YOLO(str(trainer.best))`. Any exception writes `{"status": "error", ...}` and re-raises.

`/api/recover` in server.py launches the job with `subprocess.Popen` and the current recipe; `/api/recover_status` returns the JSON file.

## What the simulations show

These are toy-MLP analogues, not YOLO. In `prune_structured` (lab/results/prune_structured.json), a 3x256 teacher cut to width 64 scores 0.1685 accuracy with no recovery and 0.8909 after fine-tuning on labels, the detector's fall-then-recover shape.

`specialist_data` (lab/results/specialist_data.json) asks whether pigeon's all-bird recovery set is enough. A teacher pruned to 3x32 is recovered on different data and scored by bird average precision ("siblings" are the three bird-like classes; means of 2 seeds):

| recovery data | pruned 3x32 | fresh 2x16 |
|---|---|---|
| everything (28k) | 0.9679 | 0.8933 |
| birds only (1403) | 0.7971 | 0.7213 |
| birds + siblings | 0.8856 | 0.8571 |
| birds + equal # random others | 0.962 | 0.9246 |
| birds + noise negatives | 0.9406 | 0.8835 |

Birds alone gave the worst recovery; adding as many non-birds recovered almost everything. pigeon's set is less extreme: its photos contain background, which the detection loss treats as negatives.

## When it breaks

- A deeper one-shot cut recovers far less. One 20% cut fine-tuned for 40 epochs reached only 0.0706, while three 7.2% cuts with 15 epochs after each kept 0.2874 with 31% fewer params (STATUS.md, native 0.4284). Each step lost more: 0.3943, 0.3377, 0.2874.
- Recovered numbers lean optimistic. The validation split is the same 125 images as `eval_set/`, and `best.pt` is picked on it.
- The curve (Ultralytics' mAP50, 125 images, QAT active) and before/after (cockpit scorer, 40 images, fused model) are different metrics.
- QAT trains per-tensor recipes per-channel (README.md), and `QATConv2d` always fits the range to the maximum, so an MSE-clipped recipe is trained on one grid and scored on another.
- Masking equals Export only on the server. The browser edition can't run torch-pruning, so it masks each conv's lowest-L1 channels on its own, an optimistic preview.

## Going further

To run it, build both datasets (`python build_eval_set.py`, `python build_finetune_set.py`), start `uvicorn server:app --port 8000 --app-dir .` and open http://localhost:8000. In Structure, press Scan, then "Apply cut" or click nodes. In Regrow, pick 6, 12 or 30 epochs and press Regrow. Headless: `python recover_job.py --masks '{"model.9":0.3}' --quant '{"bits":4}' --epochs 12 --progress runs/recover_progress.json`, or `iterative_prune.py` for repeated cuts (not yet a cockpit button). QAT in the same loop took a 3.16-bit recipe from 0.003 to 0.584, 70% of native, in 3 CPU epochs (STATUS.md, on a VOC-2007 bird stand-in set). STATUS.md's next steps are more bird images (2k or more), more epochs, and auto-shrink in the cockpit. The toy loop is `lab/compress.py:structured_prune` followed by `ce_ft` or `kd_ft` in `lab/experiments/shrink.py`.
