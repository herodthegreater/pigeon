## The idea

A one-shot structural cut takes YOLOv8n's bird mAP@50 (average precision for birds, a box counting when it overlaps the true box by at least 50%) to zero. The surviving weights still hold most of what the model knew, so fine-tuning the pruned model on bird images lets the remaining channels relearn to cover for the missing ones. On the real model, a light cut at stage 9 recovered from 0.00 to 0.4395 validation mAP@50 in about 2 epochs, and a cut of about 12%, fine-tuned for 45 epochs, reached 0.388, 91% of the native 0.428 (STATUS.md).

## How it works

1. Choose a cut fraction per stage, for example `{"model.9": 0.3}`. In the cockpit you preview it with *masking*: zero the weakest filters and measure real mAP without changing any shape.
2. Load a fresh model, rewrite its `C2f` blocks so they can be traced, and remove the channels for real (see channel_pruning).
3. Make every parameter trainable.
4. Train the pruned model with the normal detection loss on bird images: 400 training and 125 validation images in pigeon.
5. After each epoch, measure validation mAP@50 and keep the best checkpoint.
6. To shrink further, cut again and retrain. This step is planned but not working yet (see below).

## The code, line by line

`mask_stage(model, stage, frac)` is the preview. `_stage_convs` collects every `Conv2d` in `stage`. For each, `k = min(int(round(frac * out)), out - 1)` filters get zeroed, always keeping one. `l1 = w.abs().flatten(1).sum(1)` is the L1 norm (sum of absolute values) of each output filter. `idx = torch.argsort(l1)[:k]` picks the k smallest and `w[idx] = 0.0` zeros them in place. Shapes never change, so `restore(yolo.model, SNAP)` undoes it. Each conv picks its filters independently.

`recover_job.py:main` is the regrow, run as a separate process so training never blocks the web server.

- `argparse` reads `--masks` (JSON, stage to fraction), `--epochs` (default 12) and `--progress` (a JSON file path).
- `device` is `"mps"` on Apple Silicon, otherwise `"cpu"`.
- `YOLO(os.path.join(HERE, "yolov8n.pt"))` loads a fresh, unfused model. `make_prunable(y)` swaps in `C2f_v2`, and `structured_prune_per_stage(y, masks)` removes channels if any fraction is above zero.
- `p.requires_grad_(True)` on every parameter makes the whole network trainable.
- `tp.utils.count_ops_and_params` measures the pruned model; the state records `params_m`, `flops_g` and `curve: [0.0]`.
- `y._smart_load("trainer")` returns Ultralytics' `DetectionTrainer`, built with `data=finetune_data/data.yaml`, `imgsz=640`, `batch=16`, `workers=0`, `amp=False` (no mixed precision), `plots=False`, writing to `runs/recover`.
- The `model=yolov8n.pt` override only satisfies the constructor. `trainer.model = y.model.to(device)` swaps in the pruned network. STATUS.md explains why it survives: `setup_model()` returns early when `self.model` is already an `nn.Module`.
- `on_epoch(tr)` runs on `on_fit_epoch_end`, after validation. It reads `tr.metrics["metrics/mAP50(B)"]`, Ultralytics' box mAP@50 on the 125 validation images, appends it to `state["curve"]` and rewrites the progress file.
- `trainer.train()` runs the loop; `state["best"]` then records `trainer.best` (`runs/recover/weights/best.pt`). Any exception writes `{"status": "error", ...}` and re-raises.

`/api/recover` in server.py starts the job with `subprocess.Popen`, refusing while one runs; `/api/recover_status` returns the JSON file.

## What the simulations show

These are toy-MLP analogues, not YOLO. In `prune_structured` (lab/results/prune_structured.json), a 3x256 teacher (accuracy 0.9139) cut to width 64 by L1 ranking scores 0.1685 with no recovery and 0.8727 after fine-tuning on labels. That is the same fall-then-recover shape as the detector.

`specialist_data` (lab/results/specialist_data.json) asks whether pigeon's all-bird recovery set is enough. A teacher pruned to 3x32 is recovered on different data and scored by bird average precision (teacher: 0.9885). "Siblings" are the three bird-like classes. Means of 2 seeds:

| recovery data | pruned 3x32 | fresh 2x16 |
|---|---|---|
| everything (28k) | 0.8379 | 0.88 |
| birds only (1403) | 0.7424 | 0.8745 |
| birds + siblings | 0.703 | 0.9097 |
| birds + equal # random others | 0.9053 | 0.9148 |
| birds + noise negatives | 0.8534 | 0.8905 |

Birds alone gave the pruned model its worst recovery; adding an equal number of non-birds gave its best. The toy's "birds only" is stricter than pigeon's set, whose photos also contain background and other objects that the detection loss treats as negatives.

## When it breaks

- A deeper one-shot cut recovers far less. Pruning 20% and fine-tuning 40 epochs on the 400 images reached only 0.0706 (STATUS.md).
- Iterative cut-and-regrow does not work yet: re-pruning a reloaded checkpoint does nothing, and params stay at 2.775M across steps.
- Masking only approximates Export. It picks filters per conv by L1, while Export picks whole dependency groups by magnitude. And server.py fuses batch norm into each conv's bias before masking, so a masked channel still outputs a constant `SiLU(bias)` rather than nothing. Treat the live mAP as an estimate of what Export plus recovery starts from.
- `curve` starts at a hard-coded `0.0`, not a measured score of the pruned model.
- The live mAP (40 images, `BirdmAPEvaluator`) and the recovery mAP (125 validation images, Ultralytics' metric) use different images and code. Don't compare them directly.
- `build_finetune_set.py` writes to a hard-coded absolute path, but `recover_job.py` reads `finetune_data/` next to itself. On another machine, fix the path first.

## Going further

To run it, build both datasets (`python build_eval_set.py`, `python build_finetune_set.py`), start `uvicorn server:app --port 8000` and open http://localhost:8000. In Structure, press Scan, then "Apply cut" or click nodes, and "Measure this cut". In Regrow, choose 6, 12 or 30 epochs and press Recover; the panel shows mAP after each epoch. Headless: `python recover_job.py --masks '{"model.9":0.3}' --epochs 12 --progress runs/recover_progress.json`, or `finetune.py` and `iterative_prune.py`. The next steps listed in STATUS.md are more bird images (2k or more), longer training, and small repeated cuts. The toy loop is `lab/compress.py:structured_prune` followed by `ce_ft` or `kd_ft` in `lab/experiments/shrink.py`.
