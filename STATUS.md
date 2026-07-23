# pigeon — status snapshot

## Baselines (this M4, MPS)
- **yolo11n**: 2.62 M params · 5.61 MB · 113 FPS · bird mAP@50 = 0.4501
- **yolov8n**: 3.16 M params · 4.40 GMAC · ~86 FPS · bird mAP@50 = 0.4288
- Eval set: `eval_set/` = 125 COCO val bird images + 440 boxes (YOLO format)

## What works end-to-end
- **Fake-quant harness** on the real detector (`harness/quant.py`, `model.py`)
- **Accuracy backbone**: `harness/evaluate.py` `BirdmAPEvaluator` (single-class AP@50)
- **Quant confession curve** (`carve.py`): 8-bit ~free; 6-bit alive *only* with
  group_size=128 scales; 4-bit collapses. Uniform-quant floor ~6-bit.
- **REAL structured channel removal** (`harness/prune.py`, `real_prune_test.py`) —
  now works. Key fix: swap Ultralytics `C2f` -> `C2f_v2` (splits the `chunk()`
  the tracer can't follow), model in **train()** for tracing, ignore `Detect`,
  `GroupNormPruner` + `GroupMagnitudeImportance`. **Base = YOLOv8n** (YOLO11's
  C3k2/C2PSA have no equivalent recipe).
  - 30% prune: params 3.16M->1.82M (-43%), MACs 4.40G->2.86G (-35%), forward OK.

## Two hard, load-bearing findings
1. **Mac latency != FLOPs.** 30% prune cut MACs 35% but latency got *worse*
   (+6%) on MPS -- overhead-bound + non-multiple-of-8 channels. Show FLOPs, not
   Mac latency; speed win is for the edge target.
2. **Structural pruning is a CLIFF** (`prune_sweep.py`): even **5% one-shot prune
   -> mAP 0.000**. Unlike quant, there's no graceful structural headroom.
   => Fine-tuning is MANDATORY and interleaved: the loop is prune -> fine-tune ->
   test. The live-mask UX must show *recovery*, not just the drop.

## Fine-tune recovery — LOOP CLOSED (`finetune.py`, `build_finetune_set.py`)
Prune YOLOv8n 20% (mAP 0) -> fine-tune the *pruned* model 40 epochs on 400 COCO
bird images -> **mAP recovers 0.000 -> 0.0706** (P/R 0 -> 0.20/0.14). Cut-and-
regrow works end-to-end on the real model. Recovery is *partial* (vs 0.43
native) — bottleneck is the tiny 400-image train set + a hard one-shot cut, not
the method. How it works: drive `DetectionTrainer` directly, set
`trainer.model = <pruned module>` (8.4 `setup_model()` returns early on an
nn.Module, so the pruned arch survives; `_setup_train` still sets nc/names/args).
Gotcha: background shells don't inherit cwd — use absolute paths.

## Design consequences (locked)
- Auto-prune "training mode" = prune -> fine-tune -> test, repeat.
- Interactive demo value = "how small can I shrink while regrowing mAP back",
  not "watch accuracy fall".
- Open question: per-STAGE cuts (what the UI does) may be gentler than the
  GLOBAL prune swept here -- worth testing.

## Next steps
1. **Stronger recovery** (loop is proven; now push the number up): more train
   data (2k+ bird imgs), more epochs, and/or iterative small prunes with
   fine-tuning between each cut — target mAP back near native 0.43.
2. Per-stage prune control (`pruning_ratio_dict`) + FLOP-weighted columns
   (stages 3,5,head,1,7 ~= 64% of compute on yolo11n; recompute for v8n).
3. FastAPI backend + node-grid front page (masking preview + real Export).
4. Snap channel counts to multiples of 8.

## Env
Shared venv: `../weight-harness/.venv` (torch, transformers, ultralytics,
fiftyone, torch-pruning). COCO cache `~/fiftyone`. Base weights: yolo11n.pt,
yolov8n.pt in project root.
