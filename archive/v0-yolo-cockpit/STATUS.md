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

## Recovery magnitude (`iterative_prune.py`)
A ~12%-pruned YOLOv8n fine-tuned 45 epochs (3x15) on 400 bird imgs recovers to
**mAP 0.388 = 91% of native 0.428** (up from 0.00). Cut-and-regrow strongly
validated. KNOWN BUG: the iterative loop only pruned on step 1 — re-pruning a
reloaded (YOLO(best)) model no-ops (params flat at 2.775M across steps), so the
"20% in 3 steps" is really ~12% in 1 step. Fix before building auto-prune mode;
does not block the UI (single-prune + masking).

## Reverse-training cockpit (web UI) — BUILT (`server.py`, `static/index.html`, `harness/mask.py`)
FastAPI + node-grid front end, verified end-to-end in-browser:
- 17 real YOLOv8n stages as columns, 20 nodes = 5% each, FLOP-weighted glow.
- Click nodes -> **mask** channels (zero lowest-L1; live, reversible, no crash).
- **Measure** -> REAL bird mAP@50 on a 40-img subset in ~1.2s (real gradient:
  robust vs sensitive stages).
- **Export** -> REAL per-stage structured removal (`structured_prune_per_stage`,
  torch-pruning `pruning_ratio_dict`), measured params/FLOPs/latency, saved to
  `runs/pigeon_pruned.pt` (~2s). Honors WHERE you cut.
- **Export + Recover** (`recover_job.py`, subprocess) -> prune + fine-tune the
  pruned model, streaming per-epoch mAP to a JSON the UI polls. Verified: a light
  L9 cut recovered **0.00 -> 0.4395 (val mAP@50) in ~2 epochs**, saved to
  `runs/recover/weights/best.pt`. The whole cut-and-regrow loop, in one button.
  (Recover mAP is on the 125-img training-val set; the cockpit's live mAP is a
  40-img fast subset -- both real, different sample sizes.)
- **Sensitivity scan** (`GET /api/sensitivity`, ~3.5s): probes every stage
  (cut 25%, measure mAP drop on a 16-img subset), colors the grid green→red and
  ranks by **value = FLOP / sensitivity**. Teaches cut-by-value: e.g. L16 (4%
  FLOPs, sens 1) is nearly free to cut while L4 (5% FLOPs, sens 100) is
  load-bearing. Foundation of the "k-cuts" method (probe → cut-by-value →
  fine-tune → re-probe).
- **Educational redesign**: teaching banner, encodings legend, per-stage verdict
  badges (✂ safe / ⚠ careful / 🛑 costly), a live **Coach** that critiques your
  cuts, an **efficiency-frontier** chart (compute-removed vs. mAP per Measure),
  and a "Principles of reverse training" lesson.
- **Technique-split, explanation-first redesign**: a persistent scoreboard
  (mAP / size / compute vs native) over three modules — **Precision** (quantize,
  `/api/quantize`), **Structure** (prune, scan+grid), **Regrow** (fine-tune).
  Each leads with a plain-English explanation + a one-click **suggested** action
  (Apply 8-bit / Cut top-value stages / Recover 12 epochs) and hides manual
  controls behind a "details" expander. Useful first, informative second.
- Responsive (stacks on narrow viewports). Run: `uvicorn server:app --port 8000`.
Deps added: fastapi, uvicorn.

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
