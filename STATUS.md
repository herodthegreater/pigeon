# pigeon — status snapshot

## Baselines (this M4, MPS)
- **yolo11n**: 2.62 M params · 5.61 MB · 113 FPS · bird mAP@50 = 0.4501
- **yolov8n**: 3.16 M params · 4.40 GMAC · ~86 FPS · bird mAP@50 = 0.4284
  (0.4288 after the C2f→C2f_v2 rewrite `make_prunable` does: the same math
  split into two convs, so float rounding differs; the cockpit and every pruned
  model use the C2f_v2 form, so 0.4288 is their baseline)
- Eval set: `eval_set/` = 125 COCO val bird images + 440 boxes (YOLO format)

## What works end-to-end
- **Fake-quant harness** on the real detector (`harness/quant.py`, `model.py`)
- **Accuracy backbone**: `harness/evaluate.py` `BirdmAPEvaluator` (single-class AP@50)
- **Quant confession curve** (`carve.py`, DFL spared). These numbers were first
  measured on **YOLO11n** (carve.py loaded yolo11n.pt); re-measured on YOLOv8n:

  | bits | v8n per-tensor | v8n group 128 | 11n per-tensor | 11n group 128 |
  |---|---|---|---|---|
  | 16 | 0.4284 | 0.4284 | 0.4501 | 0.4501 |
  | 8 | 0.4088 | 0.4263 | 0.4403 | 0.4365 |
  | 6 | 0.3031 | 0.3905 | 0.0136 | 0.4095 |
  | 4 | 0.0000 | 0.0379 | 0.0000 | 0.0112 |

  v8n: 8-bit ~free with group-128 scales (−5% per-tensor); 6-bit −9% (g128) /
  −29% (per-tensor, *not* a collapse as on 11n); 4-bit collapses. Uniform-quant
  floor ~6-bit on both.
  - **DFL check**: the head's DFL conv holds fixed bin positions 0..15 that
    decode box edges; quantizing them was a suspect for the 4-bit collapse.
    Sparing it: v8n 4-bit g128 0.0191 → 0.0379, per-tensor 0.0000 → 0.0000;
    8-bit/6-bit change ≤0.004. Not the cause. Both `carve.py` and
    `/api/quantize` now spare it (it is 16 constants, not a learned weight).
- **REAL structured channel removal** (`harness/prune.py`, `real_prune_test.py`) —
  now works. Key fix: swap Ultralytics `C2f` -> `C2f_v2` (splits the `chunk()`
  the tracer can't follow), model in **train()** for tracing, ignore `Detect`,
  `GroupNormPruner` + `GroupMagnitudeImportance`, and `model.requires_grad_(True)`
  before building the pruner (see the iterative bug below). **Base = YOLOv8n**
  (YOLO11's C3k2/C2PSA have no equivalent recipe).
  - 30% prune: params 3.16M->1.81M (-43%), MACs 4.40G->2.78G (-37%), forward OK.
    (Before the requires_grad fix: 1.82M / 2.86G, because stages 0-1 were never
    cut.)

## Two hard, load-bearing findings
1. **Mac latency != FLOPs.** 30% prune cut MACs 35% but latency got *worse*
   (+6%) on MPS -- overhead-bound + non-multiple-of-8 channels. Show FLOPs, not
   Mac latency; speed win is for the edge target.
2. **Structural pruning is a CLIFF** (`prune_sweep.py`): even **5% one-shot prune
   -> mAP 0.000** (re-checked after the requires_grad fix: still 0.000 at 5, 10,
   20, 30%). Unlike quant, there's no graceful structural headroom.
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

## Recovery magnitude (`iterative_prune.py`) — re-pruning bug FIXED
Old result: one 7.2% cut (2.775M params, 12% fewer) fine-tuned 45 epochs
recovered to **0.388 = 91% of native**. That was the only cut that happened:
re-pruning a reloaded (YOLO(best)) model no-op'd, params flat at 2.775M.

Cause: torch-pruning traces the dependency graph through autograd, and
Ultralytics checkpoints load with `requires_grad=False`. A reloaded checkpoint
has nothing that requires grad, so the trace is empty and nothing is pruned. A
fresh model only partly works: `make_prunable` builds new C2f_v2 convs (which do
require grad), so the trace starts at the first C2f and **stages 0-1 were never
pruned**, by Export/Recover either. Fix: `model.requires_grad_(True)` before
building the pruner (as torch-pruning's YOLOv8 example does), in
`harness/prune.py`. Checked: reloaded best.pt at 7.2% went 3.111M -> 3.111M
before, -> 2.735M after; fresh model stage 0/1 channels 16/32 -> 16/32 before,
-> 14/29 after.

Re-run, now genuinely iterative (3 x [7.2% cut + 15 epochs], 400 imgs):

| step | params | MACs | mAP@50 (125 imgs) |
|---|---|---|---|
| native | 3.157M | 4.402G | 0.4284 |
| 1 | 2.774M | 3.911G | 0.3943 |
| 2 | 2.441M | 3.469G | 0.3377 |
| 3 (~20% of channels) | 2.174M | 3.155G | 0.2874 |

~20% in three regrown steps keeps 67% of native mAP with 31% fewer params and
28% fewer MACs, vs 0.0706 for one 20% cut + 40 epochs. Small cuts with
regrowth between them win by a wide margin. Each step loses more, so more data
and epochs per step are the next lever.

## Reverse-training cockpit (web UI) — BUILT (`server.py`, `static/index.html`, `harness/mask.py`)
FastAPI + node-grid front end, verified end-to-end in-browser:
- 16 real YOLOv8n stages as columns (L0-L21), 20 nodes = 5% each, glow weighted
  by share of MACs. The head (`model.22` on v8n) is not a column. It used to be:
  the code excluded `model.23`, YOLO11n's head index, so on v8n the head showed
  up as a maskable column that Export then ignored.
- **Stage compute share** is counted at each conv's *output* resolution (it used
  the input's, overcounting stride-2 convs 4x) and as a share of the whole
  model including the head (~41% of v8n's conv MACs). Before → after for some
  columns: L1 7.5% → 2.7%, L3/L5/L7 7.5% → 2.7%, L4/L6 5.0% → 7.2%,
  L12/L15 3.8% → 5.4%; the hook total (4.372 G) now matches torch-pruning's
  4.402 G count up to the non-conv ops.
- Click nodes -> **mask** channels. Live, reversible, no crash, and its mAP
  equals the exported model's (before fine-tuning); see "Masking = Export" below.
- **Measure** -> REAL bird mAP@50 on a 40-img subset in ~1s, plus the exact
  MACs/params of the model Export would produce (prunes a CPU copy, ~0.3s). It
  used to show a per-stage FLOP-share estimate, which missed the channels
  removed downstream of a cut: L15+L21 at 25% estimated 2% saved, Export
  really saves 8%.
- **Export** -> REAL per-stage structured removal (`structured_prune_per_stage`,
  torch-pruning `pruning_ratio_dict`), measured params/MACs/latency, saved to
  `runs/pigeon_pruned.pt` (~2s). Honors WHERE you cut (L0/L1 cuts were silently
  ignored before the requires_grad fix). Compute is reported as **GMACs**; the
  UI used to label torch-pruning's MAC count "GFLOP" (1 MAC = 2 FLOPs).
- **Export + Recover** (`recover_job.py`, subprocess) -> prune + fine-tune the
  pruned model, streaming per-epoch mAP to a JSON the UI polls. The curve now
  starts at the pruned model's *measured* mAP (it was a hard-coded 0.0, so the
  earlier "a light L9 cut recovered 0.00 -> 0.4395" had an unmeasured start).
  Verified: L9 cut 30% -> **0.1256 after the cut -> 0.3261 -> 0.3949** (val
  mAP@50, 2 epochs), saved to `runs/recover/weights/best.pt`. Also fixed: the
  final re-validation of best.pt was appended as an extra epoch ("epoch 13 of
  12"); it is now reported as `best_map`. (Recover mAP is Ultralytics'
  mAP50 on the 125-img val split, which is the *same 125 images* as
  `eval_set/`, and best.pt is picked on it, so recovered numbers are
  optimistic. The cockpit's live mAP is a 40-img subset of those images.)
- **Sensitivity scan** (`GET /api/sensitivity`, ~7s): probes every stage
  (cut 25%, measure mAP drop on a 16-img subset), colors the grid green→red and
  ranks by **value = compute share / sensitivity**. Teaches cut-by-value: e.g.
  L7 (2.7% of MACs, sens 0.00) and L16 (1.4%, sens 0.00) are nearly free to cut
  while L4 (7.2%, sens 1.00) is load-bearing. (Slower than the old ~3.5s
  because each probe now plans the real removal.) Foundation of the "k-cuts" method (probe → cut-by-value →
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

## Masking = Export (fixed; was not true before)
The README claimed masking was "numerically identical" to real removal. It
wasn't, for two reasons: (1) the server fuses BN into the conv bias before
masking and masking zeroed weights only, so a masked channel still output
SiLU(bias); (2) masking picked the lowest-L1 filters of each conv on its own,
while Export removes torch-pruning dependency *groups* (e.g. both sides of a
residual add, chosen by group L2). Now masking asks the same pruner which
channels it would remove (`prune.removal_plan`, interactive step, nothing
removed) and zeroes those channels' weight *and* bias in the fused model
(`mask.apply_plan`). bird mAP@50, 125 imgs, no fine-tune:

| cut | old mask | old mask + bias | Export (old) | Export (fixed) | new mask |
|---|---|---|---|---|---|
| L9 30% | 0.1298 | 0.3466 | 0.1265 | 0.1265 | 0.1265 |
| L4 25% | 0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| L0 25% | 0.3477 | 0.3897 | 0.4288 (L0 ignored) | 0.4100 | 0.4100 |
| L15+L21 25% | 0.1811 | 0.0252 | 0.0088 | 0.0088 | 0.0088 |
| all 16 stages 10% | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Raw detector outputs of the masked and exported models agree to ≤3e-3 (fused
vs unfused rounding). Planning costs ~0.2-0.5s per cut on CPU.

## Quantize size estimate
`/api/quantize` estimated size as params × bits / 8. It now counts one fp16
scale per group (or per tensor), and keeps biases and the DFL at 16-bit: group
128 at 8 / 6 / 4 bits = 3.21 / 2.42 / 1.63 MB (was 3.15 / 2.36 / 1.58), of a
6.30 MB fp16 fused model.

## Design consequences (locked)
- Auto-prune "training mode" = prune -> fine-tune -> test, repeat.
- Interactive demo value = "how small can I shrink while regrowing mAP back",
  not "watch accuracy fall".
- Per-STAGE cuts (what the UI does) can be far gentler than the UNIFORM prune
  swept here: uniform 5% -> 0.000, but L0 at 25% -> 0.4100 and L9 at 30% ->
  0.1265, unfine-tuned (125 imgs).

## Next steps
1. **Stronger recovery** (loop is proven; now push the number up): more train
   data (2k+ bird imgs), more epochs, and/or iterative small prunes with
   fine-tuning between each cut — target mAP back near native 0.43.
2. Snap channel counts to multiples of 8.

(Per-stage control, compute-weighted columns and the web cockpit are done. On
v8n the head is ~41% of conv MACs, then L4 and L6 at 7.2% each.)

## Env
Shared venv: `../weight-harness/.venv` (torch, transformers, ultralytics,
fiftyone, torch-pruning). COCO cache `~/fiftyone`. Base weights: yolo11n.pt,
yolov8n.pt in project root.
