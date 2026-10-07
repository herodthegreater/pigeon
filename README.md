# pigeon

**Reverse-train a general object detector down until it does one thing — detect and
track birds — in the smallest, fastest form possible, and watch what every cut
does to its accuracy in real time.**

`pigeon` is a hands-on instrument for *"reverse training"*: instead of growing a
model by training it, you take a capable pretrained detector and **remove** from
it — dropping precision, zeroing channels, deleting whole filters — until only
the one capability you care about remains. It ships with a web **carving
cockpit** that turns this into a point-and-click loop: cut the model, measure the
real accuracy hit, remove the weights for real, and fine-tune the survivor back
to life.

Prototype/host target: Apple Silicon (tested on an M4, MPS). Final edge target is
deliberately left open — the tool reports the portable size/compute numbers that
predict the win on whatever chip you deploy to.

---

## What it aims to accomplish

A general detector (here **YOLOv8n**, trained on COCO's 80 classes) is a
generalist: most of its weights serve 79 classes you will never use. The goal of
`pigeon` is to answer, empirically and interactively:

- **How much of a general model does one task actually need?**
- **Which specific weights / channels / stages carry that task, and which are
  dead weight?**
- **How small and fast can the model get before the task breaks — and can
  fine-tuning bring it back?**

The concrete deliverable is a genuinely smaller, faster **bird-only** detector,
plus the instrument that produced it. "Bird" (COCO class 14) is used as the
single target — no custom dataset required — but the whole pipeline is
task-agnostic: swap the class or the eval set and it reverse-trains toward something
else.

## The core idea: reverse training

Training is additive — you nudge weights up from noise until a capability
emerges. `pigeon` runs the process backward:

1. **Start** from a capable, fully-trained model.
2. **Cut** — reduce precision (quantize) or remove capacity (prune).
3. **Measure** the real accuracy of the *one task* you're keeping.
4. **Regrow** — fine-tune the wounded model so the surviving capacity
   re-specializes around the task.
5. **Repeat**, shrinking until the task is about to break.

The two "cut" levers behave very differently, and the difference is the central
finding of the project (see below): **precision has slack; structure does not.**

## What the data showed

All numbers are bird `mAP@50` for **YOLOv8n** (the cockpit's model) on a fixed
COCO bird set, measured on an M4, unless a row says otherwise.

| Operation | Effect on the model | Effect on accuracy |
|---|---|---|
| **Quantize** (fewer bits/weight) | smaller on disk | **graceful** — ~free at 8-bit, −9% at 6-bit with per-group scales, collapses at 4-bit |
| **Structural prune** (remove channels) | smaller **and** faster | **a cliff** — even a 5% one-shot cut drops mAP to 0.000 |
| **Prune + fine-tune** (cut then regrow) | smaller and faster | **recovers** — a pruned model climbs back toward native |

The quantization curve (`carve.py`, 125 images, DFL kept at native precision):

| bits | YOLOv8n, per-tensor scale | YOLOv8n, group 128 | YOLO11n, per-tensor | YOLO11n, group 128 |
|---|---|---|---|---|
| 16 (native) | 0.4284 | 0.4284 | 0.4501 | 0.4501 |
| 8 | 0.4088 (−5%) | 0.4263 (−0.5%) | 0.4403 | 0.4365 |
| 6 | 0.3031 (−29%) | 0.3905 (−9%) | 0.0136 | 0.4095 |
| 4 | 0.0000 | 0.0379 | 0.0000 | 0.0112 |

The original "6-bit only survives with per-group scales" finding was measured on
YOLO11n; YOLOv8n is more forgiving at 6-bit with one scale per tensor (−29%
instead of a collapse), and needs per-group scales to make 8-bit truly free.
Sparing the head's DFL conv (whose fixed weights are the box-decoding bin
positions 0..15) does not rescue 4-bit, so the collapse is in the learned
weights.

Takeaways that shaped the tool:

- **Quantization degrades gracefully**, so there's free precision headroom.
- **Structural removal is all-or-nothing without regrowth** — removing whole
  channels severs feature pathways the rest of the network depends on. So the
  auto-shrink loop must be *prune → fine-tune → test*, never *prune → test*.
- **Fine-tuning is not optional; it's the other half of the cut.** A 30% cut
  of L9 went `0.126 → 0.395` (val mAP@50) in 2 epochs; three ~7% cuts with 15
  epochs of regrowth after each kept 0.287 at 31% fewer params, where one 20%
  cut + 40 epochs reached only 0.071.
- On a Mac, **pruning cuts compute but often not wall-clock latency** (small models
  are overhead-bound; odd channel counts don't tile well). The portable win is
  the **MAC/param** reduction, which pays off on edge hardware — so the UI
  reports both, and never leans on Mac latency alone.

Full running log of findings and gotchas lives in [`STATUS.md`](STATUS.md).

---

## The reverse-training cockpit (the flagship)

A local web app (`server.py` + `static/index.html`) that makes the whole loop
tactile.

```
┌ node grid ────────────────────────────────┐  ┌ panel ───────────────┐
│ L0  L1  L2  L3 … L21   (real YOLOv8n stages)│  │ bird mAP@50   0.3465 │
│ ▓▓  ▓▓  ▓▓  ▓▓        each column = 20 nodes │  │ compute    4.40 GMAC │
│ ▓▓  ▓▓  ▓▓  ▓▓        1 node = 5% of channels│  │ ▶ Measure mAP        │
│ ░░  ▓▓  ░░  ▓▓        brighter = more compute│  │ ⤓ Export real model  │
│ ░░  ▓▓  ░░  ▓▓        red = masked/removed   │  │ 🌱 Export + Recover  │
└────────────────────────────────────────────┘  └──────────────────────┘
```

Each **column is a real stage** of the network (the detection head, L22, is not
a column: Export never prunes it); each column is **20 nodes = 5%** of that
stage's channels; columns **glow in proportion to the compute (MACs)** they
burn, so you attack the expensive stages first. The head alone is ~41% of
YOLOv8n's conv compute, so the 16 columns cover the other ~59%. The four actions:

- **Click nodes** → *mask* those channels (zero exactly the channels Export
  would remove). Instant, reversible, never crashes. Compute saved is the exact
  MAC count of the model Export would produce.
- **▶ Measure mAP** → runs the **real** detector on bird images and reports the
  **real** `mAP@50` (~1.2 s). This is measured, not estimated — you see exactly
  which stages the task is sensitive to.
- **⤓ Export real model** → performs **real, dependency-aware channel removal**
  honoring your per-column cuts, measures the genuinely-smaller model
  (params / MACs / latency), and saves it to `runs/pigeon_pruned.pt`.
- **🌱 Export + Recover** → prunes **and then fine-tunes** the pruned model in a
  background job, streaming per-epoch `mAP` back to the panel so you watch
  accuracy **climb off the cliff**, then saves the recovered model.

### Why two layers (masking vs. real removal)

The cockpit deliberately separates a **cheap, live editing layer** from an
**expensive, real commit** — the same "edit-cheaply, bake-for-real" split used
for quantization:

- **Masking** (zeroing channels) leaves tensor shapes unchanged, so it's always
  safe and instantly reversible, and its accuracy is **the same** as really
  removing those channels: it asks `torch-pruning` which channels Export would
  remove (whole dependency groups, e.g. both sides of a residual add), then
  zeroes each one's weights *and* bias, so the channel outputs SiLU(0) = 0
  exactly as if it were gone. Masked and exported mAP match to 4 decimals on
  every cut tested (see `STATUS.md`). It does *not* shrink the file or change
  speed.
- **Export** actually deletes the channels (via `torch-pruning`), which is what
  makes the model smaller/faster — but is slower, structurally constrained, and
  needs fine-tuning after. You only pay that cost when you commit.

So you explore with masking (real accuracy feedback, zero risk) and commit with
Export/Recover (real artifact).

---

## Architecture & components

### `harness/` — the domain-agnostic reverse-training instrument

| File | Responsibility |
|---|---|
| `quant.py` | **The doubt-dial.** `fake_quantize(w, bits, group_size)` rounds a tensor to the values an *N*-bit encoding could store, then leaves it in float. A forward pass through fake-quantized weights is numerically identical to a real quantize-then-dequantize path, so accuracy measured here transfers exactly to a real export. Handles 1-bit (binary) through 16-bit; the *scale* is shared per group (a per-weight scale is a no-op). |
| `model.py` | Applies the dial across a whole model: `snapshot`/`restore` (weights and biases, so quantization and masking both undo without reloading), `apply_uniform` (with a `skip` list; the cockpit and `carve.py` spare the DFL's fixed bin positions). Quantizable modules are `Conv2d`, `Linear`, `Embedding` — the vision retarget was literally adding `Conv2d`. |
| `mask.py` | **The live UI layer.** `stage_table` runs one forward with hooks to get each stage's channel count and share of the model's MACs (counted at each conv's *output* resolution), excluding the head (derived from the model: `model.22` on YOLOv8n). `apply_plan` zeroes the weight and bias of the channels `prune.removal_plan` lists, which reproduces the exported model. |
| `prune.py` | **Real structural removal.** Ports `torch-pruning`'s official YOLOv8 recipe: swaps Ultralytics' `C2f` for a traceable `C2f_v2` (splitting the `chunk()` the tracer can't follow), sets `requires_grad` so the autograd trace sees every layer, then prunes in *train* mode with the `Detect` head ignored. `structured_prune` (same ratio for every group, local L2 ranking), `structured_prune_per_stage` (honors the UI's per-column fractions via `pruning_ratio_dict`) and `removal_plan` (which channels that would remove, without removing them). |
| `evaluate.py` | **The accuracy backbone.** `BirdmAPEvaluator` computes single-class `AP@0.50` (VOC all-points interpolation) by treating every ground-truth box as "bird" and asking the detector for COCO class 14 — sidesteps label-space mismatches and is fast enough for the live loop (`max_images` subsets it). Defined behind an `Evaluator` interface so a different task/metric drops in. |

The harness began life as a **language-model** fake-quantization tool;
`quant.py` is byte-for-byte the same. Fake-quantization doesn't care whether a
weight belongs to an LLM or a conv filter — which is why the same instrument
reverse-trains both (see *Provenance* below).

### The cockpit

| File | Responsibility |
|---|---|
| `server.py` | FastAPI backend. Loads YOLOv8n once in its prunable `C2f_v2` form, fuses it, snapshots native weights, and serves: `GET /api/structure` (grid + native stats), `GET /api/sensitivity` (probe every stage), `POST /api/apply` (mask → real mAP + the exact MACs/params Export would give), `POST /api/quantize` (fake-quantize → real mAP + estimated size, per-group scales included), `POST /api/export` (real per-stage removal → measured + saved model), `POST /api/recover` + `GET /api/recover_status` (start/poll a fine-tune job), `POST /api/reset`. Compute is reported in **GMACs** (1 MAC = 2 FLOPs). All paths are absolute so it runs from any cwd. |
| `static/index.html` | The self-contained node-grid UI (vanilla JS, dark "terminal" theme, responsive). Renders the compute-weighted grid, handles clicks → masks, and drives Measure / Export / Recover with live polling. |
| `recover_job.py` | The recovery **subprocess** (run detached to avoid MPS/threading/signal issues). Prunes per the masks, measures the pruned model's mAP (the first point of the curve), fine-tunes it via Ultralytics' `DetectionTrainer`, and streams per-epoch `mAP` to a JSON file the server polls. |

### Command-line experiments (each reproduces one finding)

| Script | What it does |
|---|---|
| `benchmark.py` | Baseline yardstick: params, on-disk size, latency/FPS on this machine. |
| `carve.py` | Quantization confession curve — bird `mAP@50` vs. bit-width (16→1), with `--weights` (YOLOv8n by default), `--group-size` and `--skip` (DFL spared by default). Shows the graceful degradation and what per-group scales buy. |
| `prune_sweep.py` | Structural confession curve — really removes channels at several ratios and shows the cliff (5% → 0). |
| `real_prune_test.py` | Proves real structured removal: params/MACs/latency before vs. after a 30% cut. |
| `finetune.py` | Prune 20% then fine-tune → recovery (the loop, headless). |
| `iterative_prune.py` | Iterative cut→regrow→cut experiment (three ~7% cuts, 15 epochs each). |
| `pigeon.py` | The actual product use: detect/track birds (COCO class 14) on an image, video, or webcam via YOLO's built-in tracker. |
| `build_eval_set.py` | (Re)builds the 125-image COCO bird **eval** set (FiftyOne). |
| `build_finetune_set.py` | (Re)builds the 400-image COCO bird **train** set for recovery, in the model's 80-class space (bird = 14). |

---

## Results (YOLOv8n, M4)

`iterative_prune.py`: cut 7.2% of every prunable group's channels, fine-tune 15
epochs on 400 bird images, reload, repeat.

| | params | MACs | bird mAP@50 |
|---|---|---|---|
| native | 3.157 M | 4.40 G | 0.4284 |
| 5% one-shot cut, **no** fine-tune (`prune_sweep.py`) | 2.88 M | 4.05 G | 0.000 (cliff) |
| step 1: 7% cut + 15 epochs | 2.774 M | 3.91 G | 0.3943 |
| step 2: another 7% + 15 epochs | 2.441 M | 3.47 G | 0.3377 |
| step 3: another 7% + 15 epochs (~20% of channels) | 2.174 M | 3.16 G | 0.2874 |
| for comparison: one-shot 20% + 40 epochs (`finetune.py`) | — | — | 0.0706 |

mAP is the 125-image eval set (`BirdmAPEvaluator`); the cockpit's live readout
uses a 40-image subset of it. The native model scores 0.4284; after the
C2f→C2f_v2 rewrite pruning needs it scores 0.4288 (same math, different float
rounding), which is the cockpit's baseline. Fine-tuning picks `best.pt` on a
val split made of the same 125 images, so recovered numbers lean optimistic.
The earlier "~12% pruned, fine-tuned ~0.39–0.44" row mixed two runs: 0.388
was this experiment when only step 1 actually pruned (see `STATUS.md`), and 0.4395
was a separate light L9 cut in the cockpit.

## The simulation lab (`lab/`)

Real YOLO experiments are slow and come one at a time. [`lab/`](lab/README.md)
is a CPU-only, numpy-only miniature of the same problem: a 20-class
"generalist" world with one target class and a known Bayes-optimal answer.
It runs **28 experiments** across every way we could find to make a small
model from a big one, or make a small model better: distillation (12 loss and
data variants, teacher edge cases, data-free), pruning (unstructured,
structured, lottery tickets), quantization (PTQ, QAT, k-means sharing),
low-rank, layer deletion, specialization to one class, and their combinations.
Findings, and what they imply for pigeon's cut-and-regrow loop, are in
[`lab/REPORT.md`](lab/REPORT.md).

```bash
python -m lab.web            # the website: every method, runnable, with its source and lessons
python -m lab.run            # re-run all simulations (~35 min on 4 cores)
python -m pytest lab/tests   # tests generated from the method registry
```

The website groups methods by model family (small networks, language models,
vision/YOLO), lets you run each one, shows the exact code, and has a
"build it yourself" track of 10 coding lessons with tests. It is also published
as a static site on GitHub Pages (`.github/workflows/pages.yml`), where the same
Python runs in the visitor's browser via Pyodide; see
[`lab/README.md`](lab/README.md#the-same-site-on-github-pages-no-server).

## Quickstart

```bash
git clone https://github.com/herodthegreater/pigeon && cd pigeon
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python build_eval_set.py        # fetch the 125-image bird eval set (COCO subset)
python build_finetune_set.py    # fetch the 400-image bird train set (for Recover)

# launch the reverse-training cockpit
uvicorn server:app --port 8000 --app-dir .   # open http://localhost:8000
```

Model weights (`yolov8n.pt`) auto-download on first run. Datasets and weights are
git-ignored; the two `build_*` scripts regenerate them.

## Honest limitations

- **Base model is YOLOv8n, not YOLO11n.** Real pruning relies on
  `torch-pruning`'s C2f→C2f_v2 recipe; YOLO11's `C3k2`/`C2PSA` blocks have no
  equivalent yet.
- **The "shrink in many small steps" auto-mode isn't wired into the cockpit
  yet.** The re-pruning bug that blocked it is fixed (`iterative_prune.py`
  works), but the UI still does one cut per Recover.
- **On a Mac, structural pruning may not reduce latency** even as MACs drop;
  trust the MAC/param numbers for the portable story.
- **The eval set doubles as the fine-tuning val split** (same 125 COCO images),
  and `best.pt` is chosen on it.
- "Bird" stands in for the single task; specializing to *pigeons specifically*
  would need a labeled pigeon dataset and head collapse to one class.

## Status & roadmap

Living status (findings, bugs, next steps) is in [`STATUS.md`](STATUS.md);
longer-range plans, including a parked **Phase 2** (point the same harness at a
large language model and score it on a benchmark harness), are in
[`ROADMAP.md`](ROADMAP.md).

## Provenance

The instrument started as a language-model quantization harness — `harness/`
was retargeted to vision by adding `Conv2d` support and swapping the eval from
perplexity to bird `mAP@50`. Because `fake_quantize` is weight-meaning-agnostic
and the evaluator sits behind a small interface, the same cockpit could, in
principle, reverse-train an LLM toward a single benchmark — the same idea,
applied to language. That's the parked Phase 2.
