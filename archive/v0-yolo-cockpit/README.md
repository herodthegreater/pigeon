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

All numbers are bird `mAP@50` on a fixed COCO bird set, measured on an M4.

| Operation | Effect on the model | Effect on accuracy |
|---|---|---|
| **Quantize** (fewer bits/weight) | smaller on disk | **graceful** — ~free at 8-bit, still alive at 6-bit with per-group scales, collapses at 4-bit |
| **Structural prune** (remove channels) | smaller **and** faster | **a cliff** — even a 5% one-shot cut drops mAP to ~0 |
| **Prune + fine-tune** (cut then regrow) | smaller and faster | **recovers** — a pruned model climbs from 0.00 back toward native |

Takeaways that shaped the tool:

- **Quantization degrades gracefully**, so there's free precision headroom.
- **Structural removal is all-or-nothing without regrowth** — removing whole
  channels severs feature pathways the rest of the network depends on. So the
  auto-shrink loop must be *prune → fine-tune → test*, never *prune → test*.
- **Fine-tuning is not optional; it's the other half of the cut.** A light cut
  recovered `0.00 → 0.44` (val mAP@50) in ~2 epochs; a ~12% cut recovered to
  ~91% of native.
- On a Mac, **pruning cuts FLOPs but often not wall-clock latency** (small models
  are overhead-bound; odd channel counts don't tile well). The portable win is
  the **FLOP/param** reduction, which pays off on edge hardware — so the UI
  reports both, and never leans on Mac latency alone.

Full running log of findings and gotchas lives in [`STATUS.md`](STATUS.md).

---

## The reverse-training cockpit (the flagship)

A local web app (`server.py` + `static/index.html`) that makes the whole loop
tactile.

```
┌ node grid ────────────────────────────────┐  ┌ panel ───────────────┐
│ L0  L1  L2  L3 … L22   (real YOLOv8n stages)│  │ bird mAP@50   0.3457 │
│ ▓▓  ▓▓  ▓▓  ▓▓        each column = 20 nodes │  │ est. FLOPs    4.4 G  │
│ ▓▓  ▓▓  ▓▓  ▓▓        1 node = 5% of channels│  │ ▶ Measure mAP        │
│ ░░  ▓▓  ░░  ▓▓        brighter = more FLOPs  │  │ ⤓ Export real model  │
│ ░░  ▓▓  ░░  ▓▓        red = masked/removed   │  │ 🌱 Export + Recover  │
└────────────────────────────────────────────┘  └──────────────────────┘
```

Each **column is a real stage** of the network; each column is **20 nodes = 5%**
of that stage's channels; columns **glow in proportion to the compute (FLOPs)**
they burn, so you attack the expensive stages first. The four actions:

- **Click nodes** → *mask* those channels (zero the lowest-importance ones).
  Instant, reversible, never crashes. Estimated compute-saved updates live.
- **▶ Measure mAP** → runs the **real** detector on bird images and reports the
  **real** `mAP@50` (~1.2 s). This is measured, not estimated — you see exactly
  which stages the task is sensitive to.
- **⤓ Export real model** → performs **real, dependency-aware channel removal**
  honoring your per-column cuts, measures the genuinely-smaller model
  (params / FLOPs / latency), and saves it to `runs/pigeon_pruned.pt`.
- **🌱 Export + Recover** → prunes **and then fine-tunes** the pruned model in a
  background job, streaming per-epoch `mAP` back to the panel so you watch
  accuracy **climb off the cliff**, then saves the recovered model.

### Why two layers (masking vs. real removal)

The cockpit deliberately separates a **cheap, live editing layer** from an
**expensive, real commit** — the same "edit-cheaply, bake-for-real" split used
for quantization:

- **Masking** (zeroing channels) leaves tensor shapes unchanged, so it's always
  safe and instantly reversible, and its accuracy is **numerically identical** to
  really removing those channels. Perfect for interactive exploration. It does
  *not* shrink the file or change speed.
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
| `model.py` | Applies the dial across a whole model: `snapshot`/`restore` (sweep bit-widths without reloading), `apply_uniform`. Quantizable modules are `Conv2d`, `Linear`, `Embedding` — the vision retarget was literally adding `Conv2d`. |
| `mask.py` | **The live UI layer.** `stage_table` runs one forward with hooks to get each stage's channel count and FLOP share (for the grid). `mask_stage` zeros the lowest-L1 output channels of a stage; `apply_masks` applies a whole `{stage: fraction}` config. `estimated_flop_saving` gives the conservative compute estimate shown while you click. |
| `prune.py` | **Real structural removal.** Ports `torch-pruning`'s official YOLOv8 recipe: swaps Ultralytics' `C2f` for a traceable `C2f_v2` (splitting the `chunk()` the tracer can't follow), then prunes in *train* mode with the `Detect` head ignored. `structured_prune` (global ratio) and `structured_prune_per_stage` (honors the UI's per-column fractions via `pruning_ratio_dict`). |
| `evaluate.py` | **The accuracy backbone.** `BirdmAPEvaluator` computes single-class `AP@0.50` (VOC all-points interpolation) by treating every ground-truth box as "bird" and asking the detector for COCO class 14 — sidesteps label-space mismatches and is fast enough for the live loop (`max_images` subsets it). Defined behind an `Evaluator` interface so a different task/metric drops in. |

The harness began life as a **language-model** fake-quantization tool;
`quant.py` is byte-for-byte the same. Fake-quantization doesn't care whether a
weight belongs to an LLM or a conv filter — which is why the same instrument
reverse-trains both (see *Provenance* below).

### The cockpit

| File | Responsibility |
|---|---|
| `server.py` | FastAPI backend. Loads + fuses YOLOv8n once, snapshots native weights, and serves: `GET /api/structure` (grid + native stats), `POST /api/apply` (mask → real mAP + estimated compute), `POST /api/export` (real per-stage removal → measured + saved model), `POST /api/recover` + `GET /api/recover_status` (start/poll a fine-tune job), `POST /api/reset`. All paths are absolute so it runs from any cwd. |
| `static/index.html` | The self-contained node-grid UI (vanilla JS, dark "terminal" theme, responsive). Renders the FLOP-weighted grid, handles clicks → masks, and drives Measure / Export / Recover with live polling. |
| `recover_job.py` | The recovery **subprocess** (run detached to avoid MPS/threading/signal issues). Prunes per the masks, fine-tunes the pruned model via Ultralytics' `DetectionTrainer`, and streams per-epoch `mAP` to a JSON file the server polls. |

### Command-line experiments (each reproduces one finding)

| Script | What it does |
|---|---|
| `benchmark.py` | Baseline yardstick: params, on-disk size, latency/FPS on this machine. |
| `carve.py` | Quantization confession curve — bird `mAP@50` vs. bit-width (16→1), with `--group-size` and `--skip`. Shows the graceful degradation + the per-tensor-vs-group-scale cliff. |
| `prune_sweep.py` | Structural confession curve — really removes channels at several ratios and shows the cliff (5% → 0). |
| `real_prune_test.py` | Proves real structured removal: params/FLOPs/latency before vs. after a 30% cut. |
| `finetune.py` | Prune 20% then fine-tune → recovery (the loop, headless). |
| `iterative_prune.py` | Iterative cut→regrow→cut experiment. |
| `pigeon.py` | The actual product use: detect/track birds (COCO class 14) on an image, video, or webcam via YOLO's built-in tracker. |
| `build_eval_set.py` | (Re)builds the 125-image COCO bird **eval** set (FiftyOne). |
| `build_finetune_set.py` | (Re)builds the 400-image COCO bird **train** set for recovery, in the model's 80-class space (bird = 14). |

---

## Results (YOLOv8n, M4)

| | params | FLOPs | bird mAP@50 |
|---|---|---|---|
| native | 3.157 M | 4.40 G | 0.4284 |
| pruned ~12%, **no** fine-tune | 2.78 M | ~4.0 G | 0.000 (cliff) |
| pruned ~12%, **fine-tuned** | 2.78 M | ~4.0 G | ~0.39–0.44 (recovered) |

(mAP is on the fixed COCO bird set; the cockpit's live readout uses a 40-image
fast subset, the recovery job uses the 125-image val set — both real, different
sample sizes.)

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
- **Iterative (multi-step) re-pruning has a known bug** — re-pruning a reloaded
  model no-ops, so the "shrink in many small steps" auto-mode isn't wired yet.
- **On a Mac, structural pruning may not reduce latency** even as FLOPs drop;
  trust the FLOP/param numbers for the portable story.
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
