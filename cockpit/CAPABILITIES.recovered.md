<!-- Recovered 2026-09-23: CAPABILITIES.md (never committed) vanished from the repo root
     mid-session. This is the first 80 lines as read just before it disappeared;
     the remainder (from "## 5. Results") was not captured. -->

# pigeon — Capabilities

*A hands-on instrument for **reverse training**: take a capable, fully-trained
model and remove from it — dropping precision, zeroing channels, deleting whole
filters — until only the one capability you care about remains, watching what
every cut does to real accuracy in real time.*

Reference case: carve a general 80-class detector (**YOLOv8n**, COCO) down to a
**bird-only** detector. The pipeline is task-agnostic — swap the class or eval
set and it reverse-trains toward something else.

---

## 1. Three compression techniques, measured on the real model

| Technique | What it does | Empirical behavior |
|---|---|---|
| **Quantize** (`harness/quant.py`, `model.py`) | Rounds weights to what an *N*-bit encoding could store (fake-quant → numerically identical to a real quantize/dequantize). 1–16 bit, per-group scales. | **Graceful.** ~free at 8-bit; alive at 6-bit with group scales; collapses at 4-bit. |
| **Structural prune** (`harness/prune.py`) | Real, dependency-aware channel removal via torch-pruning's YOLOv8 recipe (C2f→C2f_v2, train-mode tracing, `Detect` head ignored). Global or per-stage. | **A cliff.** Even a 5% one-shot cut drops bird mAP@50 to ~0 — no structural slack. |
| **Regrow** (`finetune.py`, `recover_job.py`) | Fine-tunes the *pruned* model so surviving capacity re-specializes around the task. | **Recovers.** A ~12% cut fine-tuned back to **mAP 0.388 = 91% of native 0.428**. |

**Core finding the whole tool is built around:** *precision has slack; structure
does not.* Pruning must be `prune → fine-tune → test`, never `prune → test`.

## 2. The reverse-training cockpit (flagship)

A local FastAPI web app (`server.py` + `static/index.html`) that makes the loop
tactile:

- **FLOP-weighted node grid** — each column is a real YOLOv8n stage; 20 nodes =
  5% of that stage's channels; columns glow in proportion to the compute they
  burn, so you attack expensive stages first.
- **Live masking** — click nodes to zero the lowest-importance channels;
  instant, reversible, never crashes, and numerically identical in accuracy to
  really removing them. (Explore cheaply; commit for real separately.)
- **Measure mAP** — runs the real detector on bird images and reports the real
  `mAP@50` (~1.2 s on a 40-image subset). Measured, not estimated.
- **Export real model** — performs genuine per-stage structured removal honoring
  your cuts, re-benchmarks params/FLOPs/latency, saves to `runs/pigeon_pruned.pt`.
- **Export + Recover** — prunes **and** fine-tunes in a background subprocess,
  streaming per-epoch mAP back so you watch accuracy climb off the cliff.
- **Sensitivity scan** — probes every stage (cut 25%, measure the mAP drop),
  colors the grid green→red, and ranks stages by **value = FLOP ÷ sensitivity**.
  Teaches *cut-by-value*: some stages are nearly free to cut, others are
  load-bearing.
- **Explanation-first UX** — a persistent scoreboard (mAP / size / compute vs
  native) over three modules (Precision · Structure · Regrow), each leading with
  a plain-English explanation, a one-click suggested action, per-stage verdict
  badges (✂ safe / ⚠ careful / 🛑 costly), a live **Coach**, and an
  efficiency-frontier chart. Responsive.

## 3. Domain-agnostic by design (the retarget seam)

The cockpit talks to the model through two thin interfaces, so a new domain is a
plug-in, not a rewrite:

```
ModelAdapter                 Evaluator
  .stages()                    .score(model) -> float   # higher = better
  .prune(group, frac)          .name / .unit            # "bird mAP@50" | "MMLU acc"
  .benchmark()  # latency, FLOPs, params
  .snapshot / .restore
```

The harness **began as a language-model fake-quantization tool** — `quant.py` is
byte-for-byte the same; the vision retarget was literally adding `Conv2d`
support and swapping the evaluator from perplexity to bird `mAP@50`. Because
`fake_quantize` is weight-meaning-agnostic and the evaluator sits behind a small
interface, the same instrument could reverse-train an LLM toward a single
benchmark (parked Phase 2: score on `lm-evaluation-harness` — MMLU / GSM8K / ARC).

## 4. Command-line experiments (each reproduces one finding)

`benchmark.py` (baseline params/size/latency) · `carve.py` (quant confession
curve vs bit-width) · `prune_sweep.py` (structural cliff) · `real_prune_test.py`
(proves real removal: 30% cut → params −43%, MACs −35%) · `finetune.py` (prune →
fine-tune recovery) · `iterative_prune.py` (cut→regrow→cut) · `pigeon.py` (the
product: detect/track birds on image/video/webcam).

## 5. Results (YOLOv8n, M4)
