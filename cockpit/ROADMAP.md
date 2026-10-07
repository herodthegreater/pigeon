# pigeon — roadmap

## Now: the pigeon reverse-training cockpit
Functional over the real `yolov8n` (see STATUS for what's verified):
- [x] Node-grid UI, 20 nodes/column = 5% each, FLOP-weighted columns.
- [x] Live-mask preview (real accuracy, live-channel compute) → **Export** does
  real dependency-aware removal + re-benchmark + real pre-regrow mAP.
- [x] Real fine-tuning (bonsai regrow) — and QAT when the recipe quantizes.
- [x] One stackable recipe (masks + quant) behind every measurement.
- [x] Precision lab: granularity, INT/FP levels, clipping, mixed precision
  (fragility scan + auto-mix), microscope, confession curves.
- [x] Detection preview ("through the model's eyes"), frontier, missions.
- [ ] Auto-shrink "training mode": scan → cut → regrow → repeat until the metric
  drops past a threshold (blocked on the iterative re-prune bug).
- [ ] Snap-to-8 channels on Export; Undo; real packed low-bit export.

## Phase 2 (PARKED, but keep the door open): LLM benchmark track
Take a large-ish LLM, reverse-train it with the *same harness* (`weight-harness/` is
already the fake-quant instrument), and score it on a real benchmark harness —
**`lm-evaluation-harness`** (MMLU / GSM8K / ARC), the tool behind the HF Open
LLM Leaderboard. The on-thesis experiment: prune *toward one subject* and chart
per-subject capability retention vs. compression.

Constraints noted so we design for them:
- 24 GB caps practical size at ~7–8B fp16 / ~13B 4-bit. Truly huge = rent cloud GPU.
- Benchmarks are slow on MPS → sweep on a subset, full run only for headline.
- Meaningful axis is *same model vs compression*, not vs GPT-4/Claude.

## The constraint that makes the transition simple
The cockpit must talk to the model through two thin seams so Phase 2 is a
plug-in, not a rewrite:

```
ModelAdapter          Evaluator
  .stages()             .score(model) -> float        # higher = better
  .prune(group, frac)   .name / .unit                 # "bird mAP@50" | "MMLU acc"
  .benchmark()          # latency, FLOPs, params
  .snapshot/.restore
```

- Vision now:  `YoloAdapter`  + `BirdRetentionEvaluator`
- LLM later:   `HFCausalAdapter` + `LMEvalHarnessEvaluator`

The UI, the auto-prune loop, and the accuracy readout depend only on these
interfaces — never on YOLO or on birds directly.
