# pigeon — roadmap

## Now: the pigeon carving cockpit
Functional demo over the real `yolo11n`:
- Node-grid UI, 20 nodes/column = 5% each, FLOP-weighted columns, snap-to-8.
- Live-mask preview (instant accuracy) → **Apply** does real dependency-aware
  removal (`torch-pruning`) + re-benchmark (real latency/FLOPs) + **Undo**.
- Real fine-tuning (bonsai regrow) in settings.
- Auto-prune "training mode": shrink until the metric drops past a threshold.

## Phase 2 (PARKED, but keep the door open): LLM benchmark track
Take a large-ish LLM, carve it with the *same harness* (`weight-harness/` is
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
