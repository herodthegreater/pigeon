## The idea

To get a width-16 network you can train width 16, or train something much wider, which learns faster, then prune it to width 16 and fine-tune. Li et al. (2020) argued that for large transformers under a fixed compute budget, training large and then compressing beats training small. This page tests that on MLPs, with structured (whole-neuron) pruning and *self-KD*: recovery by distilling from the model's own unpruned version.

## How it works

1. Train a `2 x start` network from scratch on labels.
2. Score its neurons with the Taylor criterion, `sum |a * dL/da|` over examples (see Structured pruning).
3. Keep the top `target` neurons in each hidden layer.
4. Recover for half the original training budget, either on labels or by distilling from the unpruned `2 x start` net.
5. Baseline: train `2 x target` directly for the same total number of steps.

## The code, line by line

`train_big_then_compress` in `lab/experiments/small.py` loops over `target` in `[16, 32]` and `start = target * mult` for `mult` in `[1, 2, 4, 8, 16]`, 2 seeds each. `big = scratch([start, start], s)` trains the wide net for `STEPS` = 3,000 steps.

The `mult == 1` branch is the baseline. `m = scratch(big, s, steps=STEPS // 2)` continues training the target-width net for the same extra 1,500 steps the pruned nets get for recovery, so every arm sees 4,500 steps in total. There is no self-KD baseline: distilling a net into an exact copy of itself has zero gradient (`T * (ps - pt) / N` with `ps == pt`).

For `mult > 1`, `sc = neuron_importance(big, probe.X, probe.y, "taylor")` scores every hidden neuron on 2,000 training examples. `cut = structured_prune(big, [target, target], sc)` keeps the top `target` per layer. That keeps only `(target / start) ** 2` of the hidden-to-hidden weights. Two recoveries follow, each from a fresh `cut.copy()`: `scratch(..., steps=STEPS // 2)` on labels, and `distill(cut.copy(), big, s, steps=STEPS // 2)` with the wide net as teacher (KD at `distill`'s default T=1). Both use the same learning rate as training from scratch.

`run_big_then_prune` in `lab/methods.py` is the one-seed version for this page. It runs 1,000 steps for the wide net, 500 for self-KD recovery, and 1,500 for the directly trained baseline.

## What the simulations show

From `train_big_then_compress` (accuracy on 20 classes, mean of 2 seeds; every final model has the same size and the same 4,500 training steps):

| trained width | final 2x16, labels | final 2x16, self-KD | final 2x32, labels | final 2x32, self-KD |
|---|---|---|---|---|
| = target (direct) | 0.6863 | | 0.8314 | |
| 2x | 0.6425 | 0.6423 | 0.7996 | 0.8053 |
| 4x | 0.6185 | 0.6234 | 0.8004 | 0.8105 |
| 8x | 0.6124 | 0.6218 | 0.7955 | 0.8053 |
| 16x | 0.6088 | 0.6211 | 0.7886 | 0.7907 |

Every pruned arm is worse than training the small net directly: by at least 0.04 for 2x16 and 0.02 for 2x32. Most of the loss comes at the first doubling. Self-KD beats labels from 4x up, but never reaches the direct baseline. In `small_tricks`, the same recipe (2x128 → 2x16 or 2x32) with a full 3,000-step recovery landed close to plain training: 0.6568 against 0.6637 for 2x16, 0.8225 against 0.8170 for 2x32. Given enough recovery, the pruned start neither helps nor hurts much.

Page runs (1,000 + 500 steps, one seed): at the defaults (128 → 16) the 2x128 net reaches 0.9008, but pruned to 2x16 and recovered it scores 0.5043, against 0.6249 for 2x16 trained directly. Starting from 32 gives 0.5499, and from 256, 0.5073. A mild cut from 128 to 64 nearly ties: 0.8649 pruned against 0.8672 direct.

The toy does not test Li et al.'s regime: big, compute-limited models whose faster learning per step dominates, compressed lightly.

## When it breaks

- **Cuts too deep for recovery.** Removing 7 of 8 neurons per layer leaves a damaged network that trains worse than a random init with the same budget.
- **Self-KD cannot rescue a bad start.** Distilling from the unpruned wide net helps a little, but a much wider teacher has functions the small net cannot represent. See the Teacher assistants page for the capacity gap.
- **Unequal budgets.** Here every arm gets 4,500 steps, but the wide net's steps cost far more compute. Matching FLOPs instead of steps would favour training small even more.
- **In-place training.** `scratch` and `distill` train their argument in place, so `scratch(big, ...)` in the baseline branch extends `big` itself. Copy first if you need the original.

## Try this

- Keep **final width** 16 and step **trained width** through 32, 64, 128 and 256. Compare the pruned row with "trained directly" each time.
- Set **final width** to 64 and **trained width** to 128, a mild 2x cut. Compare how close the pruned net comes to the direct one with the 8x default.

## Going further

Li et al.'s result rests on two facts about large transformers: bigger models reach a given loss in fewer steps, and they are more robust to quantization and pruning, so mild compression costs little. The toy reproduces neither. pigeon is train-big-then-prune by construction: it starts from a pretrained YOLOv8n and fine-tunes on only 400 bird images, so most of what the small model knows must come from the big one. Its measured cuts were mild: a ~12% channel cut recovered to 91% of native mAP@50, and a 20% cut recovered only to 0.0706 after 40 epochs. The toy's lesson is about how deep a cut goes. On the Structured pruning page, a 2x cut per layer recovered to within 0.005 of the teacher and an 8x cut never came close.
