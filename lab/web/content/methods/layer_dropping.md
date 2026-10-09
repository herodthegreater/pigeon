## The idea

Layers run one after another, so removing a whole layer saves time on any hardware. Whether a layer can go depends on the architecture. In a *plain* network each layer transforms its input, `h = relu(h W + b)`, and the next layer is tuned to that transform, so removing one feeds it something it has never seen. In a *residual* network each layer adds a correction to a running *stream*, `h = h + relu(h W + b)`. Deleting a residual layer just drops one correction, and the stream it leaves behind is still in the format the next layer expects.

## How it works

1. Start from a deep trained network. Here that is 8 hidden layers of 128, either plain or residual.
2. List candidates: hidden-to-hidden layers whose input and output widths match.
3. Choose `k` layers greedily. At each step, try deleting each remaining candidate (on top of those already chosen), measure accuracy on a probe set, and keep the deletion that hurts least.
4. Delete the chosen layers.
5. Optionally recover by distilling from the original deep net.

## The code, line by line

`_greedy_drop` in `lab/experiments/shrink.py` builds `cand = [i for i in range(1, m.n_layers - 1) if m.sizes[i] == m.sizes[i + 1]]`. That excludes layer 0 (12 → 128) and the output layer (128 → 20), leaving layers 1-7. For each of the `k` picks it loops over unused candidates, builds `trial = drop_layers(m, dropped + [c])` from the *original* net with all deletions so far plus `c`, and scores `(trial.logits(X).argmax(1) == y).mean()` on the 2,000 probe examples. The best `c` is appended to `dropped`. Indices always refer to the original numbering: `drop_layers` in `lab/compress.py` deletes in descending order (`sorted(layers, reverse=True)`), so no deletion shifts a later index.

`MLP.drop_layer` in `lab/nn.py` asserts `0 < layer < self.n_layers - 1 and self.sizes[layer] == self.sizes[layer + 1]`. It removes that layer's entry from every per-layer list (`W`, `b`, `res`, `masks`, `lin`) and `del self.sizes[layer + 1]` removes its output width. The next layer now reads the deleted layer's input, which works only because the widths match.

What makes residual nets different lives in `MLP.forward`: `h = h + a if self.res[i] else a`. With `res=True` each layer's output is added to its input, so removing it leaves `h` exactly as it was. `MLP.__init__` halves residual layers' init (`std *= 0.5`) so the stream doesn't blow up.

`run_layer_drop` in `lab/methods.py` loads the cached 8x128 teacher of the chosen type and calls `_greedy_drop`. If recovery is on, it distills the cut net from the uncut one for `FT` = 500 steps.

## What the simulations show

From `layer_drop` (accuracy on 20 classes; KD rows are mean of 2 seeds, 1,500 recovery steps):

| layers deleted | plain, none | plain + KD | residual, none | residual + KD |
|---|---|---|---|---|
| 0 | 0.9097 | | 0.9113 | |
| 1 | 0.0622 | 0.9162 | 0.8666 | 0.9174 |
| 2 | 0.0660 | 0.9163 | 0.8044 | 0.9183 |
| 4 | 0.0534 | 0.9168 | 0.5320 | 0.9168 |
| 6 | 0.0568 | 0.9138 | 0.2580 | 0.9117 |

Deleting even one plain layer drops accuracy to chance (5% for 20 classes). Residual nets lose accuracy gradually. With recovery the difference disappears: both architectures end at or slightly above their 8-layer teacher, even with 6 of 8 hidden layers gone. The teacher was much deeper than this task needs, and recovery (distilling on 28,000 inputs) is also extra training, which is how a cut model can beat it. A DistilBERT-style student built from every other teacher layer scored 0.9162, against 0.9192 from a random init, so the inherited layers gave no head start.

Page runs (500 recovery steps): at the defaults (residual, 2 deleted, KD), 0.8051 before recovery and 0.9174 after. Plain with 2 deleted goes from 0.0722 to 0.9069. Residual with 6 deleted goes from 0.2549 to 0.8967.

The toy can't measure latency, and its layers are identical. That 2 hidden layers suffice here says nothing about a detector.

## When it breaks

- **Plain networks.** Deletion without recovery is fatal. The greedy search cannot find a safe layer because none exists.
- **Shape changes.** Only width-preserving layers can be removed this way. Downsampling layers need a replacement projection.
- **Greedy is short-sighted.** The best single deletion is not always part of the best pair, and scoring on training data can overfit the probe.
- **The probe metric.** Overall accuracy ignores which classes suffer. A layer that matters only for the target class could be dropped.
- **Recovery hides the damage only when depth was spare.** Here even 6 of 8 layers recover, because a 2-layer MLP fits this task. On a model that needs its depth, the no-recovery column is the honest warning.

## Try this

- Set **architecture** to plain and **layers to delete** to 1 with **recovery** none. Then switch to residual. Compare the "layers deleted" rows.
- With residual and recovery none, step **layers to delete** from 1 to 6. Watch the gradual decline that the plain net never shows.
- Set **recovery** to kd and **layers to delete** to 6. Compare plain and residual: with recovery, the architecture that survived deletion best no longer has an advantage.

## Going further

In language models, the ShortGPT page in this lab scores transformer blocks by how little they change the residual stream, a cheaper score than trial deletion. LayerDrop (Fan et al.) trains with random layer deletion so the model tolerates it later. The YOLOv8n cockpit does not delete layers. It cuts channels within stages (`harness/mask.py`, `harness/prune.py`). In YOLOv8, many backbone blocks contain residual bottlenecks, so block deletion is a lever the cockpit could try, guided by the cockpit's per-stage sensitivity scan and followed by fine-tuning.
