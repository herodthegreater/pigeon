## The idea

Depth costs latency. Every layer has to run after the one before it, so removing a whole layer saves time on any hardware. Whether a layer can go depends on the architecture. In a *plain* network each layer transforms its input, `h = relu(h W + b)`, and the next layer is tuned to that exact transform, so removing one feeds the next layer something it has never seen. In a *residual* network each layer adds a correction to a running *stream*, `h = h + relu(h W + b)`. Deleting a residual layer just drops one correction, and the stream it leaves behind is still in the format the next layer expects.

## How it works

1. Start from a deep trained network. Here that is 8 hidden layers of 128, either plain or residual.
2. List the candidate layers: hidden-to-hidden layers whose input and output widths match, so the layers around them still connect.
3. Choose `k` layers greedily. At each step, try deleting each remaining candidate (on top of those already chosen), measure accuracy on a probe set, and keep the deletion that hurts least.
4. Delete the chosen layers.
5. Optionally recover by distilling from the original deep net.

## The code, line by line

`_greedy_drop` in `lab/experiments/shrink.py` builds `cand = [i for i in range(1, m.n_layers - 1) if m.sizes[i] == m.sizes[i + 1]]`. That excludes layer 0 (12 → 128) and the output layer (128 → 20), leaving layers 1-7. For each of the `k` picks it loops over unused candidates, builds `trial = drop_layers(m, dropped + [c])` from the *original* net with all deletions so far plus `c`, and scores `(trial.logits(X).argmax(1) == y).mean()` on the 2,000 probe examples. The best `c` is appended to `dropped`. Indices always refer to the original numbering, because `drop_layers` in `lab/compress.py` deletes in descending order (`sorted(layers, reverse=True)`), so earlier deletions never shift later indices. The search costs about `k` times the number of candidates forward passes, which is cheap here and expensive on a large model.

`MLP.drop_layer` in `lab/nn.py` asserts `0 < layer < self.n_layers - 1 and self.sizes[layer] == self.sizes[layer + 1]`. It removes that layer's entry from every per-layer list (`W`, `b`, `res`, `masks`, `lin`) and `del self.sizes[layer + 1]` removes its output width. Nothing else changes. The layer after the deleted one now reads the deleted layer's input, which works only because the widths match.

What makes residual nets different lives in `MLP.forward`: `h = h + a if self.res[i] else a`. With `res=True` each layer's output is added to its input, so removing it leaves `h` exactly as it was. In `MLP.__init__`, residual layers start with `std *= 0.5` so the summed stream doesn't blow up over 7 additions.

`run_layer_drop` in `lab/methods.py` loads the cached 8x128 teacher of the chosen type and calls `_greedy_drop`. If recovery is on, it distills the cut net from the uncut one for `FT` = 500 steps.

## What the simulations show

From `layer_drop` (accuracy on 20 classes; KD rows are mean of 2 seeds, 1,500 recovery steps):

| layers deleted | plain, none | plain + KD | residual, none | residual + KD |
|---|---|---|---|---|
| 0 | 0.9097 | | 0.9113 | |
| 1 | 0.0622 | 0.9124 | 0.8666 | 0.9121 |
| 2 | 0.0660 | 0.9119 | 0.8044 | 0.9117 |
| 4 | 0.0534 | 0.9124 | 0.5320 | 0.9123 |
| 6 | 0.0568 | 0.8684 | 0.2580 | 0.8773 |

Deleting even one plain layer drops accuracy to chance (5% for 20 classes). Residual nets lose accuracy gradually. With recovery, the difference disappears: both architectures lose nothing with up to 4 of 8 layers gone. The teacher was deeper than this task needs. The same experiment also tried a DistilBERT-style init, building a 4-layer residual student from every other teacher layer. It scored 0.9152, against 0.9168 for a random init, so the inherited layers gave no head start.

Page runs at the defaults (residual, 2 deleted, KD): 0.8044 before recovery, 0.9131 after. Plain with 2 deleted goes from 0.066 to 0.8921 with the 500-step recovery. Residual with 6 deleted goes from 0.258 to 0.7808.

The toy can't measure latency, and its layers are cheap and identical. In real models, layers differ in cost and in what they do.

## When it breaks

- **Plain networks.** Without residual connections, deletion without recovery is fatal. The greedy search cannot find a safe layer because none exists.
- **Shape changes.** Only layers with equal input and output widths can be removed this way. Downsampling layers, and layers that change channel count, need a replacement projection.
- **Greedy is short-sighted.** The best single deletion is not always part of the best pair. Each choice is also scored on training data, so it can overfit the probe set.
- **The probe metric.** Choosing by accuracy on 2,000 examples ignores which classes suffer. A layer that matters only for the target class could be dropped.
- **Deep cuts.** At 6 of 8 layers, recovery no longer closes the gap. Depth was doing real work by then.

## Try this

- Set **architecture** to plain and **layers to delete** to 1 with **recovery** none. Then switch to residual. Compare the "layers deleted" rows.
- With residual and recovery none, step **layers to delete** from 1 to 6. Watch the gradual decline that the plain net never shows.
- Set **recovery** to kd and find the largest **layers to delete** that still recovers to within 0.01 of the uncut net.

## Going further

In language models, the ShortGPT page in this lab scores transformer blocks by how little they change the residual stream, the same idea with a cheaper score than trial deletion. LayerDrop (Fan et al.) goes further and trains with random layer deletion so the model is robust to it later. pigeon does not delete layers. Its cockpit cuts channels within stages (`harness/mask.py`, `harness/prune.py`). In YOLOv8, many backbone blocks contain residual bottlenecks, so block deletion is a lever pigeon could try. It would need the same per-stage sensitivity measurement the cockpit already does, followed by fine-tuning.
