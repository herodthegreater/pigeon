## The idea

Unstructured pruning leaves a matrix full of zeros that still has to be stored and multiplied. Structured pruning deletes whole *neurons*: one hidden unit means one column of the matrix that feeds it and one row of the matrix it feeds. The matrices get genuinely smaller, so the model gets smaller and faster on any hardware. Every neuron you remove takes all its connections with it, so the choice of *which* neurons to keep matters more than in unstructured pruning. That choice is made by an *importance score*. In a conv net the same idea removes channels, which is what pigeon does to YOLOv8n.

## How it works

1. Run the trained net on a probe batch (2,000 training examples here).
2. Give every hidden neuron a score, one of:
   - `l1`: the sum of `|w|` over the neuron's incoming weights.
   - `act`: the mean `|activation|` of the neuron on the probe data.
   - `taylor`: the sum over examples of `|a * dL/da|`, where `a` is the neuron's activation and `dL/da` is the gradient of the loss with respect to it. This is a first-order estimate of how much the loss would change if `a` were forced to 0.
   - `random`: a control.
3. In every hidden layer, keep the `k` highest-scoring neurons and delete the rest.
4. Recover by training the smaller net, with labels or by distilling from the original.

## The code, line by line

`neuron_importance` in `lab/compress.py` returns one score array per hidden layer. For `l1`, `np.abs(net.W[i]).sum(0)` sums each column of `W[i]`, and column `j` holds every weight going *into* neuron `j`. This criterion never looks at data. The other criteria run `z, hid, cache = net.forward(X, cache=True)`, where `hid` holds the post-ReLU output of each hidden layer. `act` is `np.abs(h).mean(0)`.

For `taylor`, `p = softmax(z)` and `dz = (p - np.eye(...)[y]) / len(z)` is the gradient of mean cross-entropy with respect to the logits. `net.backward(cache, dz, hidden_grads=hg)` backpropagates it and stores `dL/d(hidden[i])` in `hg[i]`. The score is `np.abs(h * g).sum(0)`. Note that the absolute value is taken per example before summing, so a neuron that helps some examples and hurts others still scores high. The `group` branch is the task-aware variant, explained on the Task-aware pruning page.

`structured_prune` copies the net, then for each hidden layer takes `idx = np.argsort(-scores[i])[:k]`, the indices of the `k` best neurons, and calls `net.keep_neurons(i, idx)`. All scores were computed once on the unpruned net, so layer 2's scores do not know that layer 1 just lost neurons.

`MLP.keep_neurons` in `lab/nn.py` does the surgery. `self.W[layer] = self.W[layer][:, idx]` keeps only the chosen output columns. `self.b[layer] = self.b[layer][idx]` keeps their biases. `self.W[layer + 1] = self.W[layer + 1][idx, :]` keeps only the matching input rows of the next layer, so the shapes still line up. Masks are sliced the same way, and `self.sizes[layer + 1] = len(idx)` records the new width. The first line refuses residual layers: on a residual stream `h + f(h)`, a neuron index is shared by every layer that adds into it, so they must be cut together. That set is a *dependency group*, the problem torch-pruning solves for YOLO.

`run_structured` in `lab/methods.py` prunes the 3x256 teacher to `[w] * 3` and evaluates it. It then recovers for `FT` = 500 steps at `FT_LR` = 3e-3 and compares with a fresh 3x`w` net distilled for `STEPS` = 1000 at the same learning rate. The fresh net gets twice the steps, so the comparison favours it a little.

## What the simulations show

From `prune_structured` (teacher accuracy 0.9139, 20 classes). Every recovered arm is the mean of 2 seeds. Pruned and fresh nets use the same optimizer and learning rate, so the only difference is the starting weights. "short" is 750 steps of KD and "full" is 3,000.

Before any recovery:

| keep width | l1 | act | taylor | random |
|---|---|---|---|---|
| 128 | 0.4247 | 0.4047 | 0.2994 | 0.3087 |
| 64 | 0.1685 | 0.1480 | 0.1519 | 0.1125 |
| 32 | 0.0816 | 0.0776 | 0.1104 | 0.0757 |
| 16 | 0.0443 | 0.0592 | 0.0441 | 0.0547 |

After recovery with KD:

| keep width | l1, short | fresh, short | l1, full | taylor, full | random, full | fresh, full |
|---|---|---|---|---|---|---|
| 128 | 0.9166 | 0.9108 | 0.9155 | 0.9154 | 0.9182 | 0.9176 |
| 64 | 0.8927 | 0.8685 | 0.9083 | 0.9025 | 0.9043 | 0.9042 |
| 32 | 0.7759 | 0.7439 | 0.8603 | 0.8360 | 0.8484 | 0.8469 |
| 16 | 0.5512 | 0.5484 | 0.6856 | 0.6579 | 0.6517 | 0.7007 |

Halving every layer without recovery drops accuracy from 0.9139 to between 0.30 and 0.42: the cliff pigeon's STATUS.md reports for YOLOv8n, where even a 5% one-shot channel cut sent bird mAP@50 to 0. Recovery repairs a 2x cut completely. Whether the pruned weights are a better start than a random init depends on the budget. With the short budget, the L1-pruned net is ahead of a fresh one at widths 128-32 (0.8927 against 0.8685 at width 64). With the full budget the advantage mostly disappears: the arms are within about 0.01 at 128 and 64, L1 is still ahead at 32 (0.8603 against 0.8469), and at width 16 (a 16x cut) the fresh net wins (0.7007 against 0.6856). The pruned start buys speed of recovery more than a better final model. L1, the data-free criterion, is the best pruned arm from width 64 down after full recovery. Taylor, despite using data and gradients, trails random after full recovery at widths 128-32 and edges it only at 16. With labels instead of KD, every full-budget arm ends lower: L1 at width 32 reaches 0.8426.

Page runs at the defaults (width 32, taylor, KD, 500 recovery steps): no recovery 0.1104, recovered 0.6649, and a fresh 3x32 with KD for 1,000 steps 0.7793. With l1 the recovered model reaches 0.7252, with random 0.6637, and with labels instead of KD 0.6648. This page gives the fresh net twice the pruned net's steps, so it favours the fresh net more than the experiment does.

The toy can't show the speed side. pigeon found that on Apple's MPS a 30% channel cut reduced MACs by 35% while latency got 6% worse, because small models are overhead-bound and odd channel counts tile badly.

## When it breaks

- **The cliff is steep.** Without recovery, even a 2x cut loses most of the accuracy. Never measure a structurally pruned model before fine-tuning it.
- **Big cuts lose to training from scratch.** At a sixteenth of the original width, the pruned weights end worse than a random init given the same budget. Below a few times compression, a pruned start mainly saves training time.
- **One-shot scores go stale.** Scores are computed once, so cutting layer 1 changes what layer 2's neurons mean.
- **Taylor scores are taken at a minimum.** On a well-trained net `dL/da` is small and noisy, so the score mostly ranks noise.
- **Residual streams and concatenations** tie neurons across layers. Removing one without its group breaks the shapes. `keep_neurons` refuses rather than guessing.

## Try this

- Set **keep width** to 128 and **recovery** to none, then cycle **importance** through l1, act, taylor and random. Look at how close random comes to the "smart" criteria.
- Keep width 128, set **recovery** to kd, and compare the recovered row (500 steps) with "fresh 3x128, KD from scratch" (1,000 steps). Then repeat at width 16. The pruned start has to beat a net trained twice as long.
- At width 32, compare **recovery** labels and kd, then try **importance** l1 with kd.
- Push **keep width** down to 4 to find where even the fresh net collapses. That is a capacity limit, not a pruning one.

## Going further

In a convolutional detector a "neuron" is an output channel: one filter in a conv layer, plus the matching input slice of every layer that reads it. Batch-norm parameters, concatenations and residual adds tie channels into dependency groups that must be removed together. pigeon's `harness/prune.py` uses torch-pruning's `GroupNormPruner` with `GroupMagnitudeImportance`, the conv equivalent of the `l1` criterion here, and ignores the `Detect` head. See the Channel pruning with dependency groups page. The loop pigeon settled on is the one these numbers argue for: a ~12% cut fine-tuned on 400 bird images recovered to 0.388 mAP@50, 91% of native 0.428.
