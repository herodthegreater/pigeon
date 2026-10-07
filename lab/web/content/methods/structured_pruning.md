## The idea

Unstructured pruning leaves zeros that still get stored and multiplied. Structured pruning deletes whole *neurons*. Removing one hidden unit removes one column of the matrix feeding it and one row of the matrix it feeds, so the matrices really shrink and the model is smaller and faster on any hardware. Which neurons to keep is decided by an *importance score*. In a conv net the same idea removes channels, which is what pigeon does to YOLOv8n.

## How it works

1. Run the trained net on a probe batch (2,000 training examples here).
2. Score every hidden neuron. `l1` is the sum of `|w|` over its incoming weights. `act` is its mean `|activation|`. `taylor` is the sum over examples of `|a * dL/da|`, where `a` is the activation and `dL/da` the loss gradient with respect to it: a first-order estimate of the loss change if `a` were zeroed. `random` is a control.
3. In every hidden layer, keep the `k` best neurons and delete the rest.
4. Recover by training the smaller net, with labels or by distilling from the original.

## The code, line by line

`neuron_importance` in `lab/compress.py` returns one score array per hidden layer. For `l1`, `np.abs(net.W[i]).sum(0)` sums each column of `W[i]`, the weights going *into* each neuron. It never looks at data. The other criteria run `z, hid, cache = net.forward(X, cache=True)`, where `hid` holds each hidden layer's post-ReLU output, and `act` is `np.abs(h).mean(0)`.

For `taylor`, `dz = (p - np.eye(...)[y]) / len(z)` is the gradient of mean cross-entropy with respect to the logits. `net.backward(cache, dz, hidden_grads=hg)` stores `dL/d(hidden[i])` in `hg[i]`, and the score is `np.abs(h * g).sum(0)`. Taking `abs` per example means a neuron that helps some examples and hurts others still scores high. The `group` branch is the task-aware variant (see Task-aware pruning).

`structured_prune` copies the net, takes `idx = np.argsort(-scores[i])[:k]` per hidden layer, and calls `net.keep_neurons(i, idx)`. Scores are computed once, on the unpruned net.

`MLP.keep_neurons` in `lab/nn.py` does the surgery. `self.W[layer][:, idx]` keeps the chosen output columns, `self.b[layer][idx]` their biases, and `self.W[layer + 1][idx, :]` the matching input rows of the next layer, so shapes still line up. Masks are sliced the same way and `self.sizes[layer + 1] = len(idx)` records the new width. The first line refuses residual layers. On a stream `h + f(h)`, one neuron index is shared by every layer that adds into it, and that *dependency group* must be cut together.

`run_structured` in `lab/methods.py` prunes the 3x256 teacher to `[w] * 3`, recovers for `FT` = 500 steps, and compares with a fresh 3x`w` net distilled for 1,000 steps at the same learning rate.

## What the simulations show

From `prune_structured` (teacher accuracy 0.9139; recovered arms are means of 2 seeds; pruned and fresh nets share optimizer and learning rate; "short" is 750 KD steps and "full" 3,000).

Before recovery:

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

Halving every layer drops accuracy to 0.30-0.42 before recovery. That is the cliff pigeon's STATUS.md reports for YOLOv8n, where even a 5% one-shot channel cut sent bird mAP@50 to 0. Recovery repairs a 2x cut completely. With the short budget, the L1-pruned net beats a fresh one at widths 128-32 (0.8927 against 0.8685 at 64). With the full budget the arms are within 0.01 at 128 and 64, L1 still leads at 32 (0.8603 against 0.8469), and at a 16x cut the fresh net wins (0.7007 against 0.6856). The pruned start mostly buys faster recovery. L1, which ignores data, is the best criterion from width 64 down. Taylor trails random at widths 128-32. Recovery with labels instead of KD ends lower in every full-budget arm.

Page runs at the defaults (width 32, taylor, KD): no recovery 0.1104, recovered 0.6649, fresh 0.7793. With l1, 0.7252. With random, 0.6637. With labels, 0.6648. The page gives the fresh net twice the steps, so it favours it.

The toy can't show speed. On Apple's MPS, pigeon saw a 30% channel cut reduce MACs by 35% while latency got 6% worse.

## When it breaks

- **The cliff is steep.** Never judge a structurally pruned model before fine-tuning it.
- **Deep cuts lose to a fresh start.** At 16x the pruned weights end worse than a random init.
- **Stale scores.** Scores come from the unpruned net, so cutting layer 1 changes what layer 2's neurons mean.
- **Taylor at a minimum.** On a well-trained net `dL/da` is small and noisy.
- **Residual streams and concatenations** tie neurons across layers, and `keep_neurons` refuses them rather than guessing.

## Try this

- Set **keep width** 128, **recovery** none, and cycle **importance**. See how close random comes.
- At **keep width** 128 with **recovery** kd, compare the recovered row with "fresh 3x128, KD from scratch". Repeat at 16.
- At width 32, compare **recovery** labels and kd, then **importance** l1.

## Going further

In a detector a "neuron" is an output channel: one conv filter plus the matching input slice of every layer that reads it. Batch-norm, concatenations and residual adds tie channels into dependency groups. pigeon's `harness/prune.py` uses torch-pruning's `GroupNormPruner` with `GroupMagnitudeImportance`, the conv equivalent of `l1` here, and leaves the `Detect` head alone (see Channel pruning with dependency groups). Its loop matches what these numbers argue for: a ~12% cut fine-tuned on 400 bird images recovered to 0.388 mAP@50, 91% of native 0.428.
