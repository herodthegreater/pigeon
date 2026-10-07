## The idea

A trained network has many weights that are close to zero. Magnitude pruning treats a weight's absolute value as a stand-in for its importance: set the smallest ones to exactly zero and the outputs should barely move. This is *unstructured* pruning: individual weights anywhere in a matrix are removed, so the matrix keeps its shape but becomes *sparse* (mostly zeros). The cut always costs some accuracy. A short *recovery* phase gets most of it back. Recovery means more training in which the removed weights are held at zero.

## How it works

1. Pick a *sparsity* `s`, the fraction of weights to remove (0.9 = keep 1 in 10).
2. Rank the weights by `|w|`. With a *global* ranking, every weight in every layer competes in one list. With a *per-layer* ranking, each layer loses the same fraction.
3. Find the threshold `thr` = the `s`-quantile of the ranked values. Build a 0/1 *mask* `m = (|w| > thr)` with the same shape as `W`.
4. Store `W * m`. From now on the forward pass uses `W * m`, and the gradient for `W` is multiplied by `m` too, so a dead weight cannot come back.
5. Recover: train for a while with the mask in place, using either the original labels or *knowledge distillation* (KD). In KD the pruned net learns to match the unpruned net's output probabilities.
6. Gradual variant: do steps 2-5 several times, raising the sparsity a little each time.

```python
alive = concat(|W_i| for every layer, nonzero entries only)
thr = quantile(alive, s)
for each layer i:
    mask_i = |W_i| > thr
    W_i = W_i * mask_i
```

## The code, line by line

`magnitude_prune` in `lab/compress.py` starts with `net = net.copy()`, so the input model is never modified. `layers` lists every layer, unless `spare_last=True`, which keeps the 256x20 output layer dense.

The inner helper `mask_for(mags, flat)` does the ranking. `k = int(round(sparsity * flat.size))` is the number of weights that must be zero, counted over *all* weights, including ones already pruned. `thr = np.partition(flat, k - 1)[k - 1]` finds the `k`-th smallest magnitude without a full sort, and the mask is `mags > thr`. The comparison is strict, so a weight exactly at the threshold is removed. Already-dead weights have magnitude 0 and sit at the bottom of the list, so they count towards `k` and stay dead. Because `sparsity` is a fraction of all weights, calling the function again with a higher target only removes the difference. The `k == 0` branch keeps every nonzero weight.

In the global branch, `flat = np.concatenate([np.abs(net.eff(i)).ravel() for i in layers])` puts every layer's magnitudes into one list, so one threshold applies everywhere. `net.eff(i)` is the weight the forward pass actually uses, with any existing mask and quantization already applied. The per-layer branch calls `mask_for(w, w.ravel())` with each layer's own list, so every layer ends at the same sparsity.

`net.apply_masks()` calls `MLP.apply_masks` in `lab/nn.py`, which runs `self.W[i] *= m` for every layer that has a mask. This physically writes the zeros into the weights. The mask then keeps them at zero in three places. `MLP.eff` computes `w = w * self.masks[i]` on every forward pass. `MLP.backward` does `dW[i] *= self.masks[i]`, so pruned weights get no gradient. `fit` in `lab/train.py` calls `net.apply_masks()` again after every optimizer step.

`run_magnitude` in `lab/methods.py` drives the page. For `k` gradual steps it calls `magnitude_prune(m, 1 - (1 - sp) ** (i / k))` on the already-pruned `m`. That is a geometric schedule of total sparsity: each step removes the same fraction of the weights still alive, and step `k` lands exactly on `sp`. With `sp = 0.9` and `k = 5`, the targets are about 0.37, 0.60, 0.75, 0.84 and 0.90. After each step the model is recovered with `scratch` (labels) or `distill` (KD from the unpruned teacher `t`) for `FT // k` steps at learning rate `FT_LR`.

## What the simulations show

The teacher is a 3x256 MLP with 140,052 parameters, 0.9139 accuracy on the 20 classes and 0.9885 average precision (AP) on class 0, the "bird". AP is the ranking score `lab/train.py` computes in the same way as pigeon's bird mAP@50. Accuracy from `prune_unstructured` (recovery is 1,500 steps, mean of 2 seeds; "gradual" spreads the same 1,500 steps over 5 prune-and-recover rounds):

| sparsity | no recovery | + labels | + KD from unpruned self | gradual (5 steps) + KD | dense net, same weight count, KD |
|---|---|---|---|---|---|
| 0.5 | 0.8961 | 0.9154 | 0.9193 | 0.9173 | 0.9203 |
| 0.75 | 0.7500 | 0.9142 | 0.9199 | 0.9183 | 0.9188 |
| 0.9 | 0.2860 | 0.9019 | 0.9125 | 0.9038 | 0.9107 |
| 0.95 | 0.1206 | 0.8741 | 0.8891 | 0.8576 | 0.8977 |
| 0.98 | 0.0597 | 0.6215 | 0.6289 | 0.6179 | 0.8531 |
| 0.99 | 0.0540 | 0.0804 | 0.0883 | 0.4007 | 0.7792 |

Without recovery there is a cliff between 75% and 90% sparsity, the same shape pigeon saw for structural cuts on YOLOv8n. Recovery by KD brings 90% sparsity back to 0.9125, level with the teacher. Recovered models at 50-75% sparsity even score slightly *above* the teacher. Recovery is also extra training on 28,000 inputs the teacher never trained on, so "above the teacher" says more about the teacher than about pruning. KD beats labels at every sparsity. Gradual pruning is slightly worse than one-shot up to 98%, but at 99% it is the only sparse arm that survives (0.4007 against 0.0883): a one-shot cut that deep disconnects too much for recovery to find. From 95% up, a small *dense* network with the same number of weights beats every sparse one.

Runs on this page use a shorter recovery (500 steps in total). At the defaults (0.9, KD, global), accuracy recovers from 0.286 to 0.8945. With **recovery** set to labels it reaches 0.8901. With a per-layer ranking, the unrecovered model scores 0.1367 and recovers to 0.8573. Gradual pruning in 5 steps scored 0.8641 at 0.9 and 0.7097 at 0.95, both below one-shot (0.8945 and 0.8213), because each of its 5 recoveries gets only 100 steps.

The toy world can't tell us about speed. Every model here runs dense numpy matmuls, so zeros cost as much as any other weight.

## When it breaks

- **Sparse is not small or fast by itself.** The zeros still occupy the dense matrix and still get multiplied. You only save memory with a sparse storage format, and you only save time with sparse kernels or hardware that supports a fixed pattern such as 2:4.
- **Per-layer ranking hurts.** It forces the tiny 12x256 input layer and the 256x20 output layer to lose as large a share as the big 256x256 hidden layers.
- **Fraction of what?** A pruning function that ranks only the surviving weights reads its argument as a fraction of those survivors. Repeated calls with rising total targets then compound and overshoot. An earlier version of this lab's `magnitude_prune` did exactly that, and its 5-step "0.9" schedule ended at 99.9% sparsity. Always check the nonzero count.
- **Past the cliff, one-shot recovery stops working.** At 99% sparsity, one-shot pruning followed by either recovery stays below 0.09 accuracy. Whole neurons have lost every input or output weight, and there is nothing left to retrain. Only gradual pruning gets through, and a dense net of the same size does far better.
- **Magnitude is only a proxy.** A small weight on an input with a large activation can matter more than a large weight on an input that is nearly always zero.

## Try this

- Set **fraction removed** to 0.75 and compare **recovery** none and kd. Then try 0.95. Watch the gap between "one-shot, no recovery" and the recovered row widen and then close again as the cut goes past what recovery can fix.
- Set **gradual steps** to 5 at 0.9, then at 0.95. Compare the gradual row with "one-shot, no recovery" and with a 1-step KD run. The budget is fixed at 500 steps, so more steps means shorter recoveries.
- Switch **ranking** between global and per-layer at 0.9. Compare the unrecovered rows to see how much the global ranking protects the small layers.
- Compare **recovery** labels and kd at 0.9. With 8,000 clean labels, distillation has no advantage over labels here.

## Going further

On large models, unstructured sparsity pays off only with a runtime that exploits it: sparse matrix formats on CPUs, or N:M patterns (2 nonzeros in every 4) on recent GPUs. The one-shot, layer-by-layer versions for language models are the Wanda and SparseGPT pages in this lab. pigeon does not prune individual weights. Its cockpit zeros whole channels (`harness/mask.py` masks the lowest-L1 output channels of a stage), because a channel can actually be removed from a conv layer and a scattered zero cannot. The cliff-then-recover pattern above is the same lesson pigeon's README draws: the loop must be prune → fine-tune → test, never prune → test.
