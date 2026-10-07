## The idea

Many weights in a trained network are close to zero. Magnitude pruning treats `|w|` as a stand-in for importance and sets the smallest weights to exactly zero. This is *unstructured* pruning: the matrix keeps its shape but becomes *sparse* (mostly zeros). A *recovery* phase (more training with removed weights held at zero) wins back most of the lost accuracy.

## How it works

1. Pick a *sparsity* `s`, the fraction of all weights to remove (0.9 keeps 1 in 10).
2. Rank weights by `|w|`. A *global* ranking puts every layer in one list. A *per-layer* ranking removes the same fraction from each layer.
3. Let `k = s * n_weights`. The threshold `thr` is the `k`-th smallest `|w|`. The *mask* is `m = (|w| > thr)`, a 0/1 array shaped like `W`.
4. Use `W * m` from now on, and multiply the gradient by `m` too, so dead weights stay dead.
5. Recover with labels or *knowledge distillation* (KD), where the pruned net learns to match the unpruned net's output probabilities.
6. Gradual variant: repeat steps 2-5 with a rising `s`.

```python
mags = concat(|W_i| for every layer)          # already-pruned weights count as 0
thr = kth_smallest(mags, int(s * len(mags)))
for i in layers:
    W_i = W_i * (|W_i| > thr)
```

## The code, line by line

`magnitude_prune` in `lab/compress.py` starts with `net = net.copy()`, so the input is never modified.

The helper `mask_for(mags, flat)` does the ranking. `k = int(round(sparsity * flat.size))` counts how many weights must be zero, over *all* weights, already-pruned ones included. `thr = np.partition(flat, k - 1)[k - 1]` finds the `k`-th smallest magnitude without a full sort, and the mask is `mags > thr`. Dead weights have magnitude 0, sit at the bottom and stay dead. So calling the function again with a higher target removes only the difference.

In the global branch, `flat = np.concatenate([np.abs(net.eff(i)).ravel() for i in layers])` pools every layer's magnitudes, so one threshold applies everywhere. `net.eff(i)` is the weight as the forward pass uses it. The per-layer branch calls `mask_for(w, w.ravel())` once per layer.

`net.apply_masks()` writes the zeros in (`self.W[i] *= m` in `lab/nn.py`). `MLP.eff` multiplies by the mask on every forward pass, `MLP.backward` does `dW[i] *= self.masks[i]`, and `fit` re-applies masks after each optimizer step, so dead weights stay dead.

`run_magnitude` in `lab/methods.py` calls `magnitude_prune(m, 1 - (1 - sp) ** (i / k))` for gradual step `i` of `k`. That is a geometric schedule: each step removes the same share of the survivors. For `sp = 0.9`, `k = 5` the targets are about 0.37, 0.60, 0.75, 0.84 and 0.90. Each step is followed by `scratch` (labels) or `distill` (KD from the unpruned teacher) for `FT // k` steps, so the total recovery budget is fixed.

## What the simulations show

The teacher is a 3x256 MLP: 140,052 parameters, 0.9139 accuracy on 20 classes, 0.9885 average precision (AP) on class 0, the "bird". AP is the same kind of single-class ranking score as pigeon's bird mAP@50. From `prune_unstructured` (accuracy; 1,500 recovery steps; mean of 2 seeds):

| sparsity | no recovery | + labels | + KD from unpruned self | gradual (5 steps) + KD | dense net, same weight count, KD |
|---|---|---|---|---|---|
| 0.5 | 0.8961 | 0.9154 | 0.9193 | 0.9173 | 0.9203 |
| 0.75 | 0.7500 | 0.9142 | 0.9199 | 0.9183 | 0.9188 |
| 0.9 | 0.2860 | 0.9019 | 0.9125 | 0.9038 | 0.9107 |
| 0.95 | 0.1206 | 0.8741 | 0.8891 | 0.8576 | 0.8977 |
| 0.98 | 0.0597 | 0.6215 | 0.6289 | 0.6179 | 0.8531 |
| 0.99 | 0.0540 | 0.0804 | 0.0883 | 0.4007 | 0.7792 |

Without recovery there is a cliff between 75% and 90%, the shape pigeon saw for channel cuts on YOLOv8n. KD recovery brings 90% sparsity back to 0.9125. At 50-75% it ends slightly *above* the teacher, because recovery is also extra training on 28,000 inputs. KD beats labels at every sparsity. Gradual pruning is a little worse than one-shot up to 98%, but at 99% it is the only sparse arm that survives (0.4007 against 0.0883). From 95% up, a small *dense* net with the same weight count beats every sparse one.

Page runs (500 recovery steps): the defaults (0.9, KD, global) recover from 0.286 to 0.8945, and labels reach 0.8901. Per-layer ranking starts at 0.1367 and recovers to 0.8573. Five gradual steps reach 0.8641 at 0.9 and 0.7097 at 0.95, below one-shot (0.8945, 0.8213), because each recovery gets only 100 steps.

## When it breaks

- **Sparse is not small or fast by itself.** Savings need a sparse storage format, sparse kernels, or hardware patterns such as 2:4.
- **Per-layer ranking** cuts the small input and output layers as hard as the big hidden ones.
- **Fraction of what?** If a pruning function ranks only surviving weights, repeated calls with rising total targets compound. An earlier version of this function did that: its 5-step "0.9" schedule ended at 99.9%. Check the nonzero count.
- **Past the cliff** whole neurons lose every input or output weight, and one-shot recovery has nothing left to retrain.
- **Magnitude is a proxy.** A small weight on a large activation can matter more than a large one on a near-zero input.

## Try this

- Compare **recovery** none and kd at **fraction removed** 0.75, then 0.95.
- Set **gradual steps** to 5 at 0.9 and at 0.95, and compare with 1 step. The budget is fixed, so more steps means shorter recoveries.
- Switch **ranking** between global and per-layer at 0.9 and compare the unrecovered rows.
- Compare **recovery** labels and kd at 0.9.

## Going further

On large models unstructured sparsity pays off only with a runtime that exploits it, such as sparse formats on CPUs or N:M patterns on recent GPUs. See also Wanda and SparseGPT. pigeon does not prune single weights. Its cockpit zeros whole channels (`harness/mask.py` masks the lowest-L1 output channels of a stage), because a channel can actually be removed from a conv layer. The cliff-then-recover pattern is the lesson pigeon's README draws: the loop must be prune → fine-tune → test, never prune → test.
