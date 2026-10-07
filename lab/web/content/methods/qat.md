## The idea

Post-training quantization rounds a finished network and hopes it copes. *Quantization-aware training* (QAT) lets the network adapt to the rounding. It keeps training with quantized weights in the forward pass, so the loss sees the real rounding error, while a full-precision copy of every weight collects the gradient updates. Rounding has a derivative of zero almost everywhere, so a plain gradient would never move anything. The *straight-through estimator* (STE) handles this by pretending rounding is the identity in the backward pass: the gradient computed for the rounded weight is applied directly to the full-precision weight.

## How it works

1. Start from a trained full-precision network `W`.
2. Each training step, compute `w_hat = quantize(W)` with the same rule PTQ uses (here symmetric, per group of 16).
3. Forward pass with `w_hat`, loss on labels or on the teacher's outputs (*knowledge distillation*, KD).
4. Backward pass: compute `dL/dw_hat` as usual, then update `W -= lr * dL/dw_hat`. This is the STE.
5. Small updates pile up in `W` until a weight crosses a rounding boundary and jumps to the next level.
6. Ship `quantize(W)`. The full-precision copy is thrown away.

```python
for X, target in data:
    w_hat = fake_quantize(W, bits, 16)
    loss = L(forward(X, w_hat), target)
    W -= lr * grad(loss, w_hat)        # straight-through
```

## The code, line by line

Nothing in `lab/nn.py` is QAT-specific. Two lines make it happen when `net.qbits` is set.

`MLP.eff(i)` returns the weight the network actually uses. `w = self.W[i]`, then `w = w * self.masks[i]` if there is a mask, then `w = fake_quantize(w, self.qbits, self.qgroup)` if `qbits` is set. `self.W[i]` itself is never rounded. It is the full-precision copy.

`MLP.forward` calls `w = self.eff(i)` for every layer and, when `cache=True`, stores it in `effs`. `MLP.backward` then works on the quantized network as it really ran. `gz` is the gradient at the layer's pre-activation, and `dW[i] = ins[i].T @ gz` is `dL/dw_hat`, the gradient with respect to the rounded weight. `g_in = gz @ effs[i].T` sends the gradient to earlier layers through the rounded weights, which is exact, since that is the network that ran. `backward` returns `dW + db`, and `fit` hands it to `Adam` together with `net.params()`, which is `self.W + self.b`, the full-precision tensors. That pairing, gradient for `w_hat` applied to `W`, is the whole straight-through estimator.

Two details follow. The scale is recomputed from `W` on every forward pass (`max|g| / qmax` inside `fake_quantize`), so it tracks the weights but is not learned. Learned Step Size Quantization (Esser et al.) makes the scale a trainable parameter. And no gradient is clipped: since the scale is set by the group's largest weight, no weight is ever outside the grid, so the usual "zero the gradient outside the clip range" rule never triggers.

`run_qat` in `lab/methods.py` copies the model and sets `m.qbits, m.qgroup = p["bits"], 16`. It trains for `FT` = 500 steps at `FT_LR` = 3e-3, with `scratch` (labels) or `distill` (KD from the 3x256 teacher, at `distill`'s default temperature T=1). The returned net still has `qbits` set, so it is evaluated through the quantizer. It is compared with PTQ at the same bits and group size.

## What the simulations show

From `quantization`. QAT rows are 1,500 steps, mean of 2 seeds. Everything uses group-16 scales. Accuracy on 20 classes:

| model | bits | PTQ | QAT + labels | QAT + KD |
|---|---|---|---|---|
| teacher 3x256 (fp32 0.9139) | 3 | 0.8970 | 0.9144 | 0.9137 |
| teacher | 2 | 0.6098 | 0.9065 | 0.9117 |
| teacher | 1 | 0.4356 | 0.9147 | 0.9137 |
| student 2x32 (fp32 0.8009) | 4 | 0.7670 | 0.8001 | 0.8020 |
| student | 3 | 0.6283 | 0.7603 | 0.7663 |
| student | 2 | 0.2173 | 0.4592 | 0.5184 |
| student | 1 | 0.1767 | 0.4714 | 0.5293 |

The big teacher can be trained down to 1 bit per weight (plus one fp16 scale per 16 weights) with no loss at all. The 2,132-parameter student cannot: QAT more than doubles its 2-bit accuracy over PTQ, but still ends far below fp32. For the student at 3 bits and below, KD beats labels in every row, and the bird AP shows the same: at 2 bits, 0.7183 with KD against 0.6249 with labels.

Page runs (500 steps): the default student at 2 bits goes from 0.2173 with PTQ to 0.4765 with KD (0.4673 with labels). The teacher at 1 bit goes from 0.4356 to 0.8983 with KD.

## When it breaks

- **No spare capacity.** QAT redistributes what the network knows across coarser weights. A model that was already capacity-limited has nothing to redistribute into.
- **STE mismatch.** The gradient is for `w_hat`, but the step is taken on `W`. A weight sitting just inside a rounding boundary gets the same gradient as one in the middle of its bin, so training can oscillate between levels.
- **Fixed scales.** Recomputing `scale` from the max means one weight growing changes the grid for its whole group mid-training.
- **Budget and learning rate.** QAT is a fine-tune, so too high a learning rate destroys the pretrained solution, and too few steps leaves rounding damage unrepaired.
- **Weights only.** Activations stay in float here. Integer hardware also needs activation quantization, which QAT should simulate too.

## Try this

- Keep **model** on student and step **bits** from 4 down to 1 with **train with** kd. Find where QAT stops closing the gap to fp32.
- At **bits** 2 on the student, compare **train with** labels and kd, and look at `target_ap` as well as accuracy.
- Switch **model** to teacher and set **bits** to 1. Compare the PTQ row with the QAT row.

## Going further

On large networks QAT is standard for int4 and below, usually with learned scales (LSQ), per-channel weight scales and simulated activation quantization, and often with distillation from the full-precision model, as in the KD arm here. pigeon has no QAT path yet. Its cockpit's Precision module is PTQ, and `carve.py` found PTQ collapsing at 4 bits on YOLOv8n. These numbers suggest the way below that floor is the same move pigeon already makes after pruning: fine-tune with the cut in place. For a detector, that means running the Ultralytics trainer with `harness/quant.py:fake_quantize` applied in the forward pass and an STE in the backward pass. See the Quantizing a detector page.
