## The idea

Post-training quantization rounds a finished network and hopes it copes. *Quantization-aware training* (QAT) lets the network adapt to the rounding. It trains with quantized weights in the forward pass, so the loss sees the real rounding error, while a full-precision copy of every weight collects the updates. Rounding has zero derivative almost everywhere. The *straight-through estimator* (STE) handles this by pretending rounding is the identity in the backward pass: the gradient computed for the rounded weight is applied directly to the full-precision weight.

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

`MLP.forward` calls `w = self.eff(i)` for every layer and, when `cache=True`, stores it in `effs`. `MLP.backward` then works on the quantized network as it really ran. `gz` is the gradient at the layer's pre-activation, and `dW[i] = ins[i].T @ gz` is `dL/dw_hat`, the gradient with respect to the rounded weight. `g_in = gz @ effs[i].T` passes the gradient back through the rounded weights, exactly as the network ran. `fit` hands `dW + db` to `Adam` together with `net.params()`, which is `self.W + self.b`, the full-precision tensors. That pairing, gradient for `w_hat` applied to `W`, is the whole straight-through estimator.

The scale is recomputed from `W` on every forward pass, so it tracks the weights but is not learned, as it is in Learned Step Size Quantization (Esser et al.). No gradient is clipped: the scale comes from the group's largest weight, so no weight ever falls outside the grid.

`run_qat` in `lab/methods.py` copies the model and sets `m.qbits, m.qgroup = p["bits"], 16`. It trains for `FT` = 500 steps at `FT_LR` = 3e-3, with `scratch` (labels) or `distill` (KD from the 3x256 teacher, at `distill`'s default temperature T=1). The returned net keeps `qbits` set, so it is evaluated through the quantizer, next to PTQ at the same bits and group size.

## What the simulations show

From `quantization`. QAT rows are 1,500 steps, mean of 2 seeds. Everything uses group-16 scales. Accuracy on 20 classes:

| model | bits | PTQ | QAT + labels | QAT + KD |
|---|---|---|---|---|
| teacher 3x256 (fp32 0.9139) | 3 | 0.8970 | 0.9160 | 0.9186 |
| teacher | 2 | 0.6098 | 0.9093 | 0.9171 |
| teacher | 1 | 0.4356 | 0.9178 | 0.9194 |
| student 2x32 (fp32 0.8281) | 4 | 0.7944 | 0.8327 | 0.8360 |
| student | 3 | 0.6397 | 0.7909 | 0.7912 |
| student | 2 | 0.1683 | 0.4789 | 0.4880 |
| student | 1 | 0.1680 | 0.5377 | 0.5291 |

The big teacher can be trained down to 1 bit per weight (plus one fp16 scale per 16 weights) and ends *above* its fp32 accuracy. That is not because the rounding helps: the QAT run is also 1,500 more training steps, which the fp32 model never got. The 2,132-parameter student cannot do the same. QAT roughly triples its 2-bit accuracy over PTQ but still ends far below fp32. Labels and KD come out almost even for the student. The bird AP at 2 bits even favours labels (0.7592 against 0.7024).

Page runs (500 steps): the default student at 2 bits goes from 0.1897 with PTQ to 0.4602 with KD (0.4260 with labels). A 2-bit student this small is chaotic: the same run in native numpy instead of the browser lands about 0.04 away, so read the KD lead as noise. The 2-seed means above have labels and KD within 0.01. The teacher at 1 bit goes from 0.4256 to 0.9118 with KD.

## When it breaks

- **No spare capacity.** QAT redistributes knowledge across coarser weights. A capacity-limited model has nowhere to put it.
- **STE mismatch.** The gradient is for `w_hat`, but the step is taken on `W`. A weight sitting just inside a rounding boundary gets the same gradient as one in the middle of its bin, so training can oscillate between levels.
- **Fixed scales.** Recomputing `scale` from the max means one weight growing changes the grid for its whole group mid-training.
- **Budget and learning rate.** Too high a rate destroys the pretrained solution, and too few steps leave rounding damage unrepaired.
- **Weights only.** Activations stay in float here. Integer hardware also needs activation quantization, which QAT should simulate too.

## Try this

- Keep **model** on student and step **bits** from 4 down to 1 with **train with** kd. Find where QAT stops closing the gap to fp32.
- At **bits** 2 on the student, compare **train with** labels and kd, and look at `target_ap` as well as accuracy.
- Switch **model** to teacher and set **bits** to 1. Compare the PTQ row with the QAT row.

## Going further

On large networks QAT is standard for int4 and below, usually with learned scales (LSQ), per-channel weight scales and simulated activation quantization, and often with distillation from the full-precision model, as in the KD arm here. The YOLOv8n cockpit's Precision module is PTQ, and `carve.py` found PTQ collapsing at 4 bits on YOLOv8n. Below that floor the cockpit uses QAT: when a recipe quantizes, Regrow fine-tunes with the quantizer in the loop. `harness/precision.py:enable_qat` swaps each quantized conv for a `QATConv2d`, whose forward pass uses the rounded weights and whose backward pass is a straight-through estimator (see Prune, then fine-tune). That took a 3.16-bit recipe from 0.003 to 0.584 bird mAP@50, 70% of native, in 3 CPU epochs (cockpit/STATUS.md, on a VOC-2007 bird stand-in set). See also Quantizing a detector.
