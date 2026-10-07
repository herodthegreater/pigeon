## The idea

A conv layer's output is a feature map of shape C x H x W: C channels over an H x W grid. Attention transfer squashes it to one H x W map of where the network is active, by summing squared activations over channels. The student is trained to make its maps match the teacher's at a few chosen depths. The channel dimension is gone, so a narrow student can match a wide teacher without an extra projection layer.

## How it works

1. Pick layer pairs, one in the teacher and one in the student, with the same H x W, for example the end of each resolution stage.
2. For a feature map `A` (batch B x C x H x W): `Q = sum over c of A[:, c]²`, shape B x H x W. Flatten to B x HW and normalize each row: `q = Q / ||Q||`.
3. Pair loss: `||q_s - q_t||²`, averaged over the batch.
4. Total: `L = task_loss + beta * Σ_pairs pair_loss`. Normalized maps differ by small amounts, so `beta` must be large enough for this term to matter next to the task loss.

```python
def attention(f):                                  # f: B x C x H x W
    return F.normalize(f.pow(2).mean(1).flatten(1), dim=1)   # B x HW

loss = task_loss(student_out, targets)
for fs, ft in zip(student_feats, teacher_feats):   # captured with forward hooks
    if fs.shape[-2:] != ft.shape[-2:]:
        fs = F.interpolate(fs, size=ft.shape[-2:], mode="bilinear")
    loss = loss + beta * (attention(fs) - attention(ft.detach())).pow(2).sum(1).mean()
```

## The code, line by line

`Hint` in lab/train.py is FitNets, an earlier feature-matching loss. `hs, ht = hid[self.s], b.t_hidden[self.t]` picks the student and teacher hidden layers. `r = hs @ self.P - ht` is the residual after a learned projection `P` (student width x teacher width). The loss is `self.w * (r ** 2).mean()`. `self.dP = self.w * hs.T @ (2 * r / n)` is its gradient for `P`, and `dh = self.w * (2 * r / n) @ self.P.T` is the gradient sent into the student layer, returned as `{self.s: dh}` with a zero logit gradient. Attention transfer replaces `hs @ self.P - ht` with `attention(hs) - attention(ht)` and drops `P`: nothing to learn, because the channel sum already removes the width mismatch.

## What the simulations show

Toy-MLP analogue, not YOLO: `loss_zoo` (lab/results/loss_zoo.json). Adding a FitNets hint to KD gave accuracy 0.6395, against 0.6398 for KD alone; hint plus labels gave 0.6223, against 0.6637 for labels alone. Feature matching added nothing here. An MLP has no spatial grid, so an attention map can't be formed at all.

## When it breaks

- Map sizes must match; for a detector, pair the P3, P4 and P5 neck outputs.
- The map keeps where, not what. A student can fire in the right place with the wrong features.
- Pairing a shallow student layer with a deep teacher layer forces the wrong job on it.

## Going further

Missing here: a torch loop running teacher and student together with hooks, and a trainer whose loss accepts extra terms. pigeon's pruned student has the teacher's layout, so spatial sizes line up at every stage. The closest analogue is `Hint`.
