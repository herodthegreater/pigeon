## The idea

Distillation needs inputs to show the teacher. When the training images are gone, DeepInversion makes them from the teacher itself. Every batch-norm (BN) layer stores the running mean and variance of its input over the training set. Optimize random-noise images until the features they cause match those stored statistics, while the teacher predicts a chosen class, and you get images that look like training data to the network.

## How it works

1. Make a batch $x$ of random images a trainable tensor; pick target classes $y$.
2. Run the frozen teacher $f$ in eval mode. At each BN layer $l$, hook its input $a_l$ and take the per-channel batch mean $\mu_l(x)$ and variance $\sigma_l^2(x)$. For a batch of $B$ feature maps of size $H \times W$ indexed by $b, h, w$, channel $c$ gives

$$
\mu_{l,c}(x) = \frac{1}{BHW} \sum_{b,h,w} a_{l,bchw}, \qquad \sigma_{l,c}^2(x) = \frac{1}{BHW} \sum_{b,h,w} \left( a_{l,bchw} - \mu_{l,c}(x) \right)^2
$$

3. Loss: the cross-entropy of the teacher's prediction $f(x)$ against $y$, plus a BN statistics term that compares the batch statistics with the layer's running mean $\hat{\mu}_l$ and variance $\hat{\sigma}_l^2$, plus a total-variation prior and an L2 prior. TV, total variation, sums differences between neighbouring pixels and keeps images smooth.

$$
\mathcal{R}_{\text{BN}}(x) = \sum_l \left( \lVert \mu_l(x) - \hat{\mu}_l \rVert_2 + \lVert \sigma_l^2(x) - \hat{\sigma}_l^2 \rVert_2 \right)
$$

$$
\mathcal{L}(x) = \operatorname{CE}(f(x), y) + \alpha_{\text{bn}} \, \mathcal{R}_{\text{BN}}(x) + \alpha_{\text{tv}} \operatorname{TV}(x) + \alpha_{\text{l2}} \lVert x \rVert_2^2
$$

4. Update only $x$ with Adam for a few thousand steps, randomly jittering and flipping it each step.
5. Repeat for many batches, then distill the student on the synthetic pool.

```python
x = torch.randn(B, 3, H, W, requires_grad=True)
feats = []
for bn in [m for m in teacher.modules() if isinstance(m, nn.BatchNorm2d)]:
    bn.register_forward_hook(lambda m, i, o: feats.append((m, i[0])))
opt = torch.optim.Adam([x], lr=0.1)
for step in range(2000):
    feats.clear()
    loss = F.cross_entropy(teacher(jitter(x)), y)
    for bn, f in feats:
        mu, var = f.mean((0, 2, 3)), f.var((0, 2, 3), unbiased=False)
        loss = loss + a_bn * ((mu - bn.running_mean).norm() + (var - bn.running_var).norm())
    loss = loss + a_tv * total_variation(x) + a_l2 * x.pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
```

## The code, line by line

`invert_teacher` in lab/experiments/transfer.py is the toy version without the BN term. `X = rng.standard_normal(...)` starts from noise, `c` holds random target classes and `Y` their one-hot rows. `Adam([X], lr=lr)` optimizes the inputs, not the weights. Each step, `dz = (softmax(z) - Y) / n` is $\partial \mathcal{L} / \partial z = (\softmax(z) - Y) / n$, the gradient of the mean cross-entropy with respect to the logits $z$, `t.backward(cache, dz)` carries it back to the input as `dX`, and `opt.step([dX + l2 * X / n])` adds the gradient of the L2 prior (the `a_l2` term above, $\alpha_{\text{l2}}$ in the loss). The toy MLP has no BN layers, so the statistics term, the part that makes DeepInversion work, is missing.

## What the simulations show

Toy-MLP analogue, not YOLO: `transfer_set` (lab/results/transfer_set.json). A student distilled on 10,000 inverted inputs reached accuracy 0.667, against 0.6601 on Gaussian noise, 0.6662 on uniform noise and 0.7996 on the real 8k labeled set (teacher 0.9139). Without a BN prior, inversion barely beats noise here.

## When it breaks

- It needs unfused BN layers. server.py fuses BN into the convs during warm-up, so start from a fresh `YOLO("yolov8n.pt")`.
- Matching statistics does not guarantee diverse images; batches can collapse onto a few patterns per class.
- A detector has no single class output for `CE`. You would need a detection target, such as raising the bird score at chosen grid cells.

## Going further

Missing here: a torch inversion loop with BN hooks, a detection-shaped target loss, and a detector distillation trainer (see detector_kd). The closest analogue is `invert_teacher`.
