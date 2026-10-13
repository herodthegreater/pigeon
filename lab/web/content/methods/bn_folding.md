## The idea

Batch norm (BN) normalizes each channel with a mean $\mu$ and variance $\sigma^2$, then applies a learned scale $\gamma$ and shift $\beta$. At inference, $\mu$ and $\sigma^2$ are fixed running averages, so BN is just $y = s x + t$, one multiply and one add per channel. A convolution is also linear, and two linear maps in a row make one. So BN can be absorbed exactly into the conv before it and deleted. Do this before quantizing, so you round the weights the deployed model will actually use.

## How it works

For each conv output channel $o$:

1. $s_o = \gamma_o / \sqrt{\sigma_o^2 + \epsilon}$.
2. $W'_o = s_o W_o$: scale the whole filter.
3. $b'_o = (b_o - \mu_o) \, s_o + \beta_o$, with $b = 0$ if the conv had no bias (YOLOv8's convs don't).
4. Replace conv + BN with one conv holding $W'$ and $b'$.

Why it is exact: channel $o$ of the conv computes $W_o \circledast x + b_o$ ($\circledast$ is the convolution), and BN then gives

$$
\begin{aligned}
y_o &= \gamma_o \, \frac{W_o \circledast x + b_o - \mu_o}{\sqrt{\sigma_o^2 + \epsilon}} + \beta_o \\
&= \left( W_o \, \frac{\gamma_o}{\sqrt{\sigma_o^2 + \epsilon}} \right) \circledast x \\
&\quad + (b_o - \mu_o) \, \frac{\gamma_o}{\sqrt{\sigma_o^2 + \epsilon}} + \beta_o
\end{aligned}
$$

so the folded conv has, per output channel $o$,

$$
\begin{aligned}
W'_o &= W_o \, \frac{\gamma_o}{\sqrt{\sigma_o^2 + \epsilon}} \\
b'_o &= (b_o - \mu_o) \, \frac{\gamma_o}{\sqrt{\sigma_o^2 + \epsilon}} + \beta_o
\end{aligned}
$$

Example: $\gamma = 1$, $\sigma^2 + \epsilon = 4$, $\mu = 1$, $\beta = 0.2$, no bias. Then $s = 0.5$, $W' = 0.5 \, W$, $b' = -0.3$. Check: BN of $x$ is $(x - 1) / 2 + 0.2 = 0.5x - 0.3$.

```python
@torch.no_grad()
def fold(conv, bn):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                      conv.padding, conv.dilation, conv.groups, bias=True)
    fused.weight.copy_(conv.weight * s.view(-1, 1, 1, 1))
    b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    fused.bias.copy_((b - bn.running_mean) * s + bn.bias)
    return fused
```

## The code, line by line

`MLP.collapse_linear` in lab/nn.py is the same algebra on the toy network, which writes layers as `h @ W + b`. For each layer flagged `net.lin[i]` (no activation after it), `W2 = net.W[i + 1]`, then `net.b[i + 1] = net.b[i] @ W2 + net.b[i + 1]` and `net.W[i + 1] = net.W[i] @ W2`, and layer `i` is deleted. The docstring states the identity, `(h W1 + b1) W2 + b2 == h (W1 W2) + (b1 W2 + b2)`:

$$
(h W_1 + b_1) W_2 + b_2 = h (W_1 W_2) + (b_1 W_2 + b_2)
$$

BN folding is the case where the second map is diagonal: $W_2 = \diag(s)$, $b_2 = \beta - \mu \odot s$. The `assert` refuses residual or masked layers, because folding is exact only when nothing else reads the intermediate value.

The cockpit already folds YOLOv8's BN: server.py, carve.py and sweep.py call `fuse(yolo)` from harness/model.py before `snapshot`. It calls `yolo.model.fuse()` explicitly, because newer Ultralytics `predict()` fuses a private copy and leaves `yolo.model` unfused.

## When it breaks

- Fold only with frozen running statistics. In training mode BN uses batch statistics, and a folded model has no BN left to fine-tune.
- A tiny $\sigma^2$ makes $s$ large, stretching a filter's weight range and making it harder to quantize. One scale per channel helps.
- Structural pruning looks for BN. `transfer_weights` in harness/prune.py reads `cv1.bn`, so prune before fusing.
- BN placed before a padded conv does not fold exactly, because the zero padding is never normalized.

## Going further

Nothing is missing to run it here: Ultralytics fuses during `predict`. What the repo lacks is a check. To verify, compare outputs before and after on one random input with `torch.allclose` at a small tolerance. The closest toy analogue is `collapse_linear`.
