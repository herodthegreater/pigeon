## The idea

To run a matrix multiply on 8-bit integer hardware, both the weights and the *activations* (the inputs flowing into the layer) must be 8-bit. LLM activations are hard: a few fixed channels are up to 100 times larger than the rest, so a scale per tensor, or even per token, spends almost all its levels on them. Weights are easy. SmoothQuant moves the difficulty from activations to weights with an exact per-channel rescaling, $Y = (X \diag(s)^{-1})(\diag(s) W)$, so both become quantizable.

## How it works

SmoothQuant writes $Y = XW$ with $X$ (tokens $\times$ in) and $W$ (in $\times$ out), the same orientation this repo stores.

1. On calibration data, record each input channel's maximum activation $\max \lvert X_j \rvert$ and maximum weight $\max \lvert W_j \rvert$ (row $j$ of $W$).
2. Set the scale of channel $j$; the *migration strength* $\alpha$ sets how much difficulty moves:

$$
s_j = \frac{(\max \lvert X_j \rvert)^{\alpha}}{(\max \lvert W_j \rvert)^{1-\alpha}}
$$

At $\alpha = 0.5$, both sides end up with the same per-channel maximum $\sqrt{\max \lvert X_j \rvert \cdot \max \lvert W_j \rvert}$ (derived below).

3. Divide input channel $j$ by $s_j$ and multiply weight row $j$ by $s_j$. The product is unchanged:

$$
\begin{aligned}
Y = XW &= \bigl(X \diag(s)^{-1}\bigr)\bigl(\diag(s)\, W\bigr) \\
&= \hat X \hat W
\end{aligned}
$$

4. Fold $1/s$ into whatever produced $X$, offline, so no extra work happens at run time.
5. Quantize weights per output channel and activations per tensor (or per token) to 8 bits.

With $\hat X = X \diag(s)^{-1}$ and $\hat W = \diag(s) W$, the per-channel maxima after smoothing are

$$
\begin{aligned}
\max \lvert \hat X_j \rvert &= \frac{\max \lvert X_j \rvert}{s_j} \\
&= (\max \lvert X_j \rvert)^{1-\alpha} \, (\max \lvert W_j \rvert)^{1-\alpha}, \\
\max \lvert \hat W_j \rvert &= s_j \max \lvert W_j \rvert \\
&= (\max \lvert X_j \rvert)^{\alpha} \, (\max \lvert W_j \rvert)^{\alpha},
\end{aligned}
$$

which are equal when $\alpha = 1/2$.

**Why per-channel activation scales are not an option.** An integer matmul sums over input channels, so a per-input-channel scale sits inside the sum and can't be pulled out afterwards. Per-token and per-output-channel scales can.

**In a transformer**, the outlier channels appear in the LayerNorm outputs that feed the attention QKV projection and the first MLP layer. $1/s$ is folded into that LayerNorm's gain and bias. The paper uses $\alpha = 0.5$ for most models, more for harsher outliers.

## The code, line by line

`inject_outliers(net, layer, k, factor)` builds the problem on purpose. It multiplies `k` neurons of hidden layer `layer` (weights and bias) by `factor` and divides the next layer's matching weight rows by `factor`. Because $\relu(cz) = c \relu(z)$ for $c > 0$, the function is unchanged, but in the page's run three channels become **outlier size** times louder.

`smoothquant(net, X, alpha)`:

- `_, hid, _ = net.forward(X)` records the calibration activations once. `hid[i - 1]` is layer `i`'s input. One pass is enough: rescaling layer `i`'s inputs never changes its outputs, which are the next layer's inputs.
- `if net.res[i - 1] or net.res[i]: continue` skips residual layers. Rescaling a residual stream would also rescale the skip path.
- `a = np.abs(hid[i - 1]).max(0)` and `w = np.abs(net.W[i]).max(1) + 1e-8` are $\max \lvert X_j \rvert$ and $\max \lvert W_j \rvert$. `.max(1)` on the (in, out) array runs over a whole input row.
- `s = a ** alpha / w ** (1 - alpha)` is step 2, $s_j = a_j^{\alpha} / w_j^{1-\alpha}$.
- `silent = a < 1e-3 * a.max()` finds ReLU neurons that never fire on calibration data. The formula would give them $s \approx 0$, multiplying their incoming weights by hundreds, so any test input that switches one on would produce a giant activation. `np.where(silent, 1.0, np.clip(s, 1e-3, 1e3))` leaves them unscaled. On the 512 calibration inputs, 1 of layer 2's 256 inputs counts as silent (2 once injected outliers raise the maximum), and 9 of the output layer's.
- `net.W[i] = net.W[i] * s[:, None]` scales the weight rows. `net.W[i - 1] / s[None, :]` and `net.b[i - 1] / s` fold $1/s$ into the previous layer, which is legal because ReLU is positively homogeneous.

`quant_act(h, bits)` is *dynamic per-token* quantization: `scale = np.abs(h).max(1, keepdims=True) / qmax`, one scale per row (per example here, per token in an LLM), computed on the fly. One huge channel still sets its row's scale, which is exactly the problem. `forward_wa(net, X, abits)` quantizes the input of every layer after the first, then runs the layer.

With **weight bits** below 16, `run_smoothquant` passes both models through `apply_layerwise(..., rtn, skip_last=True)`: per-output-row RTN *after* smoothing, so the difficulty moved into the weights is charged. The first layer is quantized; the output layer stays float, like `lm_head`.

## What the simulations show

3x256 teacher with 3 outlier channels injected into layer 2's input, at outlier size 30. Weights are 8-bit (W8), and the float model scores 0.9150.

| activation bits | plain, W8 | SmoothQuant $\alpha$ 0.5, W8 |
|---|---|---|
| 16 | 0.9146 | 0.9146 |
| 8 | 0.9137 | 0.9154 |
| 6 | 0.9049 | 0.9155 |
| 5 | 0.8517 | 0.9152 |
| 4 | 0.5661 | 0.9079 |

At outlier size 100, plain falls to 0.6850 at A6 and 0.2157 at A4; SmoothQuant gives 0.9155 and 0.9079, identical to size 30, because $s$ absorbs the injected factor exactly. Without injected outliers, plain A4 already gives 0.9096 and SmoothQuant 0.9089: there was nothing to fix.

The weight side costs little here. At W4A6, plain gives 0.8963 and SmoothQuant 0.9112. Even $\alpha = 1$, which moves all of the activation range into the weights, gives 0.9101 at W4A6. Across $\alpha$ = 0, 0.5, 0.8 and 1 at W8A4, SmoothQuant gives 0.9035, 0.9079, 0.9090 and 0.9099. So this toy does not show the paper's trade-off, where too large an $\alpha$ makes the weights hard to quantize, probably because its weights have no outliers of their own for the migrated range to collide with.

## When it breaks

- Channels that are silent during calibration have no measured range. Scaling them by the formula blows up their weights for unseen inputs; the code leaves them at $s = 1$.
- Large $\alpha$ pushes outliers into the weights, which in real LLMs eventually costs weight accuracy (barely, here).
- $s$ is static. A channel that is an outlier on only some tokens is not fully tamed by it.
- The fold needs a positively homogeneous op (ReLU, or LayerNorm's affine gain) right before the layer. The paper smooths only layers fed by a LayerNorm, where the fold is free.

## Try this

- **inject outlier channels** on, **weight bits** 8, **activation bits** 4: compare the plain row with SmoothQuant.
- Raise **outlier size** to 100 at **activation bits** 6: look at how far plain falls while SmoothQuant doesn't move.
- **weight bits** 4, then **migration strength** 0, 0.5 and 1: look for whether moving range into the weights costs anything at W4.
- Turn **inject outlier channels** off: look for SmoothQuant having nothing left to fix.

## Going further

SmoothQuant enabled W8A8 inference of models up to 530B parameters with negligible accuracy loss. Rotation methods (QuaRot, SpinQuant) push the same idea further: an orthogonal rotation spreads every outlier across all channels, which makes 4-bit activations practical.
