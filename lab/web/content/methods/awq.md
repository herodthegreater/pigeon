## The idea

In an LLM, a small share of input channels (0.1% to 1%) carry much larger activations than the rest. The weights that multiply those channels matter most, but RTN gives them the same coarse step as everything else. AWQ (activation-aware weight quantization) multiplies each such weight column by a factor $s > 1$ before rounding and divides the matching input by $s$. The function is unchanged, but the important weights get finer effective resolution. No weight update, no Hessian: one scale per input channel, chosen by a small search.

## How it works

Paper orientation: $y = Wx$, $W$ is (out, in), and $X$ holds calibration inputs.

1. Per input channel $j$, measure the average activation size: $s_{X,j} = \operatorname{mean}_{x \in X} \lvert x_j \rvert$.
2. For each $\alpha$ on a grid in $[0, 1]$, set $s = s_X^{\alpha}$. $\alpha = 0$ means no scaling, which is plain RTN. $\alpha = 1$ scales fully by activation size.
3. Quantize the scaled weights, $Q(W \diag(s))$, and feed them scaled-down inputs $\diag(s)^{-1} x$. Measure the output error $\lVert Q(W \diag(s)) \diag(s)^{-1} X - WX \rVert^2$ on calibration data.
4. Keep the $\alpha$ with the smallest error.
5. At deployment, fold $\diag(s)^{-1}$ into whatever produces $x$, so the scaling costs nothing at run time.

Steps 1 to 4 together are a one-parameter search:

$$
\begin{aligned}
s_{X,j} &= \operatorname{mean}_{x \in X} \lvert x_j \rvert, \qquad s = s_X^{\alpha}, \\
\alpha^* &= \argmin_{\alpha \in [0, 1]} \bigl\lVert Q\bigl(W \diag(s)\bigr) \diag(s)^{-1} X - W X \bigr\rVert^2 .
\end{aligned}
$$

**Why scaling helps.** RTN's error on $w \cdot x$ is up to $\Delta \cdot x / 2$, with $\Delta$ the step size. After scaling, $w \cdot s$ is rounded with step $\Delta'$ and multiplied by $x / s$, so the error is up to $\Delta' \cdot x / (2s)$:

$$
\begin{aligned}
\text{RTN:} \quad \bigl\lvert Q(w)\, x - w x \bigr\rvert &\le \frac{\Delta}{2} \lvert x \rvert, \\
\text{scaled:} \quad \Bigl\lvert Q(w s)\, \frac{x}{s} - w x \Bigr\rvert &\le \frac{\Delta'}{2} \cdot \frac{\lvert x \rvert}{s}.
\end{aligned}
$$

If scaling one column doesn't change the row's maximum, $\Delta' \approx \Delta$ and the error on that channel shrinks by $1/s$. If $s$ is too large, that column *becomes* the row maximum and $\Delta'$ grows for every other weight, which is why $\alpha$ is searched.

**In a transformer**, AWQ is applied to groups of linear layers that share one input: q, k and v share the attention LayerNorm output, and gate and up share the MLP LayerNorm output. Their $1/s$ is folded into the preceding LayerNorm or RMSNorm weight, or into the previous linear layer (v into o, up into down). Because AWQ only estimates per-channel averages from calibration text and never fits weights to it, the paper reports it is less sensitive to the calibration set than GPTQ.

## The code, line by line

`awq(Wt, X, bits, group, grid)`:

- `xmag = np.abs(X).mean(0) + 1e-8` is $s_X$ from step 1; the `1e-8` keeps always-zero channels from getting a zero scale.
- `ref = X @ Wt.T` is the full-precision output (bias left out; quantization doesn't touch it).
- `for a in np.linspace(0, 1, grid + 1)`: with `grid = 20`, that is 21 values of $\alpha$, including 0.
- `s = s / np.sqrt(s.max() * s.min())`, that is $s \leftarrow s / \sqrt{\max_j s_j \cdot \min_j s_j}$, centers the scales around 1 on a log scale. It comes from the official AWQ code, not the paper, and only decides how many columns move up versus down.
- `Wq = quant_rows(Wt * s[None, :], bits, group) / s[None, :]` is $Q(W \diag(s)) \diag(s)^{-1}$: it scales column $j$ by $s_j$, rounds with ordinary RTN, then divides by $s$ again. Dividing the weight by $s$ gives the same output as feeding $X / s$, so this is the deployed computation with the fold done inside the weight.
- `err = ((X @ Wq.T - ref) ** 2).mean()` and the `if err < best_err` comparison keep the best $\alpha$. Because $\alpha = 0$ (no scaling) is on the grid, AWQ's per-layer error can never be worse than RTN's.

**Simplifications.** The paper's method also searches a *clipping* threshold per group, shrinking the max range to cut rounding error, and that step is not implemented. The paper uses asymmetric quantization with a zero point and groups of 128; here it is symmetric with the page's group choice. The paper measures error at the whole module's output (for example the attention block); here it is one linear layer.

## What the simulations show

3x256 teacher (0.9150). The three hidden layers are quantized, including the first (12 raw inputs); the output layer stays at full precision, like `lm_head` in the papers. "2-bit" is ternary, $\{-s, 0, s\}$.

| setting | RTN | AWQ | GPTQ |
|---|---|---|---|
| 4-bit, per row | 0.9087 | 0.9102 | 0.9149 |
| 3-bit, per row | 0.8971 | 0.8971 | 0.9093 |
| 3-bit, scale per 16 | 0.9007 | 0.9035 | 0.9115 |
| 2-bit, per row | 0.4796 | 0.4711 | 0.8314 |

AWQ's gains here are small. At 3 bits per row it ties RTN exactly, 1.8 points below the teacher, while GPTQ cuts the loss to 0.6. At 4 bits, and at 3 bits with a scale per 16, it gains a little (0.9102 against 0.9087, 0.9035 against 0.9007). At 2 bits the two scores move apart: accuracy falls a little (0.4796 to 0.4711) while the bird class's average precision rises (0.7380 to 0.7534). Each layer's own error is never worse than RTN's (the search includes $s = 1$, which is RTN), but that objective does not exactly track final accuracy.

Calling `lw.rtn` and `lw.awq` directly at 3 bits on each layer's teacher inputs (512 calibration inputs; error is $\lVert hW - hW_q \rVert^2 / \lVert hW \rVert^2$), AWQ matched RTN's relative output error to four decimals on the first layer (0.0335) and barely moved the second (0.0313 to 0.0309). It improved the third layer from 0.0388 to 0.0355. The toy network has no LLM-style outlier channels, so there is little to protect; that is what this world can't reproduce.

## When it breaks

- Without activation outliers, $s_X$ is nearly flat and AWQ reduces to RTN.
- A heavily scaled column can become the maximum of its row or group, which coarsens the step for every other weight there.
- $s$ must fold into the previous operation, so layers that share an input must share one $s$.

## Try this

- **bits** 3, **scale per** row: compare RTN, AWQ and (on the GPTQ page) GPTQ. Look for AWQ tying RTN while GPTQ pulls ahead.
- **bits** 4, or **scale per** 16 at 3 bits: look for AWQ slightly above RTN, by much less than GPTQ gains.
- **bits** 2: look for AWQ's accuracy slipping while its bird AP rises. The per-layer objective is not the end-to-end metric.

## Going further

AWQ is widely used for 4-bit LLM inference (in vLLM, TensorRT-LLM and similar engines) because it produces plain group-quantized weights that standard INT4 kernels can run. For activation quantization, the same "move scale between activation and weight" trick appears in SmoothQuant, which picks $s$ in closed form instead of by search.
