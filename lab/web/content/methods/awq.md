## The idea

In an LLM, a small share of input channels (0.1% to 1%) carry much larger activations than the rest. The weights that multiply those channels matter most, but RTN gives them the same coarse step as everything else. AWQ (activation-aware weight quantization) multiplies each such weight column by a factor `s > 1` before rounding and divides the matching input by `s`. The function is unchanged, but the important weights get finer effective resolution. No weight update, no Hessian: one scale per input channel, chosen by a small search.

## How it works

Paper orientation: `y = W x`, `W` is (out, in), and `X` holds calibration inputs.

1. Per input channel `j`, measure the average activation size: `s_X[j] = mean |X[:, j]|`.
2. For each `α` on a grid in [0, 1], set `s = s_X^α`. `α = 0` means no scaling, which is plain RTN. `α = 1` scales fully by activation size.
3. Quantize the scaled weights, `Q(W · diag(s))`, and feed them scaled-down inputs `diag(s)⁻¹ x`. Measure the output error `|Q(W·diag(s)) · diag(s)⁻¹ X - W X|²` on calibration data.
4. Keep the `α` with the smallest error.
5. At deployment, fold `diag(s)⁻¹` into whatever produces `x`, so the scaling costs nothing at run time.

**Why scaling helps.** RTN's error on `w·x` is up to `Δ·x/2`, with `Δ` the step size. After scaling, `w·s` is rounded with step `Δ'` and multiplied by `x/s`, so the error is up to `Δ'·x/(2s)`. If scaling one column doesn't change the row's maximum, `Δ' ≈ Δ` and the error on that channel shrinks by `1/s`. If `s` is too large, that column *becomes* the row maximum and `Δ'` grows for every other weight, which is why `α` is searched.

**In a transformer**, AWQ is applied to groups of linear layers that share one input: q, k and v share the attention LayerNorm output, and gate and up share the MLP LayerNorm output. Their `1/s` is folded into the preceding LayerNorm or RMSNorm weight, or into the previous linear layer (v into o, up into down). Because AWQ only estimates per-channel averages from calibration text and never fits weights to it, the paper reports it is less sensitive to the calibration set than GPTQ.

## The code, line by line

`awq(Wt, X, bits, group, grid)`:

- `xmag = np.abs(X).mean(0) + 1e-8` is `s_X` from step 1; the `1e-8` keeps always-zero channels from getting a zero scale.
- `ref = X @ Wt.T` is the full-precision output (bias left out; quantization doesn't touch it).
- `for a in np.linspace(0, 1, grid + 1)`: with `grid = 20`, that is 21 values of `α`, including 0.
- `s = s / np.sqrt(s.max() * s.min())` centers the scales around 1 on a log scale. It comes from the official AWQ code, not the paper, and only decides how many columns move up versus down.
- `Wq = quant_rows(Wt * s[None, :], bits, group) / s[None, :]` scales column `j` by `s[j]`, rounds with ordinary RTN, then divides by `s` again. Dividing the weight by `s` gives the same output as feeding `X / s`, so this is the deployed computation with the fold done inside the weight.
- `err = ((X @ Wq.T - ref) ** 2).mean()` and the `if err < best_err` comparison keep the best `α`. Because `α = 0` (no scaling) is on the grid, AWQ's per-layer error can never be worse than RTN's.

**Simplifications.** The paper's method also searches a *clipping* threshold per group, shrinking the max range to cut rounding error, and that step is not implemented. The paper uses asymmetric quantization with a zero point and groups of 128; here it is symmetric with the page's group choice. The paper measures error at the whole module's output (for example the attention block); here it is one linear layer.

## What the simulations show

3x256 teacher (0.9139). The three hidden layers are quantized, including the first (12 raw inputs); the output layer stays at full precision, like `lm_head` in the papers. "2-bit" is ternary, `{-s, 0, s}`.

| setting | RTN | AWQ | GPTQ |
|---|---|---|---|
| 4-bit, per row | 0.9103 | 0.9096 | 0.9131 |
| 3-bit, per row | 0.8922 | 0.8958 | 0.9079 |
| 3-bit, scale per 16 | 0.9010 | 0.9002 | 0.9110 |
| 2-bit, per row | 0.4686 | 0.4987 | 0.8207 |

AWQ's gains here are small: at 3 bits it trims RTN's loss from 2.2 to 1.8 points, while GPTQ cuts it to 0.6. At 4 bits, and at 3 bits with a scale per 16, it is slightly *worse* than RTN end to end. Each layer's own error is no worse than RTN's, but that objective does not exactly track final accuracy. At 2 bits, accuracy rises a little (0.4686 to 0.4987) while the bird class's average precision dips (0.7947 to 0.7834).

Calling `lw.rtn` and `lw.awq` directly at 3 bits on each layer's teacher inputs, AWQ matched RTN's relative output error to four decimals on the first two layers (0.0333 and 0.0300). It improved the third layer from 0.0397 to 0.0369.  The toy network has no LLM-style outlier channels, so there is little to protect; that is what this world can't reproduce.

## When it breaks

- Without activation outliers, `s_X` is nearly flat and AWQ reduces to RTN.
- A heavily scaled column can become the maximum of its row or group, which coarsens the step for every other weight there.
- `s` must fold into the previous operation, so layers that share an input must share one `s`.

## Try this

- **bits** 3, **scale per** row: compare RTN, AWQ and (on the GPTQ page) GPTQ. Look for AWQ landing between the two.
- **bits** 4, or **scale per** 16 at 3 bits: look for AWQ slightly below RTN. The per-layer objective is not the end-to-end metric.
- **bits** 2: look for AWQ's small accuracy gain next to GPTQ's large one, and the bird AP moving the other way.

## Going further

AWQ is widely used for 4-bit LLM inference (in vLLM, TensorRT-LLM and similar engines) because it produces plain group-quantized weights that standard INT4 kernels can run. For activation quantization, the same "move scale between activation and weight" trick appears in SmoothQuant, which picks `s` in closed form instead of by search.
