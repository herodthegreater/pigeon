## The idea

In an LLM, a small share of input channels (0.1% to 1%) carry much larger activations than the rest. The weights that multiply those channels matter most, but RTN gives them the same coarse step as everything else. AWQ (activation-aware weight quantization) multiplies each such weight column by a factor `s > 1` before rounding and divides the matching input by `s`. The function is unchanged, but the important weights get finer effective resolution. No weight update, no Hessian: one scale per input channel, chosen by a small search.

## How it works

Paper orientation: `y = W x`, `W` is (out, in), and `X` holds calibration inputs.

1. Per input channel `j`, measure the average activation size: `s_X[j] = mean |X[:, j]|`.
2. For each `α` on a grid in [0, 1], set `s = s_X^α`. `α = 0` means no scaling, which is plain RTN. `α = 1` scales fully by activation size.
3. Quantize the scaled weights, `Q(W · diag(s))`, and feed them scaled-down inputs `diag(s)⁻¹ x`. Measure the output error `|Q(W·diag(s)) · diag(s)⁻¹ X - W X|²` on calibration data.
4. Keep the `α` with the smallest error.
5. At deployment, fold `diag(s)⁻¹` into whatever produces `x`, so the scaling costs nothing at run time.

**Why scaling helps.** Take one weight `w` with input `x`. RTN's error on `w·x` is about `Δ·x/2` at worst, where `Δ` is the step size. After scaling, the weight is `w·s`, rounded with step `Δ'`, then multiplied by `x/s`. The error becomes about `Δ'·x/(2s)`. If scaling one column doesn't change the row's maximum, `Δ' ≈ Δ` and the error on that channel shrinks by `1/s`. If `s` is too large, that column *becomes* the row maximum and `Δ'` grows for every other weight. That trade-off is why `α` is searched rather than set to 1.

**In a transformer**, AWQ is applied to groups of linear layers that share one input: q, k and v share the attention LayerNorm output, and gate and up share the MLP LayerNorm output. Their `1/s` is folded into the preceding LayerNorm or RMSNorm weight, or into the previous linear layer (v into o, up into down). Because AWQ only estimates per-channel averages from calibration text and never fits weights to it, the paper reports it is less sensitive to the calibration set than GPTQ.

## The code, line by line

`awq(Wt, X, bits, group, grid)`:

- `xmag = np.abs(X).mean(0) + 1e-8` is `s_X` from step 1. The `1e-8` keeps channels that are always zero from producing a zero scale.
- `ref = X @ Wt.T` is the full-precision output (bias left out; quantization doesn't touch it).
- `for a in np.linspace(0, 1, grid + 1)`: with `grid = 20`, that is 21 values of `α`, including 0.
- `s = s / np.sqrt(s.max() * s.min())` centers the scales around 1 on a log scale. It comes from the official AWQ code, not the paper, and only decides how many columns move up versus down.
- `Wq = quant_rows(Wt * s[None, :], bits, group) / s[None, :]` scales column `j` by `s[j]`, rounds with ordinary RTN, then divides by `s` again. Dividing the weight by `s` gives the same output as feeding `X / s`, so this is the deployed computation with the fold done inside the weight.
- `err = ((X @ Wq.T - ref) ** 2).mean()` and the `if err < best_err` comparison keep the best `α`. Because `α = 0` (no scaling) is on the grid, AWQ's per-layer error can never be worse than RTN's.

**Simplifications.** The paper's method also searches a *clipping* threshold per group, shrinking the max range to cut rounding error, and that step is not implemented. The paper uses asymmetric quantization with a zero point and groups of 128; here it is symmetric with the page's group choice. The real error is measured at the output of the whole module (for example the attention block); here it is one linear layer.

## What the simulations show

3x256 teacher (0.9139), every layer quantized:

| setting | RTN | AWQ | GPTQ |
|---|---|---|---|
| 4-bit, per row | 0.9078 | 0.9062 | 0.9130 |
| 3-bit, per row | 0.8669 | 0.8851 | 0.9029 |
| 3-bit, scale per 16 | 0.8947 | 0.8976 | 0.9101 |
| 2-bit, per row | 0.2483 | 0.2695 | 0.7702 |

AWQ recovers almost 40% of RTN's 3-bit loss (4.7 points down to 2.9). At 4 bits it is slightly *worse* than RTN end to end. Each layer's own error is no worse than RTN's, but that objective does not exactly track final accuracy. At 2 bits, accuracy rises a little (0.2483 to 0.2695) while the bird class's average precision falls from 0.6268 to 0.3872.

Calling `lw.rtn` and `lw.awq` directly at 3 bits on each layer's teacher inputs, AWQ matched RTN's relative output error to four decimals on the first two layers (0.0333 and 0.0300). It improved the third layer from 0.0397 to 0.0369 and the output layer from 0.0341 to 0.0221. The toy network has no LLM-style outlier channels, so there is little to protect; that is what this world can't reproduce.

## When it breaks

- Without activation outliers, `s_X` is nearly flat and AWQ reduces to RTN.
- A heavily scaled column can become the maximum of its row or group, which coarsens the step for every other weight there.
- Without the clipping search, one large weight still sets each row's step size.
- `s` must fold into the previous operation, so layers that share an input must share one `s`.

## Try this

- **bits** 3, **scale per** row: compare RTN, AWQ and (on the GPTQ page) GPTQ. Look for AWQ landing between the two.
- **bits** 4: look for AWQ slightly below RTN. The per-layer objective is not the end-to-end metric.
- **scale per** 16 at 3 bits: look for how much of AWQ's gain disappears once groups already isolate large weights.

## Going further

AWQ is widely used for 4-bit LLM inference (in vLLM, TensorRT-LLM and similar engines) because it produces plain group-quantized weights that standard INT4 kernels can run. It combines with GPTQ: apply the AWQ scales first, then run GPTQ on the scaled weights. For activation quantization, the same "move scale between activation and weight" trick appears in SmoothQuant, which picks `s` in closed form instead of by search.
