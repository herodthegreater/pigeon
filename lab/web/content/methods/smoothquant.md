## The idea

To run a matrix multiply on 8-bit integer hardware, both the weights and the *activations* (the inputs flowing into the layer) must be 8-bit. LLM activations are hard to quantize: a few fixed channels are up to 100 times larger than the rest. One scale for the whole tensor then wastes almost all the levels on those channels. Weights, by contrast, are easy. SmoothQuant moves the difficulty from activations to weights with an exact per-channel rescaling, `y = (x / s) · (s · W)`, so both become quantizable.

## How it works

Paper orientation for SmoothQuant: `Y = X W` with `X` (tokens x in) and `W` (in x out). That matches how this repo stores weights.

1. Run calibration data and record each input channel's maximum activation `max|X_j|`, and each input channel's maximum weight `max|W_j|` (row `j` of `W`).
2. Set `s_j = max|X_j|^α / max|W_j|^(1-α)`. The *migration strength* `α` sets how much difficulty moves. At `α = 0.5`, both sides end up with the same per-channel maximum `sqrt(max|X_j| · max|W_j|)`.
3. Divide input channel `j` by `s_j` and multiply weight row `j` by `s_j`. The product is unchanged.
4. Fold `1/s` into whatever produced `X`, offline, so no extra work happens at run time.
5. Quantize weights per output channel and activations per tensor (or per token) to 8 bits.

**Why per-channel activation scales are not an option.** An integer matmul sums over input channels. A per-input-channel activation scale sits inside that sum and can't be pulled out afterwards. Per-token and per-output-channel scales can. SmoothQuant makes per-tensor or per-token scales good enough.

**In a transformer**, the outlier channels appear in the LayerNorm outputs that feed the attention QKV projection and the first MLP layer. `1/s` is folded into that LayerNorm's gain and bias. The paper uses `α = 0.5` for most models and a larger `α` for models with harsher outliers.

## The code, line by line

`inject_outliers(net, layer, k, factor)` builds the problem on purpose. It multiplies `k` neurons of hidden layer `layer` (weights and bias) by `factor` and divides the next layer's matching weight rows by `factor`. Because `relu(c·z) = c·relu(z)` for `c > 0`, the function is unchanged, but in the page's run three channels become **outlier size** times louder.

`smoothquant(net, X, alpha)`:

- `_, hid, _ = net.forward(X)` records the calibration activations. `hid[i - 1]` is layer `i`'s input.
- `if net.res[i - 1] or net.res[i]: continue` skips residual layers. Rescaling a residual stream would also rescale the skip path.
- `a = np.abs(hid[i - 1]).max(0) + 1e-5` and `w = np.abs(net.W[i]).max(1) + 1e-5` are `max|X_j|` and `max|W_j|`. `.max(1)` on the (in, out) array runs over a whole input row.
- `s = (a ** alpha / w ** (1 - alpha))` is step 2. `np.clip(s, 1e-3, 1e3)` bounds it.
- `net.W[i] = net.W[i] * s[:, None]` scales the weight rows. `net.W[i - 1] / s[None, :]` and `net.b[i - 1] / s` fold `1/s` into the previous layer, which is legal because ReLU is positively homogeneous.

`quant_act(h, bits)` is per-tensor symmetric quantization: one scale, `max|h| / qmax`, for the whole array. `forward_wa(net, X, abits)` quantizes the input of every layer after the first, then runs the layer.

**What is not simulated.** `forward_wa` leaves the weights in full precision. The table measures only the activation side; the extra difficulty pushed into the weights is never charged. Also, `forward_wa` is called on all 10,000 test inputs at once, so one activation scale is shared by the whole test set.

## What the simulations show

3x256 teacher with 3 outlier channels injected into layer 2's input (outlier size 30). Full precision is 0.9139 throughout.

| activation bits | plain | SmoothQuant, α 0.5 |
|---|---|---|
| 8 | 0.8997 | 0.9133 |
| 6 | 0.5449 | 0.9121 |
| 5 | 0.0931 | 0.0628 |
| 4 | 0.0777 | 0.0540 |

At outlier size 100, plain 8-bit drops to 0.7182 and plain 6-bit to 0.0743. SmoothQuant at 6 bits gives 0.9128.

The collapse at 4-5 bits is not just "too few bits". Without injected outliers, plain 4-bit activations give 0.8786, but SmoothQuant at `α = 0.5` gives 0.0540. Tracing it: one neuron feeding layer 2 never fires on the 512 calibration inputs, so its `a` is `1e-5` and its `s` is about 0.006, and its incoming weights are multiplied by about 170. With 4-bit noise upstream it fires on a dozen test inputs, and layer 2's input peaks at 96.3 (15.7 without smoothing). Because one scale serves all 10,000 test inputs, the next layer's inputs round to almost nothing (99.9th percentile 0.048). Quantizing each test input with its own scale, the same smoothed model scores 0.909 on the first 2,000 test inputs. `α = 0` ignores activations, so it avoids the problem: 0.7961 at 4 bits. `α = 1` triggers it even at 6 bits (0.0661).

## When it breaks

- Channels that are silent during calibration get tiny `s` and huge weights. Floor `max|X_j|` (for example at a fraction of the mean) before dividing.
- Large `α` pushes outliers into the weights. Real deployments then lose weight-quantization accuracy, which this page does not measure.
- Per-tensor activation scales computed over a large batch let one bad input ruin every other input's resolution.
- The fold needs a positively homogeneous op (ReLU, or LayerNorm's affine gain) right before the layer. The paper smooths only layers fed by a LayerNorm, where the fold is free.

## Try this

- **inject outlier channels** on, **activation bits** 6: compare the plain row with SmoothQuant.
- Raise **outlier size** to 100: look at how far plain 8-bit falls.
- Set **migration strength** to 0, then 1, at 4 bits: look for `α = 0` holding up and `α = 1` collapsing, and connect it to the silent-channel trace above.
- Turn **inject outlier channels** off at 4 bits: look for SmoothQuant doing harm on a network that had no outliers.

## Going further

SmoothQuant enables W8A8 inference of models up to 530B parameters with near-zero accuracy loss, roughly halving memory and speeding up matmuls. Rotation methods (QuaRot, SpinQuant) push the same idea further: an orthogonal rotation spreads every outlier across all channels, which makes 4-bit activations practical.
