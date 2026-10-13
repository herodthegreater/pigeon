## The idea

Magnitude pruning deletes the weights with the smallest absolute value. But a weight's effect on the output is $\lvert w_{ij} \rvert \cdot \lvert x_j \rvert$, the weight times the input it multiplies. A small weight on a loud input channel can matter more than a large weight on a quiet one. Wanda ("pruning by weights and activations") scores each weight by $\lvert w_{ij} \rvert \cdot \lVert X_j \rVert_2$, then zeroes the lowest-scoring weights *within each output row*. It needs one forward pass over calibration data. There is no weight update and no retraining.

## How it works

Paper orientation: $W$ is (out, in). $X$ holds calibration inputs, one row per token.

1. Run the calibration data and take each input channel's L2 norm across all tokens:

$$
n_j = \lVert X_{:,j} \rVert_2 = \sqrt{\sum_t X_{tj}^2}
$$

2. Score every weight:

$$
S_{ij} = \lvert W_{ij} \rvert \cdot n_j
$$

3. For each output row $i$, sort that row's scores and zero the lowest $k = \round(\text{sparsity} \cdot d_{\text{in}})$.
4. Move to the next layer, feeding it outputs from the layers already pruned.

**Where the score comes from.** Optimal Brain Surgeon rates removing weight $j$ by $w_j^2 / [H^{-1}]_{jj}$, where $H = X^\top X$ (see SparseGPT). If you approximate $H$ by its diagonal, $[H^{-1}]_{jj} = 1 / \sum_t x_{tj}^2$. The saliency becomes $w_j^2 \cdot n_j^2$, whose square root is Wanda's score:

$$
\begin{aligned}
H &\approx \diag(H) \\
[H^{-1}]_{jj} &\approx \frac{1}{\sum_t x_{tj}^2} = \frac{1}{n_j^2} \\
\frac{w_j^2}{[H^{-1}]_{jj}} &\approx w_j^2\, n_j^2 \\
&= \bigl( \lvert w_j \rvert \cdot n_j \bigr)^2
\end{aligned}
$$

So Wanda is OBS with a diagonal Hessian and without the compensating update.

**Why per output row.** Each output $y_i = \sum_j W_{ij} x_j$ is its own sum. Comparing within a row keeps every output's sum equally intact. The paper found per-output comparison clearly better than per-layer for LLMs, though not for image classifiers.

**In a transformer**, every linear layer in every block is pruned (q, k, v, o, up, gate, down), but not the embeddings or the output head. Calibration is 128 sequences of 2048 tokens from C4. The paper prunes LLaMA to 50% sparsity with a modest perplexity increase, and the 2:4 variant (2 zeros in every 4 consecutive weights) runs faster on NVIDIA GPUs.

## The code, line by line

`wanda(Wt, X, sparsity)`:

- `score = np.abs(Wt) * np.sqrt((X ** 2).sum(0))[None, :]` is steps 1 and 2. `(X ** 2).sum(0)` sums over calibration inputs, giving one value per input channel. `[None, :]` broadcasts it across all output rows.
- `k = int(round(Wt.shape[1] * sparsity))` is the number to drop per row.
- `idx = np.argsort(score, 1)[:, :k]` sorts each row (axis 1) separately and takes its `k` lowest. This is the per-output comparison.
- `np.put_along_axis(out, idx, 0.0, 1)` zeroes exactly those positions. The surviving weights keep their original values.

The page's "magnitude, per row" baseline is the same function with `X` replaced by `np.ones_like(h)`. Every channel norm is then equal, and the score reduces to $\lvert w \rvert$. That makes it a clean control: same row-wise selection, no activations. Both run through `apply_layerwise(..., skip_last=True)`: the hidden layers are pruned, including the first one, which reads the 12 raw inputs, and the output layer is left dense, as papers leave `lm_head` alone.

**outlier channels injected** calls `inject_outliers(t, 1, k, 30.0)` before pruning. It makes `k` of layer 2's 256 input channels 30 times louder and their weights 30 times smaller, without changing the function.

## What the simulations show

3x256 teacher (0.9150), 512 calibration inputs, no injected outliers:

| fraction removed | magnitude, per row | Wanda |
|---|---|---|
| 0.3 | 0.9084 | 0.9090 |
| 0.5 | 0.8899 | 0.8838 |
| 0.7 | 0.7696 | 0.7187 |
| 0.9 | 0.1302 | 0.1152 |

On the plain toy network Wanda is *not* better: it ties magnitude at 0.3 and trails it from 0.5 up. Its inputs have no outlier channels, so activation norms add noise more than signal.

Injected outliers show what Wanda is for. Wanda's score is unchanged by the rescaling: $\lvert w / 30 \rvert \cdot 30\, n = \lvert w \rvert \cdot n$. Magnitude pruning instead sees the rescaled weights as tiny and removes them. At 60% removed:

| outlier channels injected | magnitude, per row | Wanda |
|---|---|---|
| 0 | 0.8528 | 0.8407 |
| 8 | 0.8477 | 0.8407 |
| 16 | 0.8353 | 0.8407 |
| 32 | 0.8083 | 0.8407 |
| 64 | 0.7781 | 0.8407 |
| 128 | 0.6340 | 0.8407 |

Wanda is identical in every row; magnitude crosses below it between 8 and 16 outlier channels. In LLMs, emergent outlier features make channel norms differ by orders of magnitude in every block, which is where the paper's gains come from.

## When it breaks

- With no weight update, nothing absorbs the removed weights' contribution, so accuracy falls fast past 50%. SparseGPT's compensation holds up much better at 0.7.
- When channel norms don't track importance (here, ReLU features of similar scale), the activation factor reweights almost at random.
- It uses only the norm of each channel, not correlations between channels. Two redundant channels both look important.
- The calibration data sets $n_j$. A channel that is silent in calibration but active in deployment gets all its weights pruned.

## Try this

- **fraction removed** 0.6, then **outlier channels injected** 0, 16, 64, 128: look for the Wanda row not moving at all while magnitude pruning falls.
- **fraction removed** 0.5, 0.7, 0.9 with no outliers: watch the gap between magnitude and Wanda, then compare with SparseGPT at the same setting.
- **fraction removed** 0.3: look for both methods staying near the teacher. Low sparsity is easy for any criterion.

## Going further

Wanda's appeal is cost: one forward pass and no matrix inverse, so it is far faster than SparseGPT on large models. Follow-ups add a light weight update (for example, refitting the surviving weights by least squares) or non-uniform sparsity across layers (OWL, which gives layers with more outliers less pruning). To deploy the speed-up, use the 2:4 pattern that sparse tensor cores support. Unstructured 50% sparsity saves memory only with a sparse storage format.
