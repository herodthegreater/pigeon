## The idea

Magnitude pruning deletes the weights with the smallest absolute value. But a weight's effect on the output is `|w_ij| · |x_j|`, the weight times the input it multiplies. A small weight on a loud input channel can matter more than a large weight on a quiet one. Wanda ("pruning by weights and activations") scores each weight by `|w_ij| · ||X_j||`, then zeroes the lowest-scoring weights *within each output row*. It needs one forward pass over calibration data. There is no weight update and no retraining.

## How it works

Paper orientation: `W` is (out, in). `X` holds calibration inputs, one row per token.

1. Run the calibration data and take each input channel's L2 norm across all tokens: `n_j = sqrt(Σ_t X[t, j]²)`.
2. Score every weight: `S_ij = |W_ij| · n_j`.
3. For each output row `i`, sort that row's scores and zero the lowest `k = round(sparsity · in)`.
4. Move to the next layer, feeding it outputs from the layers already pruned.

**Where the score comes from.** Optimal Brain Surgeon rates removing weight `j` by `w_j² / [H⁻¹]_jj`, where `H = XᵀX` (see SparseGPT). If you approximate `H` by its diagonal, `[H⁻¹]_jj = 1 / Σ_t x_tj²`. The saliency becomes `w_j² · n_j²`, whose square root is Wanda's score. So Wanda is OBS with a diagonal Hessian and without the compensating update.

**Why per output row.** Each output `y_i = Σ_j W_ij x_j` is its own sum. Comparing within a row keeps every output's sum equally intact. The paper found per-output comparison clearly better than per-layer for LLMs, though not for image classifiers.

**In a transformer**, every linear layer in every block is pruned (q, k, v, o, up, gate, down), but not the embeddings or the output head. Calibration is 128 sequences of 2048 tokens from C4. The paper prunes LLaMA to 50% sparsity with a modest perplexity increase, and the 2:4 variant (2 zeros in every 4 consecutive weights) runs faster on NVIDIA GPUs.

## The code, line by line

`wanda(Wt, X, sparsity)`:

- `score = np.abs(Wt) * np.sqrt((X ** 2).sum(0))[None, :]` is steps 1 and 2. `(X ** 2).sum(0)` sums over calibration inputs, giving one value per input channel. `[None, :]` broadcasts it across all output rows.
- `k = int(round(Wt.shape[1] * sparsity))` is the number to drop per row.
- `idx = np.argsort(score, 1)[:, :k]` sorts each row (axis 1) separately and takes its `k` lowest. This is the per-output comparison.
- `np.put_along_axis(out, idx, 0.0, 1)` zeroes exactly those positions. The surviving weights keep their original values.

The page's "magnitude, per row" baseline is the same function with `X` replaced by `np.ones_like(h)`. Every channel norm is then equal, and the score reduces to `|w|`. That makes it a clean control: same row-wise selection, no activations.

## What the simulations show

3x256 teacher (0.9139), all four layers pruned, 512 calibration inputs:

| fraction removed | magnitude, per row | Wanda |
|---|---|---|
| 0.5 | 0.8594 | 0.8555 |
| 0.7 | 0.6684 | 0.6377 |
| 0.9 | 0.1085 | 0.0872 |

Wanda is *not* better here. The toy network's inputs have no outlier channels, so activation norms add noise more than signal.

One property does show up. `inject_outliers` makes three channels 30 times louder and their weights 30 times smaller, without changing the function. Wanda's score is unchanged by this: `|w/30| · 30·n = |w| · n`. Running it directly on the outlier network gave 0.8555 at 50% and 0.6377 at 70%, identical to the table. Magnitude pruning, which now sees those weights as tiny, moved from 0.8594 to 0.8545 and from 0.6684 to 0.6601. With only 3 of 256 channels affected, the damage is small. In LLMs, emergent outlier features make channel norms differ by orders of magnitude across many layers, which is where the paper's gains come from.

```python
from lab import layerwise as lw
from lab.common import data, ev, teacher
X = data().train.X[:512]
m = lw.inject_outliers(teacher(), 1, 3, 30.0)
print(ev(lw.apply_layerwise(m, X, lambda W, h: lw.wanda(W, h, 0.5)))["acc"])
```

## When it breaks

- With no weight update, nothing absorbs the removed weights' contribution, so accuracy falls fast past 50%. SparseGPT's compensation holds up much better at 0.7.
- When channel norms don't track importance (here, ReLU features of similar scale), the activation factor reweights almost at random.
- It uses only the norm of each channel, not correlations between channels. Two redundant channels both look important.
- The calibration data sets `n_j`. A channel that is silent in calibration but active in deployment gets all its weights pruned.

## Try this

- **fraction removed** 0.5, 0.7, 0.9: watch the gap between magnitude and Wanda, then compare with SparseGPT at the same setting.
- **fraction removed** 0.3: look for both methods staying near the teacher. Low sparsity is easy for any criterion.

## Going further

Wanda's appeal is cost: one forward pass and no matrix inverse, so it is far faster than SparseGPT on large models. Follow-ups add a light weight update (for example, refitting the surviving weights by least squares) or non-uniform sparsity across layers (OWL, which gives layers with more outliers less pruning). To deploy the speed-up, use the 2:4 pattern that sparse tensor cores support. Unstructured 50% sparsity saves memory only with a sparse storage format.
