## The idea

Pruning is quantization to a single level, zero. SparseGPT therefore reuses GPTQ's machinery. Walk through the input columns in order; when a weight is set to zero, adjust the weights to its right so the layer's output on calibration data changes as little as possible. The new part is choosing *which* weights to zero. SparseGPT decides one block of columns at a time, using the Optimal Brain Surgeon (OBS) saliency computed from the already-updated weights.

## How it works

Notation as on the GPTQ page: one output row $w$, calibration inputs $X$ ($n \times d$), Hessian $H = 2 X^\top X / n$, and $U$, the upper Cholesky factor of $H^{-1}$.

1. **Cost of removing a weight.** OBS says forcing coordinate $j$ from $w_j$ to 0 (the minimum-error forced change, derived on the GPTQ page with $e = w_j$) raises the error by $w_j^2 / (2 [H_F^{-1}]_{jj})$, after the other weights are optimally adjusted. Here $F$ is the set of columns not yet processed. Since $[H_F^{-1}]_{jj} = U_{jj}^2$, the *saliency* is $w_j^2 / U_{jj}^2$.
2. **Choose the mask a block at a time.** For columns $b_0, \dots, b_0 + B - 1$, compute the saliency of every weight in the block from the *current* weights, which already include the compensation from earlier blocks. Mark the lowest-saliency fraction for removal.
3. **Prune and compensate, column by column.** For $j$ in the block: set the marked weights to zero, with $e = w_j$ where pruned and 0 where kept. Then apply the GPTQ update $W_{:,\,j+1:} \leftarrow W_{:,\,j+1:} - (e / U_{jj}) \otimes U_{j,\,j+1:}$.
4. Repeat for the next block, then the next layer, which is fed outputs of the already-pruned layers.

In symbols, step 1's cost and saliency, and step 3's update for all rows at once ($e$ is the column of per-row errors):

$$
\begin{aligned}
\Delta E_j &= \frac{w_j^2}{2\, [H_F^{-1}]_{jj}} = \frac{w_j^2}{2\, U_{jj}^2}, \qquad \text{saliency}_j = \frac{w_j^2}{U_{jj}^2}, \\
W_{:,\,j+1:} &\leftarrow W_{:,\,j+1:} - \frac{e}{U_{jj}}\, U_{j,\,j+1:} .
\end{aligned}
$$

Choosing the mask per block, not all at once, is the paper's *adaptive mask selection*. Weights get reshaped by compensation as you go, so a mask fixed up front would be stale. The paper uses blocks of 128 columns.

**In a transformer**, the setup is GPTQ's: every linear layer in each block, 128 C4 segments as calibration, blocks processed in order. The same pass can also quantize the surviving weights, giving sparse and quantized weights at once, or produce 2:4 patterns by choosing 2 of every 4 within a row.

## The code, line by line

`sparsegpt(Wt, X, sparsity, block, damp)`:

- `U, dead = _hessian_inverse_chol(X, damp)` and `W[:, dead] = 0` are identical to GPTQ (see that page for the damping and dead-input handling).
- `keep = np.ones_like(W, bool)` is the mask, filled in block by block.
- `sal = W[:, b0:b1] ** 2 / np.diag(U)[b0:b1][None, :] ** 2` is step 1's saliency $w_j^2 / U_{jj}^2$ for the current block, computed from the current `W`.
- `k = int(round(b1 * sparsity)) - int(round(b0 * sparsity))` is this block's share of the row's quota. The per-block counts add up to exactly $\round(d_{\text{in}} \cdot \text{sparsity})$ per row, the same count magnitude and Wanda remove, whatever the block size. `idx = np.argsort(sal, 1)[:, :k]` picks the `k` lowest-saliency weights *in each row* of the block. `np.put_along_axis(blk, idx, False, 1)` writes them into the mask.
- Inside the block, `q = w * keep[:, j]` is the "quantized" column: unchanged where kept, zero where pruned.
- `err = (w - q) / U[j, j]` and `W[:, j + 1:] -= np.outer(err, U[j, j + 1:])` are GPTQ's update. They also adjust weights later in the block that are already marked for pruning. Those are zeroed when their own turn comes, and their error at that point includes the adjustment, just as in the reference code.
- `return W * keep` (already zero where pruned).

**Differences from the paper.** The reference implementation picks the lowest-saliency fraction across the *whole block*, pooling all rows, so rows can end up with different sparsity. This code picks exactly `k` per row per block, so every row ends at exactly the requested sparsity (the docstring says so). Lazy batching is omitted, as in the GPTQ code here.

## What the simulations show

3x256 teacher (0.9150), 512 calibration inputs. The hidden layers are pruned, including the first (12 raw inputs). The output layer stays dense, like `lm_head` in the papers. All three methods remove the same number of weights in every row, so the nonzero counts match: 72,980 at 0.5, 46,356 at 0.7 and 19,476 at 0.9, out of 140,052.

| fraction removed, block size | magnitude | Wanda | SparseGPT |
|---|---|---|---|
| 0.5, 32 | 0.8899 | 0.8838 | 0.8946 |
| 0.7, 32 | 0.7696 | 0.7187 | 0.8326 |
| 0.7, 256 | 0.7696 | 0.7187 | 0.8275 |
| 0.7, 4 | 0.7696 | 0.7187 | 0.7344 |
| 0.9, 32 | 0.1302 | 0.1152 | 0.2417 |

At 50%, all three are within 1.1 points of each other. At 70%, error compensation is worth 6.3 points over magnitude and 11.4 over Wanda. Block size matters at the small end: with blocks of 4 columns, each mask decision sees too little of the row, and SparseGPT falls to 0.7344, between Wanda and magnitude. Blocks of 32 and 256 are close. At 90% everything fails, though SparseGPT fails least.

## When it breaks

- At very high sparsity (0.9 here) no compensation can recover. The remaining weights can't express the layer.
- Small blocks force a fixed per-row sparsity in every short run of columns. That is a hidden structural constraint, and it shows in the block-4 row.
- Like GPTQ, it fits $H$ to calibration data. Too few inputs, or data unlike deployment, and the compensation targets the wrong inputs.
- Unstructured sparsity speeds nothing up without sparse kernels. The 2:4 pattern costs more accuracy.

## Try this

- **fraction removed** 0.5, then 0.7: watch the SparseGPT gap open up as sparsity rises.
- **fraction removed** 0.7, **block size** 4, 32, 256: look for the drop at small blocks while the nonzero counts stay equal.
- **fraction removed** 0.9: compare how all three methods fail.

## Going further

SparseGPT pruned OPT-175B and BLOOM-176B to 50% sparsity in under 4.5 hours on one GPU, with small perplexity increases. Wanda is its cheap approximation: a diagonal Hessian and no update. In practice, pruned LLMs are usually followed by a short recovery fine-tune, often with LoRA (see QLoRA), especially at 2:4 sparsity.
