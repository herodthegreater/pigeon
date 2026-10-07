## The idea

Pruning is quantization to a single level, zero. SparseGPT therefore reuses GPTQ's machinery. Walk through the input columns in order; when a weight is set to zero, adjust the weights to its right so the layer's output on calibration data changes as little as possible. The new part is choosing *which* weights to zero. SparseGPT decides one block of columns at a time, using the Optimal Brain Surgeon (OBS) saliency computed from the already-updated weights.

## How it works

Notation as on the GPTQ page: one output row `w`, calibration inputs `X` (n x d), Hessian `H = 2 XᵀX / n`, and `U`, the upper Cholesky factor of `H⁻¹`.

1. **Cost of removing a weight.** OBS says forcing coordinate `j` from `w_j` to 0 (the minimum-error forced change, derived on the GPTQ page with `e = w_j`) raises the error by `w_j² / (2 [H_F⁻¹]_jj)`, after the other weights are optimally adjusted. Here `F` is the set of columns not yet processed. Since `[H_F⁻¹]_jj = U[j, j]²`, the *saliency* is `w_j² / U[j, j]²`.
2. **Choose the mask a block at a time.** For columns `b0 .. b0+B-1`, compute the saliency of every weight in the block from the *current* weights, which already include the compensation from earlier blocks. Mark the lowest-saliency fraction for removal.
3. **Prune and compensate, column by column.** For `j` in the block: set the marked weights to zero, with `e = w_j` where pruned and 0 where kept. Then apply the GPTQ update `W[:, j+1:] -= (e / U[j, j]) ⊗ U[j, j+1:]`.
4. Repeat for the next block, then the next layer, which is fed outputs of the already-pruned layers.

Choosing the mask per block, not all at once, is the paper's *adaptive mask selection*. Weights get reshaped by compensation as you go, so a mask fixed up front would be stale. The paper uses blocks of 128 columns.

**In a transformer**, the setup is GPTQ's: every linear layer in each block, 128 C4 segments as calibration, blocks processed in order. The same pass can also quantize the surviving weights, giving sparse and quantized weights at once, or produce 2:4 patterns by choosing 2 of every 4 within a row.

## The code, line by line

`sparsegpt(Wt, X, sparsity, block, damp)`:

- `U, dead = _hessian_inverse_chol(X, damp)` and `W[:, dead] = 0` are identical to GPTQ (see that page for the damping and dead-input handling).
- `keep = np.ones_like(W, bool)` is the mask, filled in block by block.
- `sal = W[:, b0:b1] ** 2 / np.diag(U)[b0:b1][None, :] ** 2` is step 1's saliency for the current block, computed from the current `W`.
- `k = int(round((b1 - b0) * sparsity))` and `idx = np.argsort(sal, 1)[:, :k]` pick the `k` lowest-saliency weights *in each row* of the block. `np.put_along_axis(blk, idx, False, 1)` writes them into the mask.
- Inside the block, `q = w * keep[:, j]` is the "quantized" column: unchanged where kept, zero where pruned.
- `err = (w - q) / U[j, j]` and `W[:, j + 1:] -= np.outer(err, U[j, j + 1:])` are GPTQ's update. They also adjust weights later in the block that are already marked for pruning. Those are zeroed when their own turn comes, and their error at that point includes the adjustment, just as in the reference code.
- `return W * keep` (already zero where pruned).

**Differences from the paper.** The reference implementation picks the lowest-saliency fraction across the *whole block*, pooling all rows, so rows can end up with different sparsity. This code picks exactly `k` per row per block. Lazy batching is omitted, as in the GPTQ code here. And `k` is rounded per block, so the actual sparsity can differ from the requested one (see below).

## What the simulations show

3x256 teacher (0.9139), all four layers pruned, 512 calibration inputs. The last column is the count of nonzero parameters for SparseGPT; magnitude and Wanda have the same count except where noted.

| fraction removed, block | magnitude | Wanda | SparseGPT | SparseGPT nonzero |
|---|---|---|---|---|
| 0.5, 32 | 0.8594 | 0.8555 | 0.8934 | 70,420 (same) |
| 0.7, 32 | 0.6684 | 0.6377 | 0.8114 | 44,372 (others 42,776) |
| 0.7, 256 | 0.6684 | 0.6377 | 0.7908 | 42,776 (same) |
| 0.7, 4 | 0.6684 | 0.6377 | 0.5754 | 35,604 (others 42,776) |
| 0.9, 32 | 0.1085 | 0.0872 | 0.1865 | 13,812 (others 14,876) |

At matched sparsity, SparseGPT wins clearly: +3.4 points over magnitude at 50%, and +12 points at 70% with block 256. The block-size rows are confounded, though. With **block size** 32, each block drops `round(32 · 0.7) = 22` of 32 columns (68.75%), so SparseGPT keeps about 1,600 more weights. With **block size** 4, it drops `round(2.8) = 3` of 4 (75%), so it keeps fewer and does worse. At 90% everything fails.

## When it breaks

- At very high sparsity (0.9 here) no compensation can recover. The remaining weights can't express the layer.
- Small blocks force the same per-row sparsity in every short run of columns. That is a hidden structural constraint, on top of the rounding problem above.
- Like GPTQ, it fits `H` to calibration data. Too few inputs, or data unlike deployment, and the compensation targets the wrong inputs.
- Unstructured sparsity speeds nothing up without sparse kernels. The 2:4 pattern costs more accuracy.

## Try this

- **fraction removed** 0.5, **block size** 32: an apples-to-apples comparison of all three methods (equal nonzero counts).
- **fraction removed** 0.7 with **block size** 256, then 32: check the nonzero-parameter column before crediting SparseGPT with the difference.
- **block size** 4: look for SparseGPT falling below Wanda, and explain it from the parameter count.

## Going further

SparseGPT pruned OPT-175B and BLOOM-176B to 50% sparsity in under 4.5 hours on one GPU, with small perplexity increases. Wanda is its cheap approximation: a diagonal Hessian and no update. In practice, pruned LLMs are usually followed by a short recovery fine-tune, often with LoRA (see QLoRA), especially at 2:4 sparsity.
