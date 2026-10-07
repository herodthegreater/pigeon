## The idea

RTN rounds each weight on its own, so the errors in a row can pile up in the layer's output. GPTQ rounds one input column at a time and, after each column, nudges the not-yet-rounded weights so they cancel the error just made. Errors are measured on real inputs, through the *Hessian* `H = 2 XᵀX / n`, the curvature of the layer's squared output error on `n` calibration inputs. The update comes from Optimal Brain Surgeon (OBS); GPTQ makes it cheap enough for 175B-parameter models.

## How it works

Notation: one output row `w` (length `d`, one weight per input), calibration inputs `X` (n x d), and quantizer `quant()`. The layer error for this row is `E(w') = |X w'ᵀ - X wᵀ|²`, which equals `(w'-w) H (w'-w)ᵀ` up to a constant factor. `H` is the same for every row, so rows are independent and share one `H`.

**OBS step.** Force coordinate `j` to its quantized value. The forced change is `δ_j = -e` with `e = w_j - quant(w_j)`. Choose the other coordinates of `δ` to minimize `½ δ H δᵀ`:

1. Lagrangian: `L = ½ δ H δᵀ + λ (δ_j + e)`.
2. Setting the gradient to zero gives `H δᵀ = -λ e_j`, so `δ = -λ H⁻¹[j, :]`.
3. The constraint `δ_j = -e` gives `λ = e / H⁻¹[j, j]`.
4. So `δ = -(e / H⁻¹[j, j]) · H⁻¹[j, :]`, and the error grows by `e² / (2 H⁻¹[j, j])`.

**Remove `j` and repeat.** The next column's problem is over the remaining coordinates `F`. Its inverse Hessian comes from one Gaussian-elimination step: `H_F⁻¹ = H⁻¹ - H⁻¹[:, j] H⁻¹[j, :] / H⁻¹[j, j]`, restricted to `F`.

**GPTQ's two tricks.**

1. Use the same column order (0, 1, 2, ...) for every row. Then the sequence of `H_F⁻¹` matrices is shared by all rows and computed once per layer, not once per row.
2. Successive Gaussian elimination is a Cholesky factorization. Let `U` be the upper Cholesky factor of `H⁻¹` (`H⁻¹ = UᵀU`). Row `j` of `U` is row `j` of `H_F⁻¹` (with `F = {j, ..., d-1}`) divided by `sqrt(H_F⁻¹[j, j])`. Substitute `H_F⁻¹[j, j] = U[j, j]²` and `H_F⁻¹[j, j+1:] = U[j, j] · U[j, j+1:]` into step 4: `δ[j+1:] = -((w_j - quant(w_j)) / U[j, j]) · U[j, j+1:]`.

So: compute `U` once; for `j = 0 .. d-1`, round column `j` of all rows and apply that update to the columns on its right.

**Damping.** Add `0.01 · mean(diag H)` to the diagonal so `H` is safely invertible.

**In a transformer**, `X` is the stack of token vectors entering one linear layer (q, k, v, o, up, gate or down projection); the paper used 128 random 2048-token segments of C4, so `n` is about 262k. Blocks go in order, and each `H` comes from the outputs of the already-quantized blocks before it, so every layer is fitted to the inputs it will actually see.

## The code, line by line

`_hessian_inverse_chol(X, damp)`:

- `H = 2 * X.T @ X / len(X)` is the Hessian.
- `dead = np.diag(H) == 0` finds inputs that are always zero (ReLU neurons that never fire on calibration data). `H[dead, dead] = 1` keeps `H` invertible, and `gptq` zeroes those weights with `W[:, dead] = 0`, which is harmless.
- `H += damp * np.mean(np.diag(H)) * np.eye(len(H))` adds the damping.
- `U = np.linalg.cholesky(np.linalg.inv(H)).T`: numpy returns the lower factor `L`, and `Lᵀ` is the upper `U` from trick 2.

`gptq(Wt, X, bits, group, damp, act_order)`:

- `scale = ... np.abs(W).max(1) / qmax`: one scale per output row, fixed from the starting weights.
- `if group and j % group == 0` resets the scale at each group boundary, computed from the *already-updated* weights `W[:, j:j + group]`, as the reference code does.
- `q = np.clip(np.round(w / scale), -qmax, qmax) * scale` rounds column `j` of every row at once.
- `err = (w - q) / U[j, j]` and `W[:, j + 1:] -= np.outer(err, U[j, j + 1:])` are the update from trick 2, applied to all rows together.
- With `act_order`, `perm = np.argsort(-np.diag(X.T @ X))` quantizes the inputs with the largest activations first. `U` is recomputed for that order, and `Q[:, np.argsort(perm)]` restores the original column order.

`apply_layerwise` calls `gptq` layer by layer, feeding each one the outputs of the already-quantized layers before it.

**Simplifications.** The paper's *lazy batch updates* (applying the update in blocks of 128 columns: same result, faster on GPUs) are omitted. The quantizer is symmetric, while the paper's default is asymmetric (min/max plus a zero point). `act_order` is a heuristic from the official code, not part of the paper's main algorithm.

## What the simulations show

3x256 teacher, 91.39% accuracy. The three hidden layers are quantized, including the first one, which reads the 12 raw inputs. The output layer stays at full precision, as papers leave `lm_head` alone. "2-bit" means the ternary levels `{-s, 0, s}` (see the RTN page).

| setting | RTN | GPTQ |
|---|---|---|
| 4-bit, per row | 0.9103 | 0.9131 |
| 3-bit, per row | 0.8922 | 0.9079 |
| 3-bit, scale per 16 | 0.9010 | 0.9110 |
| 2-bit, per row | 0.4686 | 0.8207 |
| 2-bit, scale per 16 | 0.7585 | 0.8389 |

With **largest activations first**, GPTQ reaches 0.9097 at 3 bits and 0.8367 at 2 bits. Calling `lw.rtn` and `lw.gptq` directly at 3 bits on each layer's teacher inputs, GPTQ cut the relative output error of the second layer from 0.0300 to 0.0028, but of the first layer (only 12 inputs, so few columns to push errors onto) just from 0.0333 to 0.0322.

## When it breaks

- A badly estimated `H` (few calibration inputs, or data unlike deployment) cancels error on the wrong inputs.
- Heavy damping makes `H` close to a multiple of the identity. Then `U[j, j+1:] ≈ 0` and GPTQ turns back into RTN.
- Compensation can push weights past the row scale fixed at the start, and those get clipped to `±qmax`.
- With both `act_order` and groups, groups form in activation order, so they are not contiguous columns. Real kernels then need a stored column-to-group index (`g_idx`).

## Try this

- **bits** 2, then 3, then 4, **scale per** row: watch the GPTQ-over-RTN gap shrink as rounding gets finer.
- Turn on **largest activations first** at 2 bits. Look for the gain over 0.8207.

## Going further

GPTQ quantizes a 175B model in about four GPU hours and underlies many 4-bit LLM checkpoints. Later work keeps the same layer-wise objective `(w'-w) H (w'-w)ᵀ` and changes the grid or the basis: random rotations before rounding (QuIP#, QuaRot), learned vector codebooks (AQLM). SparseGPT reuses the exact same update for pruning.
