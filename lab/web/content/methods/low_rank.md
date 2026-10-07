## The idea

A weight matrix `W` of shape `(in, out)` maps inputs to outputs. If most of what it does happens along a few directions, it can be replaced by two thin matrices, `A` of shape `(in, r)` and `B` of shape `(r, out)`, with `W ≈ A B`. That costs `r * (in + out)` numbers instead of `in * out`. The *singular value decomposition* (SVD) writes `W = U diag(S) Vt`, with the *singular values* in `S` sorted largest first. Keeping the top `r` of them gives the best rank-`r` approximation of `W` in squared error (the Eckart-Young theorem). The approximation is best for the weights, not necessarily for the network's outputs, so a recovery step still helps.

## How it works

1. For each layer, check that factorizing saves parameters: `r * (in + out) < in * out`.
2. Compute `U, S, Vt = svd(W)`.
3. Keep the top `r`: `A = U[:, :r] * sqrt(S[:r])` and `B = sqrt(S[:r])[:, None] * Vt[:r]`. Splitting `sqrt(S)` between the two keeps them at similar scales, which helps later fine-tuning.
4. Replace the layer by two layers, `x -> x A -> (x A) B + b`, with no nonlinearity in between.
5. Recover with knowledge distillation (KD) from the original net.

## The code, line by line

`low_rank` in `lab/compress.py` builds a new list of layers from `src`. For each layer, `din, dout = w.shape`. The guard `if i in layers and rank * (din + dout) < din * dout` decides whether to factorize. For the 3x256 teacher it applies to the 256x256 hidden layers when `r < 128`, to the 256x20 output layer when `r <= 18`, and to the 12x256 input layer only when `r <= 11`. At rank 128 nothing qualifies, so the page returns the teacher unchanged.

`U, S, Vt = np.linalg.svd(w, full_matrices=False)` gives the thin SVD. `A = (U[:, :r] * np.sqrt(S[:r]))` scales each kept column of `U`, and `B = (np.sqrt(S[:r])[:, None] * Vt[:r])` scales each kept row of `Vt`, so `A @ B` equals the rank-`r` truncation exactly. `W += [A, B]` and `b += [np.zeros(r, F), src.b[i].copy()]` give the first factor a zero bias and the second the original bias. `sizes += [r, dout]` records the new bottleneck of width `r`, and `lin += [True, False]` marks the first factor as linear. A layer that doesn't qualify is copied as is.

The new network is made with `out = MLP(sizes, seed=0)`, whose random weights are then overwritten with `out.W, out.b = W, b`. `out.lin = lin` tells `MLP.forward` to skip the ReLU after each `A` (`a = z if self.lin[i] else np.maximum(z, 0)`). `out.res` and `out.masks` are reset, so residual connections and masks in the source are dropped. The function reads `src.W`, not `src.eff(i)`, so it would also ignore a pruning mask or quantization. That is fine for the dense teacher used here and wrong for anything else.

`run_low_rank` in `lab/methods.py` factorizes the teacher at the chosen rank and optionally distills it for `FT` = 500 steps at `FT_LR` = 3e-3, the same learning rate as training from scratch.

## What the simulations show

From `low_rank` (KD rows: 1,500 steps, mean of 2 seeds). For comparison, the same experiment prunes whole neurons (Taylor criterion) to 3xW and recovers with KD.

| rank | params | SVD, no recovery | SVD + KD |
|---|---|---|---|
| 64 | 74,644 | 0.8846 | 0.9146 |
| 32 | 41,812 | 0.7897 | 0.9145 |
| 16 | 24,708 | 0.4197 | 0.8988 |
| 8 | 13,364 | 0.0659 | 0.6258 |
| 4 | 7,076 | 0.0739 | 0.2820 |

| neuron width | params | neuron prune + KD |
|---|---|---|
| 128 | 37,268 | 0.9049 |
| 64 | 10,452 | 0.8161 |
| 32 | 3,188 | 0.6035 |
| 16 | 1,092 | 0.4269 |

(Teacher: 140,052 params, 0.9139 accuracy.) Rank 32 with recovery matches the teacher at 30% of its parameters. Without recovery, error compounds quickly: rank 16 alone keeps only 0.4197. Per parameter, the two methods trade places. Around 40k parameters SVD is ahead (0.9145 at 41,812 vs 0.9049 at 37,268). Around 10-13k, neuron pruning is ahead (0.8161 at 10,452 vs 0.6258 at 13,364). A rank-8 bottleneck through a 256-wide layer starves every layer at once, while a 3x64 net still has 64 ReLUs per layer.

Page runs: rank 16 goes from 0.4197 to 0.8711 after 500 KD steps. Rank 4 goes from 0.0739 to 0.1871. Rank 128 reproduces the teacher exactly.

The toy's matrices are small. In large models many layers are much closer to low rank, and the savings start at a far lower relative rank.

## When it breaks

- **No savings at high rank.** Below the break-even rank `in * out / (in + out)`, factorizing costs parameters, which is why the guard exists.
- **Extra depth.** Each factorized layer becomes two sequential matmuls. With few FLOPs per layer, the extra kernel launch can cost more time than the saved FLOPs.
- **The wrong objective.** SVD minimizes weight error, not output error. Directions with small singular values can still carry inputs that matter. Activation-weighted variants (Hsu et al.) fix this with data.
- **Thin layers.** The 12-input and 20-output layers barely compress and become bottlenecks at low rank.
- **Lost structure.** `low_rank` drops masks, quantization and residual flags from its input.

## Try this

- Set **recovery** to none and step **rank** through 64, 32, 16 and 8. Find the knee where the unrecovered model falls off.
- Set **recovery** to kd at rank 16 and rank 8. Compare the recovered accuracy with the Structured pruning page at a similar parameter count (width 32 or 64).
- Set **rank** to 128 to confirm the break-even guard: the "no recovery" row equals the teacher.

## Going further

For conv layers, low-rank methods factor a `k x k` convolution into a `k x 1` and a `1 x k` convolution, or into a small conv followed by a `1 x 1` projection (Denton et al.). In language models, factorization is mostly used for adapters rather than compression: LoRA trains a low-rank update instead of compressing `W`. See the QLoRA page. pigeon does not factorize layers. The closest analogue in this repo is its channel pruning (`harness/prune.py`), and SVD beat neuron pruning per parameter here only at mild compression.
