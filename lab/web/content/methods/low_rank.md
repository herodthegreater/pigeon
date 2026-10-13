## The idea

If most of what a weight matrix $W \in \R^{m \times n}$ ($m$ inputs, $n$ outputs) does happens along a few directions, it can be replaced by two thin matrices, $A \in \R^{m \times r}$ and $B \in \R^{r \times n}$, with $W \approx AB$. That costs $r(m + n)$ numbers instead of $mn$. The *singular value decomposition* (SVD) writes $W = U \Sigma V^\top$, with the *singular values* $\sigma_1 \ge \sigma_2 \ge \dots$ on the diagonal of $\Sigma$, sorted largest first. Keeping the top $r$ of them,

$$
W \approx W_r = U_r \Sigma_r V_r^\top = \sum_{i=1}^{r} \sigma_i \, u_i v_i^\top ,
$$

gives the best rank-$r$ approximation of $W$ in squared error (the Eckart-Young theorem):

$$
\begin{aligned}
\lVert W - W_r \rVert_F^2 &= \min_{\rank(X) \le r} \lVert W - X \rVert_F^2 \\
&= \sum_{i > r} \sigma_i^2
\end{aligned}
$$

It is best for the weights, not the outputs, so recovery still helps.

## How it works

1. For each layer, check that factorizing saves parameters. That sets a break-even rank:

$$
r(m + n) < mn \iff r < \frac{mn}{m + n}
$$

2. Compute the SVD $W = U \Sigma V^\top$.
3. Keep the top $r$ and split $\Sigma_r^{1/2}$ between the two factors, which keeps them at similar scales. Their product is exactly the truncation $W_r$:

$$
\begin{aligned}
A &= U_r \Sigma_r^{1/2} \\
B &= \Sigma_r^{1/2} V_r^\top \\
AB &= U_r \Sigma_r V_r^\top = W_r
\end{aligned}
$$

4. Replace the layer by two layers, $x \mapsto xA \mapsto (xA)B + b$, with no nonlinearity in between.
5. Recover with knowledge distillation (KD) from the original net.

## The code, line by line

`low_rank` in `lab/compress.py` builds a new list of layers. For each $d_{\text{in}} \times d_{\text{out}}$ weight, the guard `if i in layers and rank * (din + dout) < din * dout` decides whether to factorize. For the 3x256 teacher it applies to the $256 \times 256$ hidden layers when $r < 128$, to the $256 \times 20$ output layer when $r \le 18$, and to the $12 \times 256$ input layer only when $r \le 11$. At rank 128 nothing qualifies, so the page returns the teacher unchanged.

`U, S, Vt = np.linalg.svd(w, full_matrices=False)` gives the thin SVD. `A = (U[:, :r] * np.sqrt(S[:r]))` scales each kept column of `U`, and `B = (np.sqrt(S[:r])[:, None] * Vt[:r])` scales each kept row of `Vt`, so `A @ B` equals the rank-$r$ truncation $W_r$ exactly. `b += [np.zeros(r, F), src.b[i].copy()]` gives the first factor a zero bias and the second the original one, `sizes += [r, dout]` records the width-$r$ bottleneck, and `lin += [True, False]` marks the first factor as linear. Other layers are copied as is.

`out = MLP(sizes, seed=0)` is built and its weights overwritten with `out.W, out.b = W, b`. `out.lin = lin` makes `MLP.forward` skip the ReLU after each `A`. `out.res` and `out.masks` are reset, and the function reads `src.W` rather than `src.eff(i)`, so residual flags, masks and quantization in the source are dropped. That is fine for the dense teacher here.

`run_low_rank` in `lab/methods.py` factorizes the teacher at the chosen rank and optionally distills it for `FT` = 500 steps.

## What the simulations show

From `low_rank` (KD rows: 1,500 steps, mean of 2 seeds). For comparison, the same experiment prunes whole neurons (Taylor criterion) to 3xW and recovers with KD.

| rank | params | SVD, no recovery | SVD + KD |
|---|---|---|---|
| 128 (unchanged) | 140,052 | 0.9139 | 0.9194 |
| 64 | 74,644 | 0.8846 | 0.9213 |
| 32 | 41,812 | 0.7897 | 0.9197 |
| 16 | 24,708 | 0.4197 | 0.9113 |
| 8 | 13,364 | 0.0659 | 0.7354 |
| 4 | 7,076 | 0.0739 | 0.3751 |

| neuron width | params | neuron prune + KD |
|---|---|---|
| 128 | 37,268 | 0.9152 |
| 64 | 10,452 | 0.8928 |
| 32 | 3,188 | 0.7906 |
| 16 | 1,092 | 0.5865 |

(Teacher: 140,052 params, 0.9139 accuracy.) Rank 128 is the teacher itself, since no layer passes the break-even guard, yet KD "recovery" lifts it to 0.9194. Recovery is also extra training, so compare recovered rows with each other. Down to rank 16 (18% of the parameters), the factorized net recovers to about the teacher's accuracy. Without recovery, error compounds quickly: rank 16 alone keeps only 0.4197. Per parameter, the two methods trade places. Around 40k parameters SVD is slightly ahead (0.9197 at 41,812 vs 0.9152 at 37,268). Around 10-13k, neuron pruning is clearly ahead (0.8928 at 10,452 vs 0.7354 at 13,364). A rank-8 bottleneck squeezes every layer through 8 numbers, while a 3x64 net keeps 64 ReLUs per layer.

Page runs (500 recovery steps): rank 16 goes from 0.4010 to 0.8953, rank 4 from 0.0724 to 0.2728, and rank 128 from 0.9150 to 0.9192. Rank 128 starts at 0.9150 here, not the table's 0.9139: the experiment's teacher scored 0.9139, and the copy the Run button loads, trained again with the same recipe, scores 0.9150.

The toy's matrices are small; in large models many layers are closer to low rank.

## When it breaks

- **No savings at high rank.** At or above the break-even rank $mn / (m + n)$, factorizing costs parameters, which is why the guard exists.
- **Extra depth.** Each factorized layer becomes two sequential matmuls. With few FLOPs per layer, the extra kernel launch can cost more time than the saved FLOPs.
- **The wrong objective.** SVD minimizes weight error, not output error. Small singular directions can still carry important inputs; activation-weighted variants (Hsu et al.) use data to fix this.
- **Thin layers.** The 12-input and 20-output layers barely compress and become bottlenecks at low rank.

## Try this

- Set **recovery** to none and step **rank** through 64, 32, 16 and 8. Find the knee where the unrecovered model falls off.
- Set **recovery** to kd at rank 16 and rank 8. Compare the recovered accuracy with the Structured pruning page at a similar parameter count (width 32 or 64).

## Going further

For conv layers, low-rank methods factor a $k \times k$ convolution into a $k \times 1$ and a $1 \times k$ convolution, or into a small conv followed by a $1 \times 1$ projection (Denton et al.). In language models, low rank is mostly used for adapters: LoRA trains a low-rank update instead of compressing $W$ (see QLoRA). The YOLOv8n cockpit does not factorize layers. The closest analogue in this repo is its channel pruning (`harness/prune.py`), and SVD beat neuron pruning per parameter here only at mild compression.
