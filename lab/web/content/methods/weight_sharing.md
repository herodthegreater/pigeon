## The idea

Uniform quantization spaces its levels evenly between `-max` and `+max`, but most trained weights sit near zero, so most levels go unused. *Weight sharing* lets the data choose the levels: k-means on a layer's weight values finds `k` *centroids*, and each weight is replaced by its nearest one. Storage is a `log2(k)`-bit index per weight plus a `k`-float codebook per layer.

## How it works

1. Take each layer's weights as a 1-D list (skipping pruned zeros).
2. Initialise `k` centroids evenly spaced from the smallest to the largest value (*linear init*).
3. Repeat: assign each weight to its nearest centroid, move each centroid to its weights' mean (Lloyd's algorithm).
4. Replace each weight by its centroid.
5. Optionally fine-tune and snap the weights back to a codebook.

```python
c = linspace(v.min(), v.max(), k)
for _ in range(iters):
    a = argmin_j |v - c[j]|
    c[j] = mean(v[a == j])     # for each non-empty j
w_shared = c[a]
```

## The code, line by line

`kmeans_share` in `lab/compress.py` copies the net and loops over every layer; biases stay in float. `w = net.W[i].reshape(-1)` flattens the matrix, and `nz = w != 0 if net.masks[i] is not None else ...` makes a pruned layer cluster only its surviving weights, so zeros stay zero. `v = w[nz]` is the list to cluster.

`c = np.linspace(v.min(), v.max(), k)` is the linear init, which Han et al. preferred because it keeps centroids on the rare large weights. Each of 25 iterations computes `a = np.abs(v[:, None] - c[None]).argmin(1)`: the broadcast builds an `(n_weights, k)` distance table and `argmin(1)` picks each weight's nearest centroid. `c[j] = v[a == j].mean()` moves each centroid to its members' mean, and the `if (a == j).any()` guard leaves empty clusters in place instead of producing NaN. A final assignment and `out[nz] = c[a]` write the centroid values back.

The result is a float matrix with at most `k` distinct values per layer. `params` counts nonzero weights, and centroids are almost never 0, so it stays at 140,052 although storage drops to `log2(k)` bits per weight.

`run_weight_sharing` in `lab/methods.py` and the `weight_sharing` experiment both compare at equal storage: `quantize(t, bits)` per tensor, with `bits = ceil(log2(k))`, against `k` centroids. That is `2^b - 1` uniform levels against `2^b`. Fine-tuning runs `kmeans_share(distill(m, t, r, steps=FT // 2, lr=FT_LR), k)` twice: KD moves the weights off their centroids, then they are re-clustered and snapped. Deep Compression instead trains the centroids themselves, summing the gradients of all weights that share one, so weights never leave the codebook.

## What the simulations show

From `weight_sharing`, at equal bits per weight (accuracy on 20 classes; the student is the 2x32 net distilled from the teacher):

| bits (k) | teacher uniform | teacher k-means | teacher + KD + re-snap | student uniform | student k-means | student + KD + re-snap |
|---|---|---|---|---|---|---|
| 1 (2) | 0.4013 | 0.3999 | 0.4426 | 0.1522 | 0.1872 | 0.1981 |
| 2 (4) | 0.0707 | 0.8487 | 0.8828 | 0.0686 | 0.4555 | 0.4639 |
| 3 (8) | 0.7766 | 0.9039 | 0.9176 | 0.4851 | 0.6453 | 0.6677 |
| 4 (16) | 0.8998 | 0.9088 | 0.9213 | 0.7229 | 0.7839 | 0.7844 |
| 5 (32) | 0.9134 | 0.9129 | 0.9200 | 0.8132 | 0.8153 | 0.8268 |

At the same storage, a learned codebook beats the uniform grid at 2, 3 and 4 bits on both models. At 2 bits, uniform symmetric is ternary, `{-s, 0, +s}` with `s` the largest weight, so most weights round to zero (teacher 0.0707), while 4 k-means values keep 0.8487. At 1 bit both have two levels and score alike; at 5 bits both are near lossless. The student, with less spare capacity, loses more at every width: at 3 bits its k-means accuracy is 0.6453 against 0.8281 in fp32, while the teacher keeps 0.9039 against 0.9139. KD fine-tuning with re-snapping helps the teacher at every `k`, ending above fp32 from 8 values up, which is partly just extra training on 28,000 inputs. For the student it adds at most 0.0224.

Page runs at the defaults (k = 4, fine-tune on): uniform 2-bit 0.0707, k-means 0.8487, after KD and re-snap 0.8646. At k = 2: uniform 1-bit 0.4013, k-means 0.3999, after KD and re-snap 0.4165. Symmetric 2-bit is ternary {-s, 0, +s}, which is why it scores below 1-bit sign quantization.

## When it breaks

- **Re-snapping throws away part of the fine-tune.** Free fine-tuning moves weights off their centroids, and each snap pulls them back. Training that keeps weights tied to centroids (shared-gradient centroid updates) avoids the round trip.
- **One codebook per layer is coarse.** Rows with very different scales get centroids that fit neither. Per-group codebooks or group-scaled uniform grids can beat it.
- **Tiny `k`.** Two values per layer is a signed binary pattern, which a small network cannot absorb.
- **Clustering cost.** The `(n, k)` distance table grows with layer size times `k`, so large models cluster in chunks.
- **Hardware.** Codebook indices need a lookup per weight, which integer matmul units can't do directly.

## Try this

- Set **values per layer** to 4 with **fine-tune + re-snap** off, then 8, then 16. The page's uniform row gets an extra bit. Watch k-means lose its lead as `k` grows.
- At **values per layer** 2 and 16, toggle **fine-tune + re-snap** and compare what the KD row adds.
- Set **values per layer** to 3. The uniform baseline uses `ceil(log2(3)) + 1` = 3 bits, so you can see how the page rounds a non-power-of-two `k`.

## Going further

Deep Compression chains pruning, then weight sharing with centroid fine-tuning, then Huffman coding of the indices. Modern non-uniform formats use fixed codebooks (QLoRA's NF4 levels follow a normal distribution) or learned vector codebooks. pigeon's harness only does uniform symmetric quantization (`harness/quant.py`). The teacher results above suggest a per-layer codebook could beat per-tensor uniform at 2-3 bits, and at equal storage the student benefited too. A per-layer k-means codebook could be added to `harness/model.py` beside `apply_uniform` and swept with `carve.py`.
