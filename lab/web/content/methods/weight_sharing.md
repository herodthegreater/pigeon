## The idea

Uniform quantization places its levels evenly between `-max` and `+max`. Trained weights are not spread evenly: most sit near zero and a few are large, so evenly spaced levels waste most of their resolution on empty space. *Weight sharing* lets the data choose the levels. Run k-means on the weight values of a layer to find `k` *centroids*, the representative values, and replace each weight by its nearest centroid. Storage becomes a `log2(k)`-bit index per weight plus a codebook of `k` floats per layer. This is the "trained quantization" step of Han et al.'s Deep Compression.

## How it works

1. For each layer, take all its weight values as a 1-D list (skipping pruned zeros if the layer is sparse).
2. Initialise `k` centroids evenly spaced from the smallest to the largest value (*linear init*).
3. Repeat: assign every weight to its nearest centroid, then move each centroid to the mean of its weights (Lloyd's algorithm for k-means).
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

`kmeans_share` in `lab/compress.py` copies the net and loops over every layer, including the output layer. Biases are left in float. `w = net.W[i].reshape(-1)` flattens the matrix. `nz = w != 0 if net.masks[i] is not None else np.ones_like(w, bool)`: on a pruned layer only surviving weights join the clustering, so zeros stay zero and don't drag a centroid towards 0. `v = w[nz]` is the list to cluster.

`c = np.linspace(v.min(), v.max(), k)` is the linear init. Han et al. found it beats random init because it keeps centroids on the rare large weights, which matter most. Each of the 25 iterations computes `a = np.abs(v[:, None] - c[None]).argmin(1)`. `v[:, None] - c[None]` broadcasts to an `(n_weights, k)` table of distances, and `argmin(1)` picks the nearest centroid for each weight. `c[j] = v[a == j].mean()` moves each centroid to the mean of its members. The `if (a == j).any()` guard leaves an empty cluster's centroid where it was instead of producing NaN. After the loop, one more assignment is done, `out[nz] = c[a]` writes the centroid values, and the matrix is reshaped back.

The result is still a float matrix, but it holds at most `k` distinct values per layer. The `params` column counts nonzero weights, and a centroid is almost never exactly 0, so it stays at 140,052 even though storage drops to `log2(k)` bits per weight.

`run_weight_sharing` in `lab/methods.py` compares three rows. The uniform baseline is `quantize(t, bits + 1)` with `bits = ceil(log2(k))`, per tensor. Symmetric uniform quantization at `b + 1` bits has `2^(b+1) - 1` levels, so k = 4 is compared with a 7-level grid that also costs one more bit per weight. The comparison favours uniform. Fine-tuning does two rounds of `kmeans_share(distill(m, t, r, steps=FT // 2, lr=FT_LR), k)`. KD from the teacher moves the weights freely, off their centroids, and then they are re-clustered and snapped. Deep Compression instead trains the centroids themselves: it sums the gradients of all weights that share a centroid and updates that one value, so the weights never leave the codebook.

## What the simulations show

From `weight_sharing` (accuracy on 20 classes; "bits" is `log2(k)`; uniform uses one bit more):

| bits (k) | teacher uniform | teacher k-means | teacher + KD + re-snap | student uniform | student k-means |
|---|---|---|---|---|---|
| 1 (2) | 0.0707 | 0.3999 | 0.2913 | 0.0764 | 0.1733 |
| 2 (4) | 0.7766 | 0.8487 | 0.8468 | 0.4461 | 0.4828 |
| 3 (8) | 0.8998 | 0.9039 | 0.8953 | 0.7186 | 0.6607 |
| 4 (16) | 0.9134 | 0.9088 | 0.9072 | 0.7747 | 0.7423 |
| 5 (32) | 0.9139 | 0.9129 | 0.9131 | 0.7966 | 0.7862 |

On the 3x256 teacher, 4 k-means values per layer (2 bits) reach 0.8487, beating a 7-level uniform grid at 3 bits (0.7766). With 16 or more values the codebook brings nothing, because uniform is already near lossless. On the 2x32 student, k-means wins only at 1-2 bits. From 8 values up, the uniform grid with nearly twice as many levels wins. The fine-tune-then-re-snap loop never clearly helped, and at k = 2 it hurt the teacher (0.3999 → 0.2913). Snapping after free fine-tuning throws away what the fine-tune learned.

Page runs at the defaults (k = 4, fine-tune on): uniform 3-bit 0.7766, k-means 0.8487, after KD and re-snap 0.8354.

The toy can't speak to decoding cost. Real kernels need a lookup per weight, which uniform integer arithmetic avoids.

## When it breaks

- **Re-snapping undoes fine-tuning.** Training must keep weights tied to centroids (shared-gradient centroid updates), or each snap starts over.
- **One codebook per layer is coarse.** A layer whose rows have very different scales gets centroids that fit neither. Per-group codebooks, or a uniform grid with group scales, can beat it.
- **Tiny `k` collapses expressiveness.** Two values per layer turns a layer into a signed binary pattern, and a small network cannot absorb that.
- **Cost of clustering.** The `(n, k)` distance table is fine here but grows with layer size times `k`. Large models cluster in chunks or sample.
- **Hardware.** Indices into a codebook need a gather. Integer matmul units can't use them directly.

## Try this

- Set **values per layer** to 4 with **fine-tune + re-snap** off, then 8, then 16. Watch the gap to the uniform row shrink and then flip.
- At **values per layer** 2, turn **fine-tune + re-snap** on and off. The KD row going down is the re-snap problem in action.
- Set **values per layer** to 3. The uniform baseline uses `ceil(log2(3)) + 1` = 3 bits, so you can see how the page rounds a non-power-of-two `k`.

## Going further

Deep Compression chains pruning, then weight sharing with centroid fine-tuning, then Huffman coding of the indices. Modern non-uniform formats follow the same idea with fixed codebooks (NF4 in QLoRA uses levels matched to a normal distribution) or learned vector codebooks for groups of weights. pigeon's harness only does uniform symmetric quantization (`harness/quant.py`). The teacher results above suggest a per-layer codebook could beat uniform at 2-3 bits. A per-layer k-means codebook could be added to `harness/model.py` beside `apply_uniform` and swept with `carve.py`.
