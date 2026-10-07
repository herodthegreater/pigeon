## The idea

A small network is hard to train well. Gradient descent on few parameters gets stuck more easily than on many. The trick is to train a bigger network that is secretly the same small one. Write each weight matrix as a product of two matrices with a wide linear layer between them, `W = W1 W2`. A product of linear maps is still one linear map, so after training you multiply the factors back together and get a network of exactly the original shape. The expanded version can express nothing new. What changes is how gradient descent moves through the weights.

## How it works

1. Take the target architecture, say 12 → 16 → 16 → 20.
2. Replace every layer `a → b` by two: `a → E → b` with `E = expand * max(a, b)`, and no activation after the first (a *linear* layer).
3. Train this expanded network normally.
4. Collapse each pair: `(h W1 + b1) W2 + b2 = h (W1 W2) + (b1 W2 + b2)`. The new weight is `W1 @ W2` and the new bias is `b1 @ W2 + b2`.
5. Deploy the collapsed network. Its outputs are identical to the trained expanded one, up to float rounding.

## The code, line by line

`overparam` in `lab/experiments/small.py` builds the expanded shape. `base = [d.world.spec.d] + list(hidden) + [d.world.C]` is the target, for example `[12, 16, 16, 20]`. For each target layer `i`, `lin.add(len(sizes) - 1)` marks the index of the layer about to be added as linear, and `sizes += [expand * max(base[i], base[i + 1]), base[i + 1]]` appends the wide middle and then the real output. With width 16 and expansion 4 the sizes are `[12, 64, 16, 64, 16, 80, 20]`, which is 6,980 parameters instead of 820. Every layer is expanded, input and output included. `MLP(sizes, seed=seed, linear=lin)` creates it, and in `MLP.forward` `a = z if self.lin[i] else np.maximum(z, 0)` skips the ReLU on the marked layers.

`MLP.collapse_linear` in `lab/nn.py` works on a copy. It walks the layers with `i` and, whenever `net.lin[i]` is true, merges layer `i` into layer `i + 1`. It first asserts the layer is not residual and has no masks, because neither survives the algebra. `net.b[i + 1] = net.b[i] @ W2 + net.b[i + 1]` pushes the first bias through the second matrix. `net.W[i + 1] = net.W[i] @ W2` is the merged weight. Layer `i` is then deleted from every per-layer list and `del net.sizes[i + 1]` drops the wide middle width. `i` is not advanced, so the merged (now nonlinear) layer is checked once more and skipped.

`run_overparam` in `lab/methods.py` trains the plain `2xW` net and the expanded net for the same 1,000 steps. It reports the expanded and the collapsed net with their true sizes from `n_params(False)`.

## What the simulations show

From `small_tricks`, with 3,000 training steps and 3 seeds (accuracy on 20 classes; the teacher reaches 0.9139):

| student | plain (labels) | over-parameterize, then collapse | 4x longer training |
|---|---|---|---|
| 2x16 (820 params) | 0.6637 | 0.7099 | 0.7146 |
| 2x32 (2,132 params) | 0.8170 | 0.8420 | 0.8405 |

The collapsed network gains as much as training 4 times longer, at no inference cost. Training is costlier: each step of the expanded 2x16 net touches 6,980 parameters instead of 820.

Page runs (1,000 steps): at width 16 and expansion 4, plain gets 0.5803 and the collapsed net 0.6589. The expanded and collapsed rows match exactly. With expansion 1 (a product of two matrices with no extra width) the collapsed net gets 0.5766, no better than plain. The extra width, not the factorization, is what helps. At width 32: plain 0.7398, collapsed 0.8278.

The toy doesn't separate *why* it helps. Candidate reasons include the implicit momentum-like acceleration Arora et al. describe, the different effective initialization of a product of two random matrices, and weight decay acting on the factors rather than on their product.

## When it breaks

- **Nonlinearities in between.** Collapsing is exact only if nothing nonlinear sits between the factors: no ReLU, no dropout. Batch-norm can be folded first (see Batch-norm folding).
- **Masks and residuals.** `collapse_linear` asserts against them. A pruning mask on a factor does not become a mask on the product.
- **No extra width, no gain.** Expansion 1 reproduced the plain result.
- **Capacity is unchanged.** It helps optimization, not expressiveness.

## Try this

- At **width** 16, step **expansion factor** through 1, 2, 4 and 8. Look for where the gain over "plain" stops growing.
- Compare **width** 16 and 32 at **expansion factor** 4. In the page run the gain over plain was 0.0786 at width 16 and 0.088 at width 32. Check whether it keeps growing at 64.
- Check the "trained expanded" and "collapsed back" rows. They should match to every printed digit while their parameter counts differ by about 8x.

## Going further

ExpandNets applies this to CNNs by expanding a `k x k` conv into a `1 x 1`, a `k x k` and another `1 x 1`, all linear, and contracting after training. RepVGG trains parallel `3x3`, `1x1` and identity branches and merges them into one `3x3` conv for inference. pigeon already depends on the same algebra in its most common form: `server.py` fuses YOLOv8n when loading it, folding each batch-norm into the preceding conv (see Batch-norm folding). Training a specialist with expanded convs and contracting it before export could give a better small detector with the same deployed shape.
