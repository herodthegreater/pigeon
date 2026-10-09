## The idea

Output distillation tells the student *what* the teacher answered. Its hidden layers also hold *how*: an internal representation of the input. FitNets adds a loss that makes a student hidden layer predict a teacher hidden layer. Similarity-preserving KD (SP) asks for less: inside each batch, examples the teacher represents as similar should look similar to the student too. SP never compares features directly, so the two networks can have any widths.

## How it works

FitNets hint, for a batch of `N` inputs:

1. Run teacher and student; take student hidden activations `hs` (shape `N × ds`) and teacher activations `ht` (shape `N × dt`). Hidden activations are a layer's outputs after its ReLU.
2. Map the student to teacher width with a trainable matrix `P` (`ds × dt`), since the widths differ.
3. Loss: `mean((hs · P − ht)²)`. Train `P` and the student together.

Similarity-preserving, for the same batch:

1. Build each network's Gram matrix, the table of dot products between every pair of examples: `A = h · hᵀ` (`N × N`).
2. Divide each row by its length: `G = A / ‖A_row‖`.
3. Loss: `Σ (Gs − Gt)² / N²`.

Both are added to the KD loss with a weight, and both need the teacher's hidden activations.

## The code, line by line

`Hint.__init__` in `lab/train.py` stores which student layer (`s_layer`) and teacher layer (`t_layer`) to align and creates `self.P`, a random `d_s × d_t` matrix scaled by `1 / sqrt(d_s)`. Its `params()` returns `[self.P]`, so `fit` adds `P` to the optimizer (`extra = [p for t in terms for p in t.params()]`).

`Hint.__call__`: `hs, ht = hid[self.s], b.t_hidden[self.t]` picks the two layers; `b.t_hidden` exists because `distill` passes `hidden=bool(extra_terms)` to `FixedSource`. `r = hs @ self.P - ht` is the residual, `n = r.size`. The gradient for `P` is `self.dP = self.w * hs.T @ (2 * r / n)`, and the gradient flowing into the student's hidden layer is `dh = self.w * (2 * r / n) @ self.P.T`. It returns `np.zeros_like(z)` for the logits and `{self.s: dh}`. In `lab/nn.py`, `MLP.backward` adds `dhid[i - 1]` when it reaches layer `i`, which injects `dh` exactly at hidden layer `s`. `fit` appends `t.dP` to the gradients in the same order it added `P` to the optimizer.

`SP._gram(h)` returns `A = h @ h.T`, the row norms `nrm`, and `A / nrm`. `SP.__call__` builds `Gs` and `Gt`, then `loss = ((Gs - Gt) ** 2).sum() / (N * N)`. `dG = 2 * (Gs - Gt) / (N * N)` is the gradient with respect to `Gs`. `dA = (dG - Gs * (Gs * dG).sum(1, keepdims=True)) / nrm` pushes it back through the row normalization: the subtracted term removes the part of `dG` that only changes a row's length, which normalization would undo anyway. Since `A = h hᵀ` uses `h` twice, `dh = (dA + dA.T) @ hs`.

`run_feature_kd` in `lab/methods.py` distills at `distill`'s default `T = 1` and uses `Hint(1, 2, w, 256, ...)` or `SP(1, 2, ...)`: student layer 1 (the last of its two hidden layers) against teacher layer 2 (the last of its three 256-wide layers). It trains labels only, KD, and KD plus your term.

## What the simulations show

`loss_zoo` (2x16 student, 3 seeds, 3000 steps, KD at `T = 1`; hint weight 0.05, SP weight 50):

| arm | accuracy |
|---|---|
| labels only | 0.6637 |
| KD T=1 | 0.6714 |
| KD + FitNets hint | 0.6719 |
| KD + similarity-preserving | 0.6666 |
| hint only (+labels) | 0.6223 |

Neither term adds to plain KD beyond seed noise (standard deviations here are 0.004-0.010), and a hint with labels but no KD is worse than labels alone. This page's runs (1000 steps, seed 0, width 16) agree: KD 0.5850; KD + SP at weight 50 gave 0.5800, at weight 1 gave 0.5846; KD + hint at weight 0.05 gave 0.5892, at weight 5 gave 0.5092.

Likely reason: forcing 16 student units to linearly predict 256 teacher units uses capacity the student needs for the task. The toy also lacks what makes these methods work on images: spatial feature maps, where "which region is active" is a cheap, transferable signal.

## When it breaks

- The loss weight must be tuned to the term's natural scale: here a hint works around 0.05 and SP around 50 (the page default).
- Too large a hint weight crowds out the task (0.5092 at weight 5 above).
- Which layers to pair is a guess; a hint at the wrong depth can fight the student's own solution.
- SP depends on the batch: small batches give a poor Gram matrix, and cost grows as `N²`.

## Try this

- Keep **extra loss** at `sp` and compare **loss weight** 50 (the scale used in `loss_zoo`) with 1 and 100.
- Set **extra loss** to `hint` and sweep **loss weight** over 0.05, 0.5 and 5. Watch for the point where accuracy drops below plain KD.
- Raise **student width** to 64 with `hint`; a wider student has capacity to spare.

## Going further

On convolutional networks, FitNets matches feature maps through a 1x1 convolution (the analogue of `P`), and attention transfer matches spatial activation maps. For detectors, feature distillation is usually restricted to regions near objects, since most of an image is background. In transformers, distilling attention maps and hidden states layer by layer is common for compressing language models. `lab/train.py:Hint` and `lab/train.py:SP` are the closest analogues here; `MLP.backward`'s `dhid` argument is how any such term plugs into training without autograd.
