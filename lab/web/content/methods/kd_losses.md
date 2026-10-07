## The idea

Hinton KD is one way to compare a student's outputs with a teacher's. The variants change what the student is punished for (reverse KL), skip probabilities (logit MSE), keep only the top answer (hard pseudo-labels), or model an API that reveals only the top-k probabilities. Each keeps a different part of the teacher's signal.

## How it works

Let `pt = softmax(zt / T)` and `ps = softmax(zs / T)` be the teacher's and student's tempered probabilities (logits are raw scores; temperature `T` flattens the probabilities).

1. **Forward KL** (`kl`): `T² · Σ pt · (log pt − log ps)`. Large where the teacher puts mass and the student does not, so the student spreads over every class the teacher finds plausible ("mode-covering").
2. **Reverse KL** (`rkl`): `T² · Σ ps · (log ps − log pt)`. Large where the student puts mass the teacher does not, so a weak student commits to the teacher's main answer ("mode-seeking").
3. **Logit MSE** (`mse`): `mean((zs − zt)²)`. No softmax, no temperature. Hinton showed KD approaches this as `T` grows, if logits are zero-mean.
4. **Hard** (`hard`): replace `pt` by a one-hot vector at the teacher's top class and use cross-entropy. Pseudo-labels carry nothing about the other classes.
5. **Top-k** (`topk`): take the teacher's `k` largest probabilities at `T = 1` (what an API would return), spread the remaining mass evenly over the other classes, then soften the result with `T` and apply forward KL.

## The code, line by line

`KD._teacher_probs` in `lab/train.py` builds the target. `T = 1.0 if self.kind == "hard" else self.T`: pseudo-labels ignore temperature. For `topk` it starts from `p = softmax(tl)` at `T = 1`; `thr = -np.sort(-p, 1)[:, self.k - 1:self.k]` is each row's k-th largest probability, `keep = p >= thr` marks the top k, and `rest = (1 - (p * keep).sum(1, keepdims=True)) / (~keep).sum(1, keepdims=True).clip(1)` is the leftover mass shared evenly; `np.where(keep, p, np.maximum(rest, 1e-12))` assembles it. Only then is temperature applied: `p = p ** (1 / T)` and renormalize, which equals `softmax(log p / T)`. Every other kind uses `p = softmax(tl, T)`. If a `group` matrix `G` is set, `p @ self.G` sums classes into a coarser label space ("bird / other" for a specialist). For `hard`, `onehot(p.argmax(1), ...)` keeps only the top class.

`KD.__call__` returns the loss and its gradient with respect to the student logits `z` (the MLP has hand-written backprop).

- `mse` is handled first: `r = z - t`, loss `(r ** 2).mean()`, gradient `2 * r / r.size`. `.mean()` divides by rows times classes; the KL terms divide by rows only.
- `hard`: `loss = -(pt * log_softmax(z)).sum() / N`, gradient `(softmax(z) - pt) / N`, the cross-entropy gradient.
- `rkl`: `per = (ps * (ls - lt)).sum(1, keepdims=True)` is the per-row reverse KL. Differentiating `Σ ps (ls − lt)` through the softmax gives `ps * (ls - lt - per)`; multiplied by `T` (from `T²` and the `1/T` inside the softmax) that is `dz`.
- `kl` and `topk` share the forward-KL branch: `loss = T * T * (pt * (lt - ls)).sum() / N`, `dz = T * (ps - pt) / N`. For `kl`, `lt = log_softmax(b.t_logits, T)` is exact; for the others it is `np.log(pt + 1e-12)`, since the target was built as probabilities.

`run_kd_losses` in `lab/methods.py` trains three students with the same seed and steps through `distill` (8k labeled plus 20k unlabeled inputs): labels only, forward KL at your **temperature**, and your `kind`.

## What the simulations show

`loss_zoo` (2x16 student, 3 seeds, 3000 steps, same inputs for every arm; `T = 1` unless named):

| loss | accuracy | NLL |
|---|---|---|
| labels only | 0.6637 | 1.0180 |
| hard pseudo-labels | 0.6712 | 0.9925 |
| KD T=1 | 0.6714 | 0.9909 |
| KD T=4 | 0.6398 | 1.8185 |
| reverse-KL T=4 | 0.6288 | 3.4385 |
| MSE on logits | 0.5086 | 2.6093 |
| top-3 probs only | 0.6710 | 0.9897 |
| top-1 prob + uniform rest | 0.6734 | 0.9957 |

NLL is the mean negative log-probability of the true class (lower is better). Hard pseudo-labels and `T = 1` KD are almost identical, because this teacher's `T = 1` outputs are close to one-hot (mean top probability 0.9684 on the unlabeled inputs). MSE is far worst: the teacher's logits span a median of 47.2 units, and a 2x16 network spends its capacity chasing that scale. At `T = 1`, top-3 and even top-1 plus a uniform rest match full KD, so almost nothing the student uses lives in the tail. Temperature is what hurts: KD and reverse KL at `T = 4` are both below labels only.

This page's runs (1000 steps, seed 0, width 16, page default `T = 4`): labels only 0.5803, forward KL 0.5534, reverse KL 0.5311, MSE 0.4255, hard 0.5820, top-1 0.5372, top-3 0.5516. At temperature 1, forward KL gave 0.5854. Hard labels and `T = 1` on top, MSE far below. A toy teacher this confident cannot show where the tail matters most, such as language models with thousands of plausible next tokens.

## When it breaks

- `mse` is not shift-invariant: adding a constant to every logit changes the loss but not the probabilities, so the student must match an offset that carries no information.
- `rkl` with a small student locks onto the top class and is badly calibrated (NLL 3.4385 above).
- `topk` guesses that the hidden mass is spread evenly. At high `T` that guess becomes a large, flat part of the target that the real teacher never expressed.
- `topk` keeps more than k classes when probabilities tie, because `keep = p >= thr`.

## Try this

- Set **loss** to `hard` and **temperature** to 1; the forward-KL row then also runs at `T = 1`. They should nearly coincide.
- Set **loss** to `topk` and **temperature** to 1, then sweep **k (for topk)** from 1 to 10. Find where accuracy stops changing; repeat at temperature 4.
- Set **loss** to `rkl` and **student width** to 64; a larger student can follow more than one mode.

## Going further

For language models the choice matters more: tens of thousands of classes per token, too many for a small student to cover. MiniLLM argues for reverse KL there, so the student commits to high-probability continuations. Hosted models often expose only top-k log-probabilities, which is what `topk` imitates. The grouping matrix in `_teacher_probs` is the closest analogue in this repo to distilling a detector into a one-class specialist.
