## The idea

Hinton KD is one way to compare student and teacher outputs. The variants change what the student is punished for (reverse KL), skip probabilities (logit MSE), keep only the top answer (hard pseudo-labels), or model an API that reveals only top-k probabilities. Each keeps a different part of the signal.

## How it works

Let $p_t = \softmax(z_t / T)$ and $p_s = \softmax(z_s / T)$ be the teacher's and student's tempered probabilities over $C$ classes ($z_t$ and $z_s$ are the logits, raw scores; temperature $T$ flattens the probabilities).

1. **Forward KL** (`kl`): $T^2 \cdot \KL(p_t \,\|\, p_s)$. Punishes missing mass where the teacher has some, so the student covers every plausible class ("mode-covering").
2. **Reverse KL** (`rkl`): $T^2 \cdot \KL(p_s \,\|\, p_t)$. Punishes mass the teacher lacks, so a weak student commits to the main answer ("mode-seeking").
3. **Logit MSE** (`mse`): the mean of $(z_s - z_t)^2$. No softmax, no temperature. Hinton showed KD approaches this as $T$ grows, if logits are zero-mean.
4. **Hard** (`hard`): replace $p_t$ by a one-hot vector at the teacher's top class and use cross-entropy. No information about other classes.
5. **Top-k** (`topk`): take the teacher's $k$ largest probabilities at $T = 1$ (what an API would return), spread the remaining mass evenly over the other classes, then soften the result with $T$ and apply forward KL.

Written out, per input:

$$
\begin{aligned}
\mathcal{L}_{\text{kl}} &= T^2 \sum_{c=1}^{C} p_{t,c} \left( \log p_{t,c} - \log p_{s,c} \right) \\
\mathcal{L}_{\text{rkl}} &= T^2 \sum_{c=1}^{C} p_{s,c} \left( \log p_{s,c} - \log p_{t,c} \right) \\
\mathcal{L}_{\text{mse}} &= \frac{1}{C} \sum_{c=1}^{C} \left( z_{s,c} - z_{t,c} \right)^2 \\
\mathcal{L}_{\text{hard}} &= -\log \softmax(z_s)_{\hat{y}}, \qquad \hat{y} = \argmax_c z_{t,c}
\end{aligned}
$$

For top-k, let $p = \softmax(z_t)$ and $K$ be the set of its $k$ largest entries. The kept and filled-in target, and its softened version, are

$$
\tilde{p}_c =
\begin{cases}
p_c & c \in K \\[4pt]
\dfrac{1 - \sum_{c' \in K} p_{c'}}{C - \lvert K \rvert} & c \notin K
\end{cases}
\qquad
p_{t,c} = \frac{\tilde{p}_c^{\,1/T}}{\sum_{c'} \tilde{p}_{c'}^{\,1/T}} = \softmax\!\left( \frac{\log \tilde{p}}{T} \right)_c
$$

## The code, line by line

`KD._teacher_probs` in `lab/train.py` builds the target; `hard` ignores temperature. For `topk` it starts from `p = softmax(tl)` at $T = 1$; `thr = -np.sort(-p, 1)[:, self.k - 1:self.k]` is each row's k-th largest probability, `keep = p >= thr` marks the top k, and `rest = (1 - (p * keep).sum(1, keepdims=True)) / (~keep).sum(1, keepdims=True).clip(1)` is the leftover mass shared evenly, assembled by `np.where(keep, p, np.maximum(rest, 1e-12))`. Only then comes temperature: `p = p ** (1 / T)`, renormalized, which equals $\softmax(\log \tilde{p} / T)$. Other kinds use `p = softmax(tl, T)`. A `group` matrix `G` sums classes into a coarser label space (`p @ self.G`, the product $p G$). For `hard`, `onehot(p.argmax(1), ...)` keeps the top class.

`KD.__call__` returns the loss and its gradient with respect to the student logits `z`, averaged over the $N$ rows of a batch.

- `mse` is handled first: `r = z - t`, loss `(r ** 2).mean()`, gradient `2 * r / r.size`, that is $\frac{2}{NC}(z_s - z_t)$. `.mean()` divides by rows times classes; the KL terms divide by rows only.
- `hard`: `loss = -(pt * log_softmax(z)).sum() / N`, gradient `(softmax(z) - pt) / N`, the cross-entropy gradient $(\softmax(z_s) - e_{\hat{y}}) / N$ with $e_{\hat{y}}$ the one-hot vector.
- `rkl`: `per = (ps * (ls - lt)).sum(1, keepdims=True)` is the per-row reverse KL; differentiating through the softmax gives `dz = T * ps * (ls - lt - per) / N`:

$$
\frac{\partial \mathcal{L}_{\text{rkl}}}{\partial z_{s,c}} = T \, p_{s,c} \left( \log p_{s,c} - \log p_{t,c} - \KL(p_s \,\|\, p_t) \right)
$$

- `kl` and `topk` share the forward-KL branch: `loss = T * T * (pt * (lt - ls)).sum() / N`, `dz = T * (ps - pt) / N`, the gradient $T (p_s - p_t) / N$. For `kl`, `lt = log_softmax(b.t_logits, T)` is exact; otherwise `np.log(pt + 1e-12)`.

`run_kd_losses` in `lab/methods.py` trains three students through `distill` on the 28k inputs: labels only, forward KL at your **temperature**, and your `kind`.

## What the simulations show

`loss_zoo` (2x16 student, 3 seeds, 3000 steps, same inputs for every arm; $T = 1$ unless named):

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

NLL is the mean negative log-probability of the true class. Hard pseudo-labels and $T = 1$ KD nearly coincide: this teacher's $T = 1$ outputs are close to one-hot (mean top probability 0.9684 on unlabeled inputs). MSE is far worst: the teacher's logits span a median of 47.2 units, a scale a 2x16 network wastes capacity chasing. At $T = 1$, top-3 and even top-1 plus a uniform rest match full KD, so almost nothing the student uses lives in the tail. Temperature hurts: KD and reverse KL at $T = 4$ fall below labels only.

This page's runs (1000 steps, seed 0, width 16, page default $T = 4$): labels only 0.5803, forward KL 0.5551, reverse KL 0.5309, MSE 0.4134, hard 0.5838, top-1 0.5373, top-3 0.5525. At temperature 1, forward KL gave 0.5850. A toy teacher this confident cannot show where the tail matters most, such as language models with thousands of plausible next tokens.

## When it breaks

- `mse` is not shift-invariant: a constant $b$ added to every logit changes the loss but not the probabilities ($\softmax(z + b) = \softmax(z)$), so the student must match a meaningless offset.
- `rkl` with a small student locks onto the top class and is badly calibrated (NLL 3.4385 above).
- `topk` guesses that the hidden mass is spread evenly. At high $T$ that guess becomes a large, flat part of the target that the real teacher never expressed.
- `topk` keeps more than $k$ classes when probabilities tie, because `keep = p >= thr`.

## Try this

- Set **loss** to `hard` and **temperature** to 1 (the forward-KL row follows). They should nearly coincide.
- Set **loss** to `topk` and **temperature** to 1, then sweep **k (for topk)** from 1 to 10. Find where accuracy stops changing; repeat at temperature 4.
- Set **loss** to `rkl` and **student width** to 64; a larger student can follow more than one mode.

## Going further

For language models, with tens of thousands of classes per token, MiniLLM argues for reverse KL so a small student commits to high-probability continuations. Hosted models often expose only top-k log-probabilities, which `topk` imitates. The grouping matrix in `_teacher_probs` is this repo's analogue of distilling a detector into a one-class specialist.
