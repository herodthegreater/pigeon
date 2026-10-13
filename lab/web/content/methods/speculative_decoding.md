## The idea

Generating one token from a large model is limited by memory bandwidth: all the weights are read to produce a single token. Checking several tokens at once costs about the same as generating one. Speculative decoding lets a small *draft* model guess $\gamma$ tokens cheaply. The large *target* model then scores all of them in one forward pass and keeps the longest acceptable prefix. A rejection rule makes the output distribution *exactly* the target model's, so quality is unchanged. Only speed depends on how well the draft imitates the target.

## How it works

Draft model $q$, target model $p$, current prefix $x$.

1. The draft generates $\gamma$ tokens one by one: $x_1 \sim q(\cdot \mid x)$, $x_2 \sim q(\cdot \mid x, x_1)$, and so on, saving each $q(x_i)$.
2. The target runs once on $x, x_1, \dots, x_\gamma$, giving $p(\cdot \mid \dots)$ at all $\gamma + 1$ positions.
3. For $i = 1, \dots, \gamma$, accept $x_i$ with probability

$$
P(\text{accept } x_i) = \min\!\left( 1, \frac{p(x_i)}{q(x_i)} \right)
$$

4. At the first rejection, sample a replacement from $\operatorname{norm}(\max(0, p - q))$, the residual distribution, at that position and stop:

$$
p_{\text{res}}(t) = \frac{\max(0,\, p(t) - q(t))}{\sum_{t'} \max(0,\, p(t') - q(t'))}
$$

5. If all $\gamma$ are accepted, sample one bonus token from $p$ at position $\gamma + 1$.

**Why the output is exactly $p$.** A token $t$ is emitted either as an accepted draft or as a replacement after a rejection. Let $\beta = \sum_t \min(p(t), q(t))$ be the total acceptance probability. Since $\max(0, p - q) = p - \min(p, q)$, the residual's normalizer is

$$
\begin{aligned}
&\sum_t \max(0,\, p(t) - q(t)) \\
&\quad = \sum_t p(t) - \sum_t \min(p(t), q(t)) \\
&\quad = 1 - \beta ,
\end{aligned}
$$

so the probability of emitting token $t$ is

$$
\begin{aligned}
&P(\text{emit } t) \\
&\quad = q(t) \cdot \min\!\left( 1, \frac{p(t)}{q(t)} \right) \\
&\qquad + (1 - \beta) \cdot \frac{\max(0,\, p(t) - q(t))}{1 - \beta} \\
&\quad = \min(p(t), q(t)) + \max(0,\, p(t) - q(t)) \\
&\quad = p(t) .
\end{aligned}
$$

**Speed.** The per-token acceptance rate is one minus the total variation distance (TV) between $p$ and $q$:

$$
\begin{aligned}
\alpha &= \sum_t \min(p(t), q(t)) \\
&= 1 - \operatorname{TV}(p, q)
\end{aligned}
$$

The expected number of tokens per target pass is a geometric sum in $\alpha$:

$$
\begin{aligned}
\E[\text{tokens per pass}] &= \sum_{k=0}^{\gamma} \alpha^k \\
&= \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
\end{aligned}
$$

With greedy decoding the rule reduces to "accept while $\argmax q = \argmax p$".

```python
def speculative_step(x, draft, target, gamma):
    xs, qs = [], []
    for _ in range(gamma):
        dist = draft(x + xs); t = sample(dist); xs.append(t); qs.append(dist)
    ps = target(x, xs)                       # gamma + 1 distributions, one pass
    for i, t in enumerate(xs):
        if random() < min(1, ps[i][t] / qs[i][t]):
            continue
        return xs[:i] + [sample(normalize(maximum(0, ps[i] - qs[i])))]
    return xs + [sample(ps[gamma])]
```

## The code, line by line

This method has no code in the repo; the sketch above is the whole algorithm. `draft(prefix)` and `target(prefix, xs)` return next-token distributions. `target` must score every drafted position in one batched call, or nothing is saved. `normalize(maximum(0, ps[i] - qs[i]))` is the residual distribution from step 4.

## When it breaks

- Low acceptance rate: a draft that doesn't imitate the target wastes the target's verification work. Speed can fall below plain decoding.
- Compute-bound settings (large batches) erase the "verifying several tokens costs the same as one" premise.
- The draft and target must share a tokenizer and vocabulary.
- Sampling settings (temperature, top-p) must be applied identically to $p$ and $q$ before the acceptance test, or the output is no longer exact.

## Going further

You need two autoregressive models with one tokenizer (for example a 1B draft and a 70B target from the same family) and a decoding loop with KV caches, which can be rolled back on rejection. Inference engines such as vLLM and `transformers` (assisted generation) implement it. Draft-free variants (Medusa, EAGLE) add small prediction heads to the target instead.

This is where distillation pays off: a draft distilled from the target accepts more tokens (DistillSpec). The closest analogue in this repo is the `fidelity` metric in `lab/train.py:evaluate`, top-1 agreement between student and teacher. It is exactly the greedy acceptance rate for one token. The `fidelity_gap` experiment (`lab/experiments/teacher.py`) studies how to raise it.
