## The idea

Generating one token from a large model is limited by memory bandwidth: all the weights are read to produce a single token. Checking several tokens at once costs about the same as generating one. Speculative decoding lets a small *draft* model guess `γ` tokens cheaply. The large *target* model then scores all of them in one forward pass and keeps the longest acceptable prefix. A rejection rule makes the output distribution *exactly* the target model's, so quality is unchanged. Only speed depends on how well the draft imitates the target.

## How it works

Draft model `q`, target model `p`, current prefix `x`.

1. The draft generates `γ` tokens one by one: `x1 ~ q(· | x)`, `x2 ~ q(· | x, x1)`, and so on, saving each `q(xi)`.
2. The target runs once on `x, x1 .. xγ`, giving `p(· | ...)` at all `γ + 1` positions.
3. For `i = 1 .. γ`, accept `xi` with probability `min(1, p(xi) / q(xi))`.
4. At the first rejection, sample a replacement from `norm(max(0, p - q))` at that position and stop.
5. If all `γ` are accepted, sample one bonus token from `p` at position `γ + 1`.

**Why the output is exactly `p`.** The probability of emitting token `t` is `q(t) · min(1, p(t)/q(t)) + (1 - β) · max(0, p(t) - q(t)) / (1 - β)`, where `β = Σ min(p, q)` is the total acceptance probability. That simplifies to `min(p(t), q(t)) + max(0, p(t) - q(t)) = p(t)`.

**Speed.** The per-token acceptance rate is `α = Σ_t min(p(t), q(t)) = 1 - TV(p, q)`, where TV is total variation distance. The expected number of tokens per target pass is `(1 - α^(γ+1)) / (1 - α)`. With greedy decoding the rule reduces to "accept while `argmax q = argmax p`".

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
- Sampling settings (temperature, top-p) must be applied identically to `p` and `q` before the acceptance test, or the output is no longer exact.

## Going further

You need two autoregressive models with one tokenizer (for example a 1B draft and a 70B target from the same family) and a decoding loop with KV caches, which can be rolled back on rejection. Inference engines such as vLLM and `transformers` (assisted generation) implement it. Draft-free variants (Medusa, EAGLE) add small prediction heads to the target instead.

This is where distillation pays off: a draft distilled from the target accepts more tokens (DistillSpec). The closest analogue in this repo is the `fidelity` metric in `lab/train.py:evaluate`, top-1 agreement between student and teacher. It is exactly the greedy acceptance rate for one token. The `fidelity_gap` experiment (`lab/experiments/teacher.py`) studies how to raise it.
