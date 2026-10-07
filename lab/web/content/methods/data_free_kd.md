## The idea

Distillation needs inputs to show the teacher, but sometimes the original data is private, lost, or too large to ship. Two fallbacks: feed the teacher random noise and copy whatever it says, or *invert* the teacher, running gradient descent on the input, with the weights frozen, until the teacher confidently predicts a chosen class. The second produces inputs the teacher "recognizes", built from nothing but its weights.

## How it works

Teacher inversion:

1. Start with `n` random inputs `X ~ N(0, 1)` and assign each a random target class `c`.
2. Freeze the teacher. Compute its logits on `X` and the cross-entropy loss against `c`.
3. Backpropagate to the *input*: get `dLoss/dX` instead of gradients for the weights.
4. Add a small L2 penalty `l2 · ‖x‖² / 2` so the inputs stay near the scale of real data (the real inputs are standardized to mean 0, standard deviation 1 per feature).
5. Update `X` with Adam for a fixed number of steps.
6. Use the final `X` as the transfer set: query the teacher on it and distill with ordinary KD.

Noise baselines skip steps 2-5 and draw `X` from a Gaussian or a uniform distribution.

```python
X = randn(n, d); c = randint(C, n)
for _ in range(steps):
    g = grad_wrt_input(cross_entropy(teacher(X), c)) + l2 * X / n
    X = adam_step(X, g)
```

## The code, line by line

`invert_teacher` in `lab/experiments/transfer.py`:

- `X = rng.standard_normal((n, t.sizes[0]))` starts from noise in the teacher's input dimension (12).
- `c = rng.integers(0, t.sizes[-1], n)` and `Y = np.eye(...)[c]` give each input a random one-hot target, so the synthetic set is roughly class-balanced.
- `opt = Adam([X], lr=lr)` hands the *inputs* to the optimizer. The teacher's weights are never in any optimizer, so they cannot change.
- Each step, `z, _, cache = t.forward(X, cache=True)` runs the teacher and keeps the intermediate values backprop needs.
- `dz = (softmax(z) - Y) / n` is the gradient of mean cross-entropy with respect to the logits.
- `_, dX = t.backward(cache, dz)` backpropagates through the teacher. `MLP.backward` in `lab/nn.py` returns weight gradients and, as its second value, the gradient with respect to the input; only the latter is used.
- `opt.step([dX + l2 * X / n])` adds the gradient of the L2 prior and moves `X`.

`run_data_free` in `lab/methods.py` builds the transfer inputs for the chosen `source`: the first `n` real inputs (8k labeled, then the unlabeled pool), Gaussian noise, uniform noise on `[-3, 3]`, or `invert_teacher(t, n, steps=200)`. It then trains a 2x32 student with `FixedSource(X, None, ..., teacher=t)` (teacher logits precomputed, no labels) and `KD(T=4)`, and compares it with a student trained on the 8k labels.

## What the simulations show

`transfer_set` (2x32 student, 3 seeds, 3000 steps, KD at `T = 4`; fidelity is how often the student's top class matches the teacher's):

| transfer inputs | accuracy | fidelity |
|---|---|---|
| labels only, 8k (no teacher) | 0.8170 | 0.8309 |
| real: labeled set (8k) | 0.7996 | 0.8184 |
| real: + unlabeled pool (28k) | 0.8059 | 0.8251 |
| real: 500 only | 0.6612 | 0.6719 |
| fake: gaussian noise N(0,1) | 0.6601 | 0.6885 |
| fake: uniform noise [-3,3] | 0.6662 | 0.6926 |
| fake: teacher inversion | 0.6670 | 0.6938 |
| fake: inversion + 500 real | 0.6922 | 0.7167 |
| labels only, 500 (no teacher) | 0.5872 | 0.5947 |

With no real data, all three fake sources land near 0.66, about the same as distilling on 500 real inputs, and clearly better than training on 500 labels. Inversion does not beat plain noise. A look at 1000 inverted inputs explains part of it: their mean length is 2.12 against 3.39 for real test inputs (the L2 prior pulls them inward), and the teacher's mean top probability on them is 0.9983, so the targets are nearly one-hot and carry little information about the other classes. Gaussian noise, by contrast, already has the same mean length as real inputs (3.39) and the teacher assigns it to all 20 classes.

This page's runs (1000 steps, seed 0, `n` = 8000): labels only 0.7398; real 0.6805, Gaussian 0.5528, uniform 0.5532, inversion 0.5656. Inversion took about 98 seconds here, much longer than the other sources. The toy flatters noise: 12 standardized numbers are a small space, so random points often land near real data. Random images almost never look like photographs.

## When it breaks

- In high-dimensional inputs like images, pure noise covers almost none of the data distribution; the teacher sees only out-of-distribution inputs.
- Inversion with a plain cross-entropy target finds inputs where the teacher is *most* confident, which are the least informative for distillation.
- Too strong an L2 prior shrinks inputs toward zero; too weak lets them drift far from real data.
- Every inverted point costs a full forward and backward pass through the teacher per step.
- A teacher can be confident on inputs that look nothing like real data, so a high score proves nothing about realism.

## Try this

- Set **transfer inputs** to `gaussian`, then `uniform`, then `inversion`, at the same **number of inputs**. Compare with the labels-only row.
- Set **transfer inputs** to `real` and **number of inputs** to 500, then 8000, then 28000. This shows how much real data the fake sources are worth.
- Try `inversion` at 2000 inputs to see whether a smaller synthetic set loses much (it is also faster).

## Going further

For image classifiers, DeepInversion adds a much stronger prior: it matches the mean and variance of each batch-norm layer to the statistics stored in the teacher, plus image smoothness terms, and produces recognizable images. Later methods train a generator to produce inputs on which teacher and student disagree, so the transfer set targets the student's weak spots. This MLP has no batch-norm, so `lab/experiments/transfer.py:invert_teacher` is the closest analogue; the DeepInversion reference page covers the rest.
