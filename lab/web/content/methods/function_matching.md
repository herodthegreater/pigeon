## The idea

Standard KD precomputes the teacher's answers on fixed inputs, so the student only matches the teacher at those points and eventually memorizes them. Beyer et al. treat distillation as matching a *function*: every step, make new inputs by augmenting real ones, ask the teacher about exactly those inputs, and train on its answers. The teacher sees the same augmented input as the student ("consistent"), and because new points keep coming, training can run far longer ("patient") without overfitting.

## How it works

1. Keep the teacher available during training, not just its stored outputs.
2. For each batch, draw real inputs and augment them. *Mixup* replaces each input by a random blend with another input in the batch: `x' = λ · x_i + (1 − λ) · x_j`, `λ ~ Beta(1, 1)` (uniform on 0 to 1). *Jitter* adds Gaussian noise: `x' = x + 0.3 · ε`.
3. Run the teacher on `x'` to get fresh targets.
4. Train the student on `x'` with KD against those targets.
5. Train for many more steps than you would on fixed targets.

```python
for step in range(steps):
    x = sample_batch(real_inputs)
    x = lam * x + (1 - lam) * x[perm]       # mixup, new lam each time
    loss = kd(student(x), teacher(x))       # teacher sees the same x
```

## The code, line by line

`OnlineSource` in `lab/train.py` is a *batch source*: its `batches()` yields training batches. `FixedSource` runs the teacher once in `__init__`; `OnlineSource` keeps the teacher and calls it inside `batches()`.

- `__init__` stores the inputs `X`, labels `y`, the `teacher`, the `aug` string and `noise` (default 0.3). `extra_X` appends unlabeled inputs with label `-1`.
- `batches()` shuffles indices with `rng.permutation(len(self.X))` and walks through them in chunks of `bs`.
- If `"mixup" in self.aug`: `k = rng.permutation(len(j))` pairs every row with another row of the same batch, `lam = rng.beta(1.0, 1.0, size=(len(j), 1))` draws one blend weight per row, and `X = lam * X + (1 - lam) * X[k]` mixes. `Y` (one-hot labels) is mixed the same way, and `lab = lab & lab[k]` marks a mixed row as labeled only if both parents were. These matter only with a cross-entropy term.
- If `"noise" in self.aug`: `X = X + rng.standard_normal(X.shape) * self.noise`.
- `out, hid, _ = self.teacher.forward(X)` queries the teacher on the augmented batch. This line is the whole method: the target belongs to the exact input the student trains on.
- With `teacher=None` it yields plain augmented batches (ordinary mixup training).

`run_function_matching` in `lab/methods.py` trains two 2x32 students for `STEPS * length` steps with `KD(T=1)`: one from `FixedSource(d.train.X, None, ..., teacher=t)` and one from `OnlineSource(d.train.X, d.train.y, ..., t, aug=...)`. Both see the same 8k real inputs; only the online one sees new points.

## What the simulations show

`function_matching` (2x32 student, 2 seeds, `T = 1`, 8k real inputs, test accuracy):

| steps | fixed targets | online mixup | online mixup + jitter | labels only |
|---|---|---|---|---|
| 1500 | 0.7761 | 0.7532 | 0.7388 | 0.7767 |
| 6000 | 0.8377 | 0.8379 | 0.8317 | 0.8364 |
| 24000 | 0.8378 | 0.8719 | 0.8642 | 0.8387 |

Short training favors fixed targets; online augmentation is a harder task and needs time. At 6000 steps they tie. At 24000, fixed targets have stopped improving and started overfitting: their NLL (mean negative log-probability of the true class) rises from 0.5133 to 0.6753, while online mixup keeps improving to 0.8719 with NLL 0.3897. The teacher scores 0.9139.

The effect is largest when real data is scarce. In `transfer_set`, a student distilled on only 500 real inputs reaches 0.5931; with online mixup on the same 500 it reaches 0.7382, and 0.7556 with mixup plus jitter. In `small_tricks`, combining the unlabeled pool, online mixup, weight averaging and 4x longer training gives a 2x32 student 0.8607, against 0.8170 for labels only.

This page's runs (seed 0, mixup), fixed against online: length 2 (2000 steps) 0.7939 against 0.7747; length 4, 0.8294 against 0.8213; length 8, 0.8416 against 0.8480; length 24 (24000 steps), 0.8331 against 0.8716. The crossover falls between 4x and 8x, and at 24x the fixed student has slipped back while the online one is still gaining. In 12 dimensions, mixup between two real points stays fairly close to the data; the toy cannot show how far this carries to images, where Beyer et al. trained for thousands of epochs.

## When it breaks

- Short budgets: online augmentation loses to fixed targets until training is long enough.
- Cost: every step pays for a teacher forward pass.
- Augmentations that leave the data distribution (too much jitter) give targets on inputs that never occur. Here jitter on top of mixup was slightly worse with 8k inputs, but helped with 500.
- Mixup with plain labels needs care: the blended label `λ · y_i + (1 − λ) · y_j` is a guess about what a blended input is. Querying the teacher on the blend replaces that guess with an answer.

## Try this

- Set **training length (x base)** to 2, 8, then 24, with **augmentation** `mixup`. Watch the online student overtake the fixed one.
- Compare **augmentation** `mixup`, `noise` and `mixup+noise` at length 8.
- At length 4, compare the fidelity of the two students (how often each one's top class matches the teacher's). Online targets should raise fidelity before they raise accuracy.

## Going further

Beyer et al. distilled large image classifiers into ResNet-50 with consistent mixup crops and very long schedules, and found the student kept improving long after a fixed-target run would plateau. For detectors, the same idea means running teacher and student on the same augmented image (same crop, flip, mosaic) rather than caching teacher boxes per original image. For language models, scoring freshly generated text on the fly plays the same role. `lab/train.py:OnlineSource` is the closest analogue in this repo.
