## The idea

Most training tricks were invented for big networks that *overfit*: they memorise the training set and do worse on new data. A small network usually *underfits*, because it lacks the capacity to fit even the training data. This page tests four of them on a 2-layer MLP whose bottleneck is capacity, not data. *Label smoothing* trains on targets that are 90% the true class and 10% spread over all classes. *Mixup* trains on blends of two examples and their labels. *EMA of weights* keeps an exponential moving average of the weights and ships the average. *Longer training* uses four times the steps.

## How it works

1. **Label smoothing** with `ε = 0.1`: `y_smooth = (1 - ε) * onehot(y) + ε / C`, so the model is never pushed to put 100% on one class.
2. **Mixup**: pair each example with a random partner, draw `λ ~ Beta(1, 1)` (uniform on 0-1), and train on `x = λ x_a + (1 - λ) x_b` with target `λ y_a + (1 - λ) y_b`.
3. **EMA**: after every optimizer step, `shadow = d * shadow + (1 - d) * W` with decay `d` close to 1. At the end, replace `W` by `shadow`.
4. **Longer**: the same cosine learning-rate schedule stretched over 4x the steps.

## The code, line by line

`CE` in `lab/train.py` is cross-entropy against `b.Y`, a matrix of target probabilities (one-hot, or soft for mixup). With smoothing, `Y = Y * (1 - self.ls) + self.ls / Y.shape[1]` mixes in the uniform distribution. `m = b.lab.astype(F)[:, None]` masks out unlabeled rows and `n` counts the labeled ones. The loss is `-(m * Y * log_softmax(z)).sum() / n`, with logit gradient `(softmax(z) - Y) * m / n`.

Mixup lives in `OnlineSource.batches`. `k = rng.permutation(len(j))` pairs examples within the batch, `lam = rng.beta(1.0, 1.0, size=(len(j), 1))` draws one `λ` per example, and `X` and `Y` are blended with it. `lab = lab & lab[k]` keeps a row labeled only if both parents were.

`fit` in `lab/train.py`, the lab's training loop, uses AdamW (`wd=1e-4`), fixes the budget with `total = max_steps or epochs * steps_per`, and on each step sets `opt.lr = lr * 0.5 * (1 + np.cos(np.pi * min(step / total, 1.0)))`, a cosine decay to 0. Stretching `total` 4x is all "4x longer" does.

For EMA, `shadow = [p.copy() for p in net.params()] if ema else None` copies the **initial** weights. Each step does `s_ *= ema; s_ += (1 - ema) * p`, and at the end `p[...] = s_` replaces the trained weights. There is no warm-up or bias correction, so after `T` steps the random init still makes up `ema ** T` of the average. The `small_tricks` experiment uses `ema=0.999` over 3,000 steps, leaving 0.0497 of the init in the shipped weights.

`run_tricks` in `lab/methods.py` trains a baseline and the chosen trick for 1,000 steps each (4,000 for "4x longer"), seed 0. Its EMA uses `ema=1 - 10 / STEPS`, which is 0.99 at this budget, so the init's share is about `0.99 ** 1000`, effectively zero.

## What the simulations show

From `small_tricks`, 3,000 steps, mean of 3 seeds, accuracy on 20 classes. The 3x256 teacher reaches 0.9139.

| trick | 2x16 | 2x32 |
|---|---|---|
| baseline (labels, CE) | 0.6637 | 0.8170 |
| label smoothing 0.1 | 0.6696 | 0.8230 |
| mixup | 0.6315 | 0.7989 |
| EMA of weights | 0.6481 | 0.8065 |
| 4x longer training | 0.7146 | 0.8405 |
| over-parameterize, then collapse | 0.7099 | 0.8420 |
| KD from teacher + pool | 0.6714 | 0.8318 |
| KD + pool + online mixup + EMA, 4x long | 0.7100 | 0.8607 |

Training longer is the only one of the four tricks with a clear gain. Label smoothing is within noise (2x16 standard deviations are about 0.008). Mixup and EMA hurt. That fits underfitting: regularizers fight memorisation, and this model has none to fight. Distilling from the teacher (T=1, on the labeled inputs plus 20,000 unlabeled ones) adds only 0.0077 for 2x16 and 0.0148 for 2x32.

Page runs (1,000 steps, width 16): baseline 0.5803, label smoothing 0.5846, mixup 0.5397, EMA 0.5796, 4x longer 0.6909. At width 64, EMA gives 0.8508 against 0.8514. With a decay matched to the run, EMA is neutral.

With 8,000 clean labels the toy cannot show where regularizers usually help: big models on small or noisy data.

## When it breaks

- **EMA decay too slow for the run.** The average spans roughly `1 / (1 - ema)` steps. With 0.999 over 1,000 steps, 37% of the shipped weights would still be the random init. Warm the decay up, or bias-correct it the way Adam corrects its moments.
- **Mixup on an underfitting model** makes a task it can't fit even harder.
- **Label smoothing and distillation.** A smoothed model's probabilities carry less class-similarity information. In `teacher_smoothing`, a 2x16 student distilled at T=4 reached 0.6398 from the plain teacher and 0.5738 from one trained with smoothing 0.1, though that teacher was more accurate (0.9216 vs 0.9139). At T=1 the gap nearly vanishes (0.6714 vs 0.6696).
- **Unequal budgets.** "4x longer" costs 4x the compute.

## Try this

- Switch **trick** between label smoothing and mixup at **width** 64, a model closer to fitting the task, and see whether the sign changes.
- Run **trick** = 4x longer at widths 4, 16 and 64. The gain should shrink as the model gets big enough to converge early.

## Going further

Big models overfit, so these tricks are standard there. Ultralytics' trainer, which the YOLOv8n cockpit's `recover_job.py` drives, keeps an EMA of the weights by default and ramps its decay up over early updates, avoiding the init problem above. STATUS.md lists "more train data, more epochs" as the cockpit's next lever. On this toy, more steps was the one trick that reliably helped a capacity-limited model.
