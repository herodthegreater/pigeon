## The idea

Most training tricks were invented for big networks that *overfit*, models that memorise the training set and do worse on new data. A small network usually has the opposite problem: it *underfits*, because it lacks the capacity to fit even the training data. This page tests four standard tricks on a 2-layer MLP of width 16 (820 parameters) and asks which of them help when capacity, not data, is the bottleneck:

- *Label smoothing*: train on targets that are 90% the true class and 10% spread evenly over all classes.
- *Mixup*: train on blends of two examples and their labels.
- *EMA of weights*: keep an exponential moving average of the weights during training and ship the average.
- *Longer training*: four times the steps.

## How it works

1. **Label smoothing** with `ε = 0.1`: `y_smooth = (1 - ε) * onehot(y) + ε / C`. The model is never pushed to put 100% on one class.
2. **Mixup**: pair each example with a random partner, draw `λ ~ Beta(1, 1)` (uniform on 0-1), and train on `x = λ x_a + (1 - λ) x_b` with target `λ y_a + (1 - λ) y_b`.
3. **EMA**: after every optimizer step, `shadow = 0.999 * shadow + 0.001 * W`. At the end, replace `W` by `shadow`.
4. **Longer**: the same cosine learning-rate schedule stretched over 4x the steps.

## The code, line by line

`CE` in `lab/train.py` is cross-entropy against `b.Y`, a matrix of target probabilities (one-hot, or soft for mixup). With smoothing, `Y = Y * (1 - self.ls) + self.ls / Y.shape[1]` mixes in the uniform distribution. `m = b.lab.astype(F)[:, None]` is 1 for rows that have a label and 0 for unlabeled ones, and `n = max(m.sum(), 1.0)` averages over labeled rows only. The loss is `-(m * Y * log_softmax(z)).sum() / n`. Its gradient with respect to the logits is the familiar `(softmax(z) - Y) * m / n`. Because `Y` can be any distribution, the same class serves plain, smoothed and mixed targets.

Mixup itself lives in `OnlineSource.batches`. `k = rng.permutation(len(j))` pairs examples within the batch, `lam = rng.beta(1.0, 1.0, size=(len(j), 1))` draws one `λ` per example, and `X` and `Y` are blended with it. `lab = lab & lab[k]` keeps a row labeled only if both parents were.

`fit` in `lab/train.py` is the training loop for every method in the lab. `opt = Adam(net.params() + extra, lr=lr, wd=wd)` is AdamW with weight decay `1e-4`. `total = max_steps or epochs * steps_per` fixes the budget, and each step sets `opt.lr = lr * 0.5 * (1 + np.cos(np.pi * min(step / total, 1.0)))`, a cosine decay from `lr` to 0 over the run. Each step runs forward, sums the gradients of all loss terms into `dz` (and `dhid` for hidden-layer losses), backpropagates, steps the optimizer and re-applies masks. Stretching `total` 4x is all "4x longer" does.

For EMA, `shadow = [p.copy() for p in net.params()] if ema else None` makes a copy of the **initial** weights. Each step does `s_ *= ema; s_ += (1 - ema) * p`, and at the end `p[...] = s_` overwrites the trained weights with the average. There is no warm-up or bias correction, so after `T` steps the random initial weights still make up `0.999 ** T` of the average: 0.368 after 1,000 steps and 0.0497 after 3,000.

`run_tricks` in `lab/methods.py` trains a baseline and the chosen trick for 1,000 steps each (4,000 for "4x longer"), seed 0.

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
| KD from teacher + pool | 0.6398 | 0.8059 |
| KD + pool + online mixup + EMA, 4x long | 0.7100 | 0.8607 |

Training longer is the only one of the four tricks with a clear gain. Label smoothing is within noise (the 2x16 standard deviations are about 0.008). Mixup and EMA hurt. That pattern fits underfitting: regularizers fight memorisation, and this model has none to fight. Distilling from the big teacher doesn't help either, unless it is combined with more steps and fresh inputs.

Page runs (1,000 steps, width 16): baseline 0.5803, label smoothing 0.5846, mixup 0.5397, 4x longer 0.6909, EMA 0.351. That EMA collapse is the initial-weights effect described above: 37% of the shipped average is still the random init.

The toy has 8,000 clean labels and no input noise, so it cannot show the regime where these regularizers usually help: big models on small or noisy datasets.

## When it breaks

- **EMA too slow for the run.** The decay must suit the step count: the average spans roughly `1 / (1 - ema)` steps. Start the shadow after a warm-up, or bias-correct it the way Adam corrects its moments.
- **Mixup on an underfitting model.** It makes the training task harder. That helps only when the model can already fit the plain task.
- **Label smoothing and distillation.** A smoothed model's probabilities carry less information about class similarity, which can make it a worse teacher. In the `teacher_smoothing` experiment, a 2x16 student distilled at T=4 reached 0.6398 from the plain teacher and 0.5738 from a teacher trained with smoothing 0.1, even though that teacher was more accurate (0.9216 vs 0.9139).
- **Comparisons at unequal budgets.** "4x longer" uses 4x the compute. Compare tricks at matched steps before concluding that a trick works.

## Try this

- Run **trick** = ema at **width** 16, then at **width** 64. The init's share of the average is the same 0.368 in both. See whether the wider net, which trains faster, suffers less.
- Switch **trick** between label smoothing and mixup at **width** 64. See whether a wider model, closer to fitting the task, changes the sign.
- Run **trick** = 4x longer at widths 4, 16 and 64. The gain from more steps should shrink as the model gets big enough to converge early.

## Going further

For large models these tricks are standard because those models are in the overfitting regime: label smoothing and mixup for image classifiers, and weight averaging (EMA, or stochastic weight averaging over checkpoints) for nearly everything. Ultralytics' trainer, which pigeon's `recover_job.py` drives, keeps an EMA of the weights by default and warms its decay up over early updates, avoiding the init problem shown here. For pigeon's recovery runs, STATUS.md lists "more train data, more epochs" as the next lever. On this toy, more steps was the one trick that reliably helped a capacity-limited model.
