## The idea

Several networks trained from different random starting points make different mistakes, so averaging their predictions (an *ensemble*) is usually more accurate than any one of them. But an ensemble costs several models at inference time. Distilling the ensemble's averaged output into one small student aims to keep part of the ensemble's accuracy at the cost of one small model. There are two ways to average, and they give different targets.

## How it works

1. Train `n` teachers with the same recipe but different random seeds.
2. Run all of them on the transfer inputs and combine their outputs, either way:
3. **Average logits**: `z = mean_i(z_i)`, then `softmax(z / T)`. This is the normalized *geometric* mean of the teachers' probabilities. Classes only keep mass where all teachers agree, so the target is sharper.
4. **Average probabilities**: `p = mean_i(softmax(z_i))`, then use `log p` as the logits. This is the *arithmetic* mean: a class keeps mass if any teacher gives it some, so the target preserves the teachers' disagreement.
5. Distill one student on the combined target with ordinary KD.

Feeding `log p` back in as logits works because `softmax(log p) = p` for a probability vector `p`.

```python
logit_avg = mean([t.logits(X) for t in teachers])
prob_avg  = log(mean([softmax(t.logits(X)) for t in teachers]) + 1e-8)
student = train_kd(X, targets=logit_avg or prob_avg, T=4)
```

## The code, line by line

`ensemble_teacher` in `lab/experiments/teacher.py`:

- `ts = [teacher(seed=i) for i in range(1 if QUICK else 8)]` loads up to eight 3x256 teachers that differ only in their seed (initial weights and data order), cached on disk.
- `X, y = transfer_X(d)` is the 8k labeled plus 20k unlabeled inputs.
- For `k` in 1, 2, 4, 8 it takes the first `k` teachers. `logit_avg = np.mean([t.logits(X) for t in sub], 0)` averages raw logits. `prob_avg = np.log(np.mean([softmax(t.logits(X)) for t in sub], 0) + 1e-8)` averages probabilities and returns to log space.
- `test_probs` and `ens_acc` measure the ensemble itself on the test set by averaging probabilities, the usual way to score an ensemble.
- For each way of combining, `train(net(STUDENT, s), FixedSource(X, y, ..., t_logits=tl), [KD(T=4)], seed=s)` trains a 2x32 student. Passing `t_logits` instead of a `teacher` tells `FixedSource` to use these stored outputs as if a single teacher had produced them. The labels `y` are passed, but with only a `KD` term they are never used.

`run_ensemble` in `lab/methods.py` does the same for `n` up to 4 teachers and adds a row for a student distilled from teacher 0 alone.

## What the simulations show

`ensemble_teacher` (2x32 student, 3 seeds, 3000 steps, `T = 4`, test accuracy):

| teachers | ensemble itself | student from avg logits | student from avg probs |
|---|---|---|---|
| 1 | 0.9139 | 0.8059 | 0.8122 |
| 2 | 0.9222 | 0.8068 | 0.8115 |
| 4 | 0.9256 | 0.8078 | 0.8121 |
| 8 | 0.9263 | 0.8067 | 0.8127 |

The ensemble gains 0.0124 in accuracy from one teacher to eight; the student gains nothing from either kind of average. The teacher is not what limits a 2x32 student here (labels alone give it 0.8170 in `size_ladder`).

The probability column is higher even with **one** teacher, where there is nothing to average. That difference comes from the `+ 1e-8`, not from averaging. This teacher is extremely confident: a fraction 0.7655 of its class probabilities on the transfer inputs are below `1e-8`. Taking `log(p + 1e-8)` raises all of those to about `log(1e-8) ≈ −18.4`, which cuts the tail of the logits off. At `T = 4` that changes the target noticeably: the mean top probability of the softened target drops from 0.8231 to 0.731. The floor caps how far apart the logits can be, which plausibly gives a small student an easier target.

This page's runs (1000 steps, seed 0) show the same thing. With 3 teachers: teacher 0 alone 0.6781, logits averaged 0.6797, probabilities averaged 0.6993. With 1 teacher: logits 0.6781 (identical to the single-teacher row, as it should be), probabilities 0.6937. Most of the apparent ensemble gain is the floor. With 4 teachers, probabilities averaged gave 0.6994.

The toy cannot show the case where ensembles help most: teachers that each learned different features, which in a 12-dimensional world with one recipe and different seeds is limited.

## When it breaks

- If the student, not the teacher, is the bottleneck, a better teacher does not produce a better student.
- Logit averaging assumes the teachers' logits are on comparable scales; one overconfident teacher dominates the average.
- Small constants like `1e-8` change the target when teachers are very confident, especially at high temperature. Compare methods at equal clipping before crediting the averaging.
- Ensemble members trained the same way on the same data agree on most inputs, so the average adds little new information.

## Try this

- Set **teachers** to 1 and switch **combine by averaging** between `logits` and `probs`. Any difference is the `1e-8` floor, not ensembling.
- With `probs`, step **teachers** from 1 to 4 and see how much the student moves beyond the 1-teacher `probs` result.
- Compare the "ensemble of n teachers" row with the student rows to see how much of the ensemble's accuracy is lost in distillation.

## Going further

Ensemble distillation is used to deploy one model in place of many, for example in speech recognition, Hinton et al.'s original setting. Allen-Zhu and Li argue that it works when different members learn different "views" of the data, which a single student can absorb. Diversity is the ingredient: members trained on different data, augmentations or architectures give a richer target than different seeds. When teachers are very confident, clip or temperature-scale their logits explicitly, rather than through an incidental epsilon. `lab/experiments/teacher.py:ensemble_teacher` is the closest analogue in this repo.
