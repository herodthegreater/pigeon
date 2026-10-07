## The idea

Networks trained from different random starts make different mistakes, so averaging their predictions (an *ensemble*) usually beats any one of them, at the cost of running several models. Distilling the averaged output into one small student aims to keep part of that accuracy. There are two ways to average, giving different targets.

## How it works

1. Train `n` teachers with the same recipe but different random seeds.
2. Run all of them on the transfer inputs and combine their outputs, either way:
3. **Average logits**: `z = mean_i(z_i)`, then `softmax(z / T)`. This is the normalized *geometric* mean of the teachers' probabilities. Classes only keep mass where all teachers agree, so the target is sharper.
4. **Average probabilities**: `p = mean_i(softmax(z_i))`, then use `log p` as the logits. This is the *arithmetic* mean: a class keeps mass if any teacher gives it some, so the target preserves the teachers' disagreement.
5. Distill one student on the combined target with ordinary KD.

Feeding `log p` back in as logits works because `softmax(log p) = p` for a probability vector `p`.

Computing `log(mean(p))` naively fails when probabilities underflow to 0, so the code uses the log-sum-exp trick: `log mean_i p_i = m + log(mean_i exp(log p_i − m))`, with `m` the largest `log p_i`.

```python
logit_avg = mean([t.logits(X) for t in teachers])
prob_avg  = log_mean_softmax([t.logits(X) for t in teachers])   # log of mean probability
student = train_kd(X, targets=logit_avg or prob_avg, T=4)
```

## The code, line by line

`ensemble_teacher` in `lab/experiments/teacher.py`:

- `ts = [teacher(seed=i) for i in range(1 if QUICK else 8)]` loads up to eight 3x256 teachers that differ only in their seed (initial weights and data order), cached on disk.
- `X, y = transfer_X(d)` is the 8k labeled plus 20k unlabeled inputs.
- For `k` in 1, 2, 4, 8 it takes the first `k` teachers. `logit_avg = np.mean([t.logits(X) for t in sub], 0)` averages raw logits. `prob_avg = log_mean_softmax([t.logits(X) for t in sub])` averages probabilities in log space.
- `log_mean_softmax` in `lab/train.py`: `ls = np.stack([log_softmax(z) for z in logit_list])` holds each teacher's exact log-probabilities, `m = ls.max(0)` the largest per class, and `m + np.log(np.exp(ls - m).mean(0))` the log of the mean probability. The largest term inside `exp` is `exp(0) = 1`, so nothing underflows.
- `test_probs` and `ens_acc` score the ensemble itself on the test set by averaging probabilities.
- For each way of combining, `train(net(STUDENT, s), FixedSource(X, y, ..., t_logits=tl), [KD(T=4)], seed=s)` trains a 2x32 student. Passing `t_logits` instead of a `teacher` makes `FixedSource` treat these as one teacher's logits. The labels `y` go unused with only a `KD` term.

`run_ensemble` in `lab/methods.py` does the same for up to 4 teachers, with `KD(T=4)`, and adds a row for a student distilled from teacher 0 alone through `distill`, which uses `T = 1`.

## What the simulations show

`ensemble_teacher` (2x32 student, 3 seeds, 3000 steps, `T = 4`; accuracy, with NLL in brackets):

| teachers | ensemble itself | student from avg logits | student from avg probs |
|---|---|---|---|
| 1 | 0.9139 | 0.8059 (0.9307) | 0.8060 (0.9307) |
| 2 | 0.9222 | 0.8068 (0.9300) | 0.8065 (0.8794) |
| 4 | 0.9256 | 0.8078 (0.9279) | 0.8091 (0.8427) |
| 8 | 0.9263 | 0.8067 (0.9327) | 0.8101 (0.8222) |

NLL is the mean negative log-probability of the true class. With one teacher both averages are the same target, and the students agree. The ensemble gains accuracy from one teacher to eight; the students barely move. Probability averaging does change one thing: the student's NLL falls steadily, plausibly because the averaged target keeps the teachers' disagreement as softer probabilities, while logit averaging does not.

All of these students sit below a 2x32 distilled from one teacher at `T = 1` (0.8318 in `size_ladder`) and even below labels alone (0.8170), because this experiment still distills at `T = 4`, which hurts small students here (see the KD page). The teacher is not the bottleneck; the temperature and the student are.

This page's runs (1000 steps, seed 0) carry the same mismatch more visibly. With 3 teachers: student from teacher 0 alone 0.7465, from averaged logits 0.6797, from averaged probabilities 0.6825. With 4 teachers and probabilities: 0.6818. The single-teacher row is trained at `T = 1` and the ensemble rows at `T = 4`, so that gap is temperature, not ensembling. The fair comparison is **teachers** = 1, which gave 0.6781 for both ways of combining.

The toy cannot show where ensembles help most: members that learned genuinely different features. Here they differ only by seed.

## When it breaks

- If the student, not the teacher, is the bottleneck, a better teacher does not produce a better student.
- Logit averaging assumes the teachers' logits are on comparable scales; one overconfident teacher dominates the average.
- Averaging probabilities with `log(mean(p) + eps)` instead of log-sum-exp changes the target when teachers are very confident, especially at high temperature.
- Comparisons must use the same temperature for every arm; on this page the single-teacher row does not.

## Try this

- Set **teachers** to 1 and switch **combine by averaging** between `logits` and `probs`. The ensemble student should come out identical both ways.
- With `probs`, step **teachers** from 1 to 4 and compare each result with the 1-teacher ensemble row, not with "teacher #0 alone".
- Compare the "ensemble of n teachers" row with the student rows to see how much of the ensemble's accuracy is lost in distillation.

## Going further

Ensemble distillation is used to deploy one model in place of many, for example in speech recognition, Hinton et al.'s original setting. Allen-Zhu and Li argue that it works when different members learn different "views" of the data, which a single student can absorb. Diversity is the ingredient: members trained on different data, augmentations or architectures give a richer target than different seeds. `lab/experiments/teacher.py:ensemble_teacher` is the closest analogue in this repo.
