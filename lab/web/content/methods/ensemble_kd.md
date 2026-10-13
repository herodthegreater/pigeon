## The idea

Networks trained from different random starts make different mistakes, so averaging their predictions (an *ensemble*) usually beats any one of them, at the cost of running several models. Distilling the averaged output into one small student aims to keep part of that accuracy. There are two ways to average, giving different targets.

## How it works

1. Train $n$ teachers with the same recipe but different random seeds.
2. Run all of them on the transfer inputs and combine their outputs, either way:
3. **Average logits**: $\bar{z} = \frac{1}{n} \sum_i z_i$, then $\softmax(\bar{z} / T)$. This is the normalized *geometric* mean of the teachers' probabilities. Classes only keep mass where all teachers agree, so the target is sharper.
4. **Average probabilities**: $\bar{p} = \frac{1}{n} \sum_i \softmax(z_i)$, then use $\log \bar{p}$ as the logits. This is the *arithmetic* mean: a class keeps mass if any teacher gives it some, so the target preserves the teachers' disagreement.
5. Distill one student on the combined target with ordinary KD.

With $p_i = \softmax(z_i / T)$ the $i$-th teacher's probabilities, averaging logits gives, for class $c$, the normalized geometric mean

$$
\begin{aligned}
\softmax\!\Big( \frac{1}{n} \sum_{i=1}^{n} \frac{z_i}{T} \Big)_c
&= \frac{\big( \prod_{i=1}^{n} p_{i,c} \big)^{1/n}}{\sum_{c'} \big( \prod_{i=1}^{n} p_{i,c'} \big)^{1/n}}
\end{aligned}
$$

and averaging probabilities gives the arithmetic mean

$$
\bar{p}_c = \frac{1}{n} \sum_{i=1}^{n} p_{i,c}
$$

Feeding $\log \bar{p}$ back in as logits works because $\softmax(\log p) = p$ for a probability vector $p$.

Computing $\log \bar{p}$ naively fails when probabilities underflow to 0, so the code uses the log-sum-exp trick, here a log-mean-exp:

$$
\begin{aligned}
\log \frac{1}{n} \sum_{i=1}^{n} p_i
&= m + \log \frac{1}{n} \sum_{i=1}^{n} \exp\big( \log p_i - m \big) \\
m &= \max_i \log p_i
\end{aligned}
$$

taken per class, so $m$ is the largest $\log p_i$.

```python
logit_avg = mean([t.logits(X) for t in teachers])
prob_avg  = log_mean_softmax([t.logits(X) for t in teachers])   # log of mean probability
student = train_kd(X, targets=logit_avg or prob_avg, T=1)
```

## The code, line by line

`ensemble_teacher` in `lab/experiments/teacher.py`:

- `ts = [teacher(seed=i) for i in range(1 if QUICK else 8)]` loads up to eight 3x256 teachers that differ only in their seed (initial weights and data order), cached on disk.
- `X, y = transfer_X(d)` is the 8k labeled plus 20k unlabeled inputs.
- For `k` in 1, 2, 4, 8 it takes the first `k` teachers. `logit_avg = np.mean([t.logits(X) for t in sub], 0)` averages raw logits. `prob_avg = log_mean_softmax([t.logits(X) for t in sub])` averages probabilities in log space.
- `log_mean_softmax` in `lab/train.py`: `ls = np.stack([log_softmax(z) for z in logit_list])` holds each teacher's exact log-probabilities, `m = ls.max(0)` the largest per class, and `m + np.log(np.exp(ls - m).mean(0))` the log of the mean probability. The largest term inside `exp` is $\exp(0) = 1$, so nothing underflows.
- `test_probs` and `ens_acc` score the ensemble itself on the test set by averaging probabilities.
- For each way of combining, `train(net(STUDENT, s), FixedSource(X, y, ..., t_logits=tl), [KD(T=1)], seed=s)` trains a 2x32 student. Passing `t_logits` instead of a `teacher` makes `FixedSource` treat these as one teacher's logits. The labels `y` go unused with only a `KD` term.

`run_ensemble` in `lab/methods.py` does the same for up to 4 teachers, and adds a row for a student distilled from teacher 0 alone through `distill`; all students use $T = 1$.

## What the simulations show

`ensemble_teacher` (2x32 student, 3 seeds, 3000 steps, $T = 1$; accuracy, with NLL in brackets):

| teachers | ensemble itself | student from avg logits | student from avg probs |
|---|---|---|---|
| 1 | 0.9139 | 0.8318 (0.5112) | 0.8318 (0.5112) |
| 2 | 0.9222 | 0.8331 (0.5082) | 0.8336 (0.5072) |
| 4 | 0.9256 | 0.8329 (0.5052) | 0.8329 (0.5061) |
| 8 | 0.9263 | 0.8336 (0.5036) | 0.8344 (0.5031) |

NLL is the mean negative log-probability of the true class. With one teacher both averages are the same target, and the students agree exactly. The ensemble gains accuracy from one teacher to eight (0.9139 to 0.9263); the students move from 0.8318 to at most 0.8344, inside their seed spread (standard deviations 0.002-0.005). The two ways of averaging are indistinguishable here. Every student beats labels alone (0.8170), but that gain comes from distilling at all, not from the ensemble.

The teacher is not the bottleneck: one 3x256 teacher already knows far more than a 2x32 student can hold, so a better teacher has little to add.

This page's runs (1000 steps, seed 0) agree. With 3 teachers: student from teacher 0 alone 0.7455, from averaged logits 0.7465, from averaged probabilities 0.7471. With 4 teachers and probabilities: 0.7482. With 1 teacher, averaged logits give 0.7455, the same as the single-teacher row. Averaged probabilities give 0.7453, two test inputs away: with one teacher the target is the same distribution, but it goes through `log_mean_softmax`, and rounding at the last float digit nudges 1,000 steps of training.

The toy cannot show where ensembles help most: members that learned genuinely different features. Here they differ only by seed.

## When it breaks

- If the student, not the teacher, is the bottleneck, a better teacher does not produce a better student.
- Logit averaging assumes the teachers' logits are on comparable scales; one overconfident teacher dominates the average.
- Averaging probabilities with $\log(\bar{p} + \varepsilon)$ instead of log-sum-exp changes the target when teachers are very confident, especially at high temperature.
- Members trained the same way on the same data agree on most inputs, so the average adds little.

## Try this

- Set **teachers** to 1 and switch **combine by averaging** between `logits` and `probs`. `logits` matches the "teacher #0 alone" row exactly; `probs` lands within a few test inputs of it, from float rounding alone.
- With `probs`, step **teachers** from 1 to 4 and watch whether the student moves more than the ensemble does.
- Compare the "ensemble of n teachers" row with the student rows to see how much of the ensemble's accuracy is lost in distillation.

## Going further

Ensemble distillation is used to deploy one model in place of many, for example in speech recognition, Hinton et al.'s original setting. Allen-Zhu and Li argue that it works when different members learn different "views" of the data, which a single student can absorb. Diversity is the ingredient: members trained on different data, augmentations or architectures give a richer target than different seeds. `lab/experiments/teacher.py:ensemble_teacher` is the closest analogue in this repo.
