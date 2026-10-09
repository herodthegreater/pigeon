## The idea

A classifier outputs a probability for every class. Those probabilities say which wrong answers are *nearly* right: "bird, maybe a sibling class, certainly not a vehicle." A one-hot label (1 at the true class, 0 elsewhere) carries none of that. Knowledge distillation (KD) trains a small *student* to reproduce a big *teacher*'s whole probability vector, using a *temperature* to make the small probabilities large enough to learn from.

## How it works

1. Train the teacher on labeled data.
2. Choose a *transfer set*: inputs on which to query the teacher. They need no labels.
3. For a batch, compute teacher logits `zt` and student logits `zs` (raw scores before softmax).
4. Soften both: `pt = softmax(zt / T)`, `ps = softmax(zs / T)`. Larger `T` flattens them so runner-up classes get visible mass.
5. KD loss: `T² · KL(pt ‖ ps) = T² · Σ pt · (log pt − log ps)`. KL divergence is zero when they agree.
6. Mix with cross-entropy (CE) on rows that have a label: `loss = alpha · KD + (1 − alpha) · CE`.
7. Use the student at `T = 1` at test time.

The gradient of the KL term with respect to `zs` is `(ps − pt) / T`; the `T²` factor keeps its size comparable across temperatures.

```python
pt = softmax(teacher(x) / T)
ls = log_softmax(student(x) / T)
loss = alpha * T*T * sum(pt * (log(pt) - ls)) + (1 - alpha) * ce(student(x), y)
```

## The code, line by line

`softmax` in `lab/train.py` divides by `T`, then subtracts each row's maximum (`z - z.max(1, keepdims=True)`); softmax ignores constant shifts, so this only stops `exp` from overflowing.

`KD.__init__` stores `T`, a `weight` (this is `alpha`) and a `kind`; Hinton KD is `kind="kl"`. For that kind `KD._teacher_probs` just returns `softmax(tl, T)`.

In `KD.__call__`, `ls = log_softmax(z, T)` is the student's log-probabilities, `ps = np.exp(ls)` its probabilities, and `lt = log_softmax(b.t_logits, T)` the teacher's log-probabilities, computed exactly from its logits rather than as `log` of a probability that may have underflowed to 0. The loss is `T * T * (pt * (lt - ls)).sum() / N`, the formula above averaged over `N` rows. The MLP in `lab/nn.py` has hand-written backprop, so each loss term also returns its gradient with respect to the logits, `dz = T * (ps - pt) / N`, and `{}` for (unused) hidden-layer gradients.

`distill` in `lab/common.py` wires it together; its default is `T = 1.0` (this page defaults to 4 so you can see the effect). `net(student, seed=seed)` builds a fresh student. `transfer_X(d, transfer)` returns the inputs: `"train"` is the 8k labeled set, `"pool"` adds 20k unlabeled inputs labeled `-1`. `FixedSource(X, y, ..., teacher=t)` runs the teacher once over every input and stores its logits. The terms are `KD(T=T, weight=alpha, ...)` plus `CE(weight=1 - alpha)` when `alpha < 1`; `CE` multiplies by `b.lab`, so unlabeled rows only feed the KD term. `train` fixes the number of optimizer steps, so the 28k pool does not win by training longer. `run_kd` compares this with `scratch` (labels only) at the same step count.

## What the simulations show

From `size_ladder` (3 seeds, 3000 steps, `T = 1`, test accuracy):

| student | labels only | KD, labeled inputs | KD, + pool |
|---|---|---|---|
| 2x8 | 0.4428 | 0.4428 | 0.4556 |
| 2x16 | 0.6637 | 0.6631 | 0.6714 |
| 2x32 | 0.8170 | 0.8180 | 0.8318 |
| 2x64 | 0.8820 | 0.8817 | 0.8957 |
| 2x128 | 0.9042 | 0.9040 | 0.9156 |

KD on the 8k labeled inputs matches labels only at every width; KD on the pool helps every width. The reason is the teacher: on its own training inputs it is 1.0 accurate with mean top probability 0.9994, so its `T = 1` output there is nearly the one-hot label. On the unlabeled pool the mean top probability is 0.9684, and that is where its soft labels carry new information.

Temperature makes things worse here. In `temperature_alpha` (2x16, labeled inputs), `T = 1` gives 0.6631-0.6639 for every `alpha`, and accuracy falls as `T` rises: 0.6399 at `T = 4`, 0.5815 at `T = 16` (`alpha = 1.0`). The teacher's logits are very spread out (median max-minus-min: 47.2), so a high `T` asks a tiny student to copy structure it cannot represent, and it becomes overconfident: NLL (mean negative log-probability of the true class) rises from 1.0176 at `T = 1` to 1.8606 at `T = 4`.

This page's runs (1000 steps, seed 0, pool): at width 16, labels only 0.5803, KD at `T = 4` 0.5551, KD at `T = 1` 0.5850. At width 64, labels only 0.8514, `T = 4` 0.8232, `T = 1` 0.8596. Single-seed gaps under about 0.015 are noise. The toy cannot say how a softer teacher, trained with heavy augmentation, would shift the best `T`.

## When it breaks

- A teacher that memorized its training set gives near one-hot targets there. Distill on inputs it has not seen.
- High temperatures hurt students too small to match the teacher's logits, and damage calibration.
- The student copies the teacher's mistakes.

## Try this

- At **student width** 16, set **temperature** to 1, 4, then 12. Accuracy should fall as `T` rises.
- At **temperature** 1, compare **student sees** = train vs pool. Only the pool should beat labels only.
- Set **weight on KD (vs. labels)** to 0.5 and see whether labels pull the student toward the labels-only score.

## Going further

Real models use the same loss with large unlabeled or augmented transfer sets and many epochs (see the function-matching page). Teacher logits are precomputed (as `FixedSource` does) or computed on the fly for augmented inputs. For a detector like the cockpit's YOLOv8n the outputs are per-box class scores and coordinates, so the KD term is applied per anchor or matched box; `lab/common.py:distill` is the closest analogue here.
