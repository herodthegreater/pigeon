## The idea

Distillation does not need a bigger teacher. Furlanello et al. trained a network, then trained an identical, freshly initialized copy to match it, and found the copy was often more accurate than the original. Repeating this gives *generations*. The explanation usually offered is that the parent's soft outputs act as a regularizer: they tell the child which examples are ambiguous and which classes are related, so it fits the labels less rigidly.

## How it works

1. Generation 0: train a network of a fixed architecture on the labels.
2. Generation `g`: create a new network of the *same* architecture with new random weights.
3. Train it on `alpha · KD(parent) + (1 − alpha) · CE(labels)`, where KD compares temperature-softened outputs `softmax(z / T)` of child and parent, and CE (cross-entropy) is the ordinary label loss.
4. The child becomes the parent; repeat.
5. Evaluate every generation on held-out data. Optionally average several generations' predictions.

```python
parent = train_on_labels(arch, seed=0)
for g in range(1, G + 1):
    parent = distill(arch, teacher=parent, seed=100 * g, T=2, alpha=0.5)
```

## The code, line by line

`born_again` in `lab/experiments/teacher.py` runs the method for two architectures and two transfer sets.

- The outer loops cover `("2x32 (student-size)", STUDENT)` and `("3x256 (teacher-size)", [256, 256, 256])`, then `transfer` in `"train"` (8k labeled inputs) and `"pool"` (plus 20k unlabeled), then two seeds (`SEEDS[:2]`).
- Generation 0: `prev = scratch(hid, s)` trains the network on labels with the same 3000-step budget every later generation gets, for both sizes, so any change across generations comes from distillation, not from a different recipe.
- Each row records `ev(prev)`, the test metrics.
- For `g` in 1 to 3: `cur = distill(hid, prev, seed=100 * g + s, transfer=transfer, T=2, alpha=0.5)`. The seed changes each generation, so the child starts from new random weights and sees batches in a new order. `T=2, alpha=0.5` means half the loss is KD at a mild temperature and half is CE on the labels. With `transfer="pool"`, the unlabeled rows only get the KD half.
- `ev(cur, prev)` adds `fidelity`: how often the child's top class matches its parent's.
- `prev = cur` makes the child the next parent.
- `agg(rows, ["arch", "transfer", "generation"])` averages over seeds; `series` labels each line for the plot.

`run_born_again` in `lab/methods.py` runs one chain at the width you choose, on the labeled inputs, with `seed=100 * g` and the same `T` and `alpha`.

## What the simulations show

`born_again` (2 seeds, 3000 steps per generation, test accuracy):

| generation | 2x32, train | 2x32, pool | 3x256, train | 3x256, pool |
|---|---|---|---|---|
| 0 | 0.8163 | 0.8163 | 0.9168 | 0.9168 |
| 1 | 0.8016 | 0.7920 | 0.9193 | 0.9203 |
| 2 | 0.7951 | 0.7762 | 0.9203 | 0.9224 |
| 3 | 0.7921 | 0.7730 | 0.9180 | 0.9231 |

The small network loses accuracy every generation. The big one gains a little: with the pool it climbs every generation, on the labeled inputs alone it peaks at generation 2. Fidelity to the parent rises with each generation (2x32 on train inputs: 0.8194, 0.8383, 0.8481), so children become better copies of increasingly worse parents. A 2x32 network cannot fit this problem well, and its soft outputs encode its own errors; the KD half of the loss pulls the child toward them. The 3x256 parent is close to fitting the data, so its soft outputs mostly add calibration information: its NLL (mean negative log-probability of the true class) falls from 0.3735 to 0.3184 by generation 3 with the pool. The 3x256 gains are small next to their seed spread (up to 0.004), so treat them as a tendency.

`small_tricks` agrees for small models: self-distillation gave 0.6234 against a 0.6637 baseline for 2x16, and 0.8008 against 0.8170 for 2x32.

This page's runs (1000 steps, seed 0). Width 32: 0.7398, 0.7032, 0.6980. Width 128, three generations: 0.9010, 0.8920, 0.8938, 0.8904. Width 8: 0.3645, 0.3437, 0.3343, 0.3893. The last generation's jump at width 8 is a single seed on a very unstable model, not a trend. At this page's short budget, even width 128 gets no benefit.

## When it breaks

- Capacity-limited models: the parent's mistakes become the child's targets. Self-distillation helps models that already fit the data well.
- Short budgets: each generation is trained from scratch, so a child that has not converged is worse than its parent.
- Too much weight on KD (`alpha` near 1) with a weak parent copies its errors; the labels are what keep generations anchored.
- Gains are small, often within seed noise. Run several seeds before trusting a generation-to-generation difference.
- Give generation 0 the same recipe and budget as its children; otherwise the comparison measures the recipe.

## Try this

- Set **width** to 128 and **generations** to 3. Compare with width 32, where the model is capacity-limited.
- Set **width** to 4 or 8 and watch how unstable the generations are; that variation is the noise floor for any conclusion.
- At **width** 64, compare **generations** 1 and 4 to see whether later generations keep the first one's change or drift further.

## Going further

Born-again networks were reported on image classifiers (DenseNets and ResNets on CIFAR-100) that fit their training data well, the regime where `born_again` shows a gain for the 3x256 network. Practical variants distill within one training run: the model's own predictions from an earlier epoch, or its deeper layers teaching its shallower ones, avoiding a full retraining per generation. Allen-Zhu and Li argue self-distillation acts like an implicit ensemble of two models. In this repo, `lab/experiments/teacher.py:born_again` is the full method and the "self-distillation (born-again)" arm of `small_tricks` is the one-generation version.
